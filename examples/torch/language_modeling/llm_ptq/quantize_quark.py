#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from quark.common.profiler import GlobalProfiler, ProfileStep
from quark.common.utils.log import ScreenLogger
from quark.torch import (
    LLMTemplate,
    ModelQuantizer,
    RuntimeOptions,
    export_gguf,
    export_onnx,
    export_safetensors,
    import_model_from_safetensors,
    load_params,
    save_params,
)
from quark.torch.export.api import _move_quantizer_to_dict
from quark.torch.quantization.config.config import load_quant_algo_config_from_file
import register_qwen3_5  # noqa: E402  (registers qwen3_5 template)
from quark.torch.utils import TPDeviceManager

# TODO: Using sys.path.append is bad practice.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quark.contrib.llm_eval import eval_model
from quark.torch.utils.llm import (
    check_compatibility_before_quantization,
    get_calib_dataloader,
    get_model,
    get_tokenizer,
    maybe_save_preprocessors,
    preprocess_for_quantization,
)

logger = ScreenLogger(__name__)

# set CUDA_VISIBLE_DEVICES for profiling
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# The code below demonstrates how to register custom model templates and
# quantization schemes. If you need to add support for a new model architecture
# or define custom quantization configurations, uncomment and modify this section.
#
# To use:
#   1. Uncomment the code below
#   2. Modify the templates and/or schemes to match your model's architecture and/or quantization scheme
#   3. Run quantize_quark.py with your custom --quant_scheme name if new quantization schemes are registered
#

# from quark.torch.quantization.config.config import (
#     Int8PerTensorSpec,
#     QLayerConfig,
# )

# # --- Custom Model Templates ---
# # Define templates for model architectures not in the built-in list.
# # Model: internlm/internlm2-chat-7b
# internlm2_template = LLMTemplate(
#     model_type="internlm2",
#     kv_layers_name=["*wqkv"],
#     q_layer_name="*wqkv",
#     exclude_layers_name=["lm_head"],
# )
# LLMTemplate.register_template(internlm2_template)
# print(f"[INFO]: Registered template '{internlm2_template.model_type}'")

# # --- Custom Quantization Schemes ---
# # Define custom quantization schemes using Quark's public QuantizationSpec classes.
# # These schemes can then be used via --quant_scheme <scheme_name>.
# # INT8 weight-only quantization
# int8_wo_scheme = QLayerConfig(weight=Int8PerTensorSpec().to_quantization_spec())
# LLMTemplate.register_scheme("int8_wo", config=int8_wo_scheme)
# print(f"[INFO]: Registered quantization scheme 'int8_wo'")


def _get_hf_model_config(model_dir: str) -> dict:
    """Read config.json from the model directory without loading the model."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)


def _build_quant_config(args: argparse.Namespace, model_config_type: str):
    """Build quant_config from args and model_config_type (shared by normal and file-to-file paths)."""
    if model_config_type not in LLMTemplate.list_available():
        error_msg = (
            f"\n[ERROR]: Model type '{model_config_type}' is not supported.\n\n"
            f"Available templates: {LLMTemplate.list_available()}\n\n"
            f"To add support for this model, uncomment and modify the 'Custom Model Templates'\n"
            f"section at the top of this file to register a template for '{model_config_type}'.\n"
        )
        raise ValueError(error_msg)
    template = LLMTemplate.get(model_config_type)

    # Load algorithm configs from files if provided
    algo_configs = {}
    if args.quant_algo_config_file is not None:
        for algo_name, algo_config_file in args.quant_algo_config_file:
            algo_configs[algo_name] = load_quant_algo_config_from_file(algo_config_file)
            print(f"[INFO]: Loaded algorithm configuration for {algo_name} from {algo_config_file}.")

    # Build layer_config if --layer_quant_scheme is provided
    layer_config = {}
    if args.layer_quant_scheme is not None:
        for layer_info in args.layer_quant_scheme:
            layer_name = layer_info[0]
            layer_scheme = layer_info[1]
            layer_config[layer_name] = layer_scheme

    quant_config = template.get_config(
        scheme=args.quant_scheme,
        algorithm=args.quant_algo,
        kv_cache_scheme=args.kv_cache_dtype,
        min_kv_scale=args.min_kv_scale,
        layer_config=layer_config,
        attention_scheme=args.attention_dtype,
        exclude_layers=args.exclude_layers,
        algo_configs=algo_configs if algo_configs else None,
    )
    quant_config.keep_prequantized_layers = not args.no_keep_prequantized_layers
    return quant_config


def _strip_gdn_cpu_norm_artifacts(model):
    """[R19j EXPORT-FIX] Drop forward-time-only GDN '_cpu_norm' submodules that alias 'norm.weight'.

    transformers' Qwen3.5 GDN forward lazily builds self._cpu_norm on the CPU branch
    (modeling_qwen3_5.py) with `self._cpu_norm.weight = self.norm.weight`, leaving two
    state-dict keys sharing one storage (e.g. layers.61.linear_attn.{norm,_cpu_norm}.weight).
    save_pretrained's remove_tied_weights_from_state_dict rejects them because only
    lm_head.weight is declared tied. norm keeps the AWQ-updated weight, so dropping
    _cpu_norm loses nothing. Returns the list of stripped module names.
    """
    removed = []
    for name, mod in list(model.named_modules()):
        if getattr(mod, "_cpu_norm", None) is not None:
            del mod._cpu_norm
            removed.append(name)
    if removed:
        print(
            "[EXPORT-FIX] stripped %d GDN _cpu_norm artifact(s): %s"
            % (len(removed), ", ".join(removed)),
            flush=True,
        )
    return removed


def _restore_vision_bf16(model, src_dir):
    """[R19j VISION-PARITY] Replace quantized vision-tower linears with bf16 nn.Linear loaded
    from the source checkpoint, matching the AMD reference which EXCLUDES the entire vision
    tower from AWQ (kept in bf16). Only touches 2-D bf16 '*.weight' tensors under model.visual.*
    (the attn.qkv/proj + mlp.linear_fc1/fc2 projections); conv/norm params (1-D/3-D) are left
    as-is. The matching 1-D bf16 '*.bias' is restored alongside its weight (R19j post-mortem:
    an earlier version built these with bias=False, silently dropping all 110 vision linear
    biases => systematically distorted image features / hallucinated vision). Returns the
    number of restored linears."""
    import glob
    import torch.nn as _nn
    from safetensors import safe_open

    src_w = {}
    src_b = {}
    for fn in sorted(glob.glob(os.path.join(src_dir, "*.safetensors"))):
        with safe_open(fn, framework="pt") as f:
            for k in f.keys():
                if "visual" not in k:
                    continue
                sl = f.get_slice(k)
                if k.endswith(".weight"):
                    if len(sl.get_shape()) == 2 and sl.get_dtype() == "BF16":
                        src_w[k] = f.get_tensor(k)  # materialize only the vision tensors
                elif k.endswith(".bias"):
                    if len(sl.get_shape()) == 1 and sl.get_dtype() == "BF16":
                        src_b[k] = f.get_tensor(k)
    print("[VISION-PARITY] %d bf16 vision linear weights (+%d biases) found in source %s"
          % (len(src_w), len(src_b), src_dir), flush=True)
    restored = 0
    n_bias = 0
    for k, w in src_w.items():
        pname = k[: -len(".weight")]
        parent_name, _, child = pname.rpartition(".")
        try:
            parent = model.get_submodule(parent_name)
        except AttributeError:
            print("[VISION-PARITY] skip (no module) " + pname, flush=True)
            continue
        cur = getattr(parent, child, None)
        # Replace only genuine Quark quantized linears (QuantLinear carries
        # _weight_quantizer_inv); leave plain nn.Linear / Embedding / Parameter
        # (e.g. model.visual.pos_embed) untouched.
        if cur is None or not (hasattr(cur, "_weight_quantizer_inv") or "Quant" in type(cur).__name__):
            continue
        b = src_b.get(pname + ".bias")
        newlin = _nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        newlin.weight = _nn.Parameter(w.detach().clone(), requires_grad=False)
        if b is not None:
            newlin.bias = _nn.Parameter(b.detach().clone(), requires_grad=False)
            n_bias += 1
        setattr(parent, child, newlin)
        restored += 1
    print("[VISION-PARITY] restored %d quantized vision linears -> bf16 nn.Linear (%d with bias; %d bf16 candidates in source)"
          % (restored, n_bias, len(src_w)), flush=True)
    return restored


def _patch_config_exclude_for_bf16_linears(out_dir):
    """[R19j VISION-PARITY] Post-export fix: sync config.json's quantization_config.exclude
    with the bf16 (unquantized) vision-tower + MTP linears actually present on disk.

    The exporter derives `exclude` from the QConfig baked into the guard snapshot
    (["lm_head"]), which does NOT reflect (a) the vision linears we restored to bf16 via
    _restore_vision_bf16, nor (b) the mtp.* weights that patch_missing_weights appends as
    bf16 after the shard write. A config-driven loader treats `exclude` as the source of
    truth for which linears stay unquantized, so it would otherwise instantiate those
    modules as QuantLinear and fail against their bf16 weights (no weight_scale on disk).
    We derive the set from the ACTUAL exported shards (BF16 keys ending '.weight' under
    *.visual.* [2-D, module-name form] or top-level mtp.* [any dim, both module-name and
    tensor-name forms]), making the result self-consistent with what is on disk. Mirrors
    the AMD reference, whose exclude lists the entire vision tower (111 visual entries)
    plus the mtp.* weights. embed_tokens is deliberately NOT added (AMD leaves it out too)."""
    import glob
    import json
    from safetensors import safe_open

    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.isfile(cfg_path):
        print("[CONFIG-EXCLUDE] no config.json at " + str(out_dir) + "; skipping", flush=True)
        return
    with open(cfg_path) as fp:
        cfg = json.load(fp)
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        print("[CONFIG-EXCLUDE] no quantization_config block; leaving config untouched", flush=True)
        return

    bf16_lin = set()
    for fn in sorted(glob.glob(os.path.join(out_dir, "*.safetensors"))):
        with safe_open(fn, framework="pt") as f:
            for k in f.keys():
                # Unquantized tensors shipped as bf16: vision tower (parity mode) + MTP
                # head (restored bf16 by patch_missing_weights). Not blanket-all-bf16:
                # embed_tokens stays out of exclude, matching the AMD reference.
                if not (k.endswith(".weight") and (".visual." in k or k.startswith("mtp."))):
                    continue
                sl = f.get_slice(k)
                if sl.get_dtype() != "BF16":
                    continue
                if ".visual." in k:
                    # Vision: 2-D linears only, module-name form (AMD's visual style).
                    if len(sl.get_shape()) == 2:
                        bf16_lin.add(k[: -len(".weight")])
                else:
                    # MTP: every bf16 param, in BOTH module-name and tensor-name forms
                    # (AMD lists the tensor-name form; emitting both so either matching
                    # style in a config-driven loader hits).
                    bf16_lin.add(k[: -len(".weight")])
                    bf16_lin.add(k)
    excl = set(qc.get("exclude") or [])
    added = sorted(bf16_lin - excl)
    qc["exclude"] = sorted(excl | bf16_lin)
    with open(cfg_path, "w") as fp:
        json.dump(cfg, fp, indent=2)
    print(
        "[CONFIG-EXCLUDE] %d bf16 unquantized linear(s) on disk (vision+mtp); added %d new exclude entr%s (%s)"
        % (
            len(bf16_lin),
            len(added),
            "y" if added else "ies",
            ", ".join(added[:6]) + ("..." if len(added) > 6 else ""),
        ),
        flush=True,
    )


def main(args: argparse.Namespace) -> None:
    if args.revision is not None and os.path.isdir(args.model_dir):
        raise ValueError(
            f"The argument --revision {args.revision} is not supported using a local directory: {args.model_dir}"
        )
    elif not os.path.isdir(args.model_dir):
        args.model_dir = snapshot_download(args.model_dir, revision=args.revision)

    # Initialize global profiler
    profiler = GlobalProfiler(output_path=os.path.join(args.output_dir, "quark_profile.yaml"))

    # File-to-file quantization mode: bypass model loading, calibration and quantization,
    # directly quantize safetensors files shard-by-shard and export.
    if args.file2file_quantization:
        print("\n[INFO]: File-to-file quantization mode enabled.")
        hf_model_config = _get_hf_model_config(args.model_dir)
        architectures = hf_model_config.get("architectures", [])
        model_config_type = hf_model_config.get("model_type", architectures[0] if architectures else None)
        quant_config = _build_quant_config(args, model_config_type)

        print("\n[INFO]: Quantizing safetensors shards directly (file-to-file) ...")

        weight_converters = LLMTemplate.get(model_config_type).f2f_weight_converters
        if weight_converters:
            logger.info(f"Applying {len(weight_converters)} weight converter(s) for model type '{model_config_type}'")

        with profiler.scope(ProfileStep.FILE_TO_FILE_QUANTIZATION):
            quantizer = ModelQuantizer(quant_config)
            quantizer.direct_quantize_checkpoint(
                pretrained_model_path=args.model_dir,
                save_path=args.output_dir,
                weight_converters=weight_converters,
                keep_excluded_layers_as_original_model_state=args.keep_excluded_layers_as_original_model_state,
            )

        print(f"[INFO]: File-to-file quantization output saved to {args.output_dir}")
        return

    # 1. Define original model
    model = None
    # [R19j RESUME-GUARD] Fast export+eval retry from a pre-export snapshot:
    # load the fully-AWQ'd frozen model and skip load/calib/AWQ below.
    _resumed = False
    if getattr(args, "resume_guard", None):
        print("\n[RESUME-GUARD] loading pre-export model from " + str(args.resume_guard), flush=True)
        import cloudpickle as _cp

        with open(args.resume_guard, "rb") as _gf:
            model = _cp.load(_gf)
        args.skip_quantization = True
        _resumed = True
        print("[RESUME-GUARD] loaded; skipping load/calib/AWQ -> export + PPL eval", flush=True)
    # Load the pretrained model for quantization or for reload later (the old way).
    if not _resumed and (not args.model_reload or args.import_model_dir):
        print("\n[INFO]: Loading model ...")

        # We currently use CPU memory to load large models because GPU memory is typically smaller.
        # The model will be dispatched to different GPUs based on the total number of GPUs specified by torchrun --nproc-per-node.
        # TODO:
        # The current method results in high CPU memory consumption due to multiple copies of the same model.
        # We plan to address this in the future by implementing a more efficient way to dispatch the model to devices.
        if args.use_tp:
            device = "cpu"
        else:
            device = args.device

        try:
            with profiler.scope(ProfileStep.MODEL_LOADING):
                model, _ = get_model(
                    args.model_dir,
                    args.data_type,
                    device,
                    args.multi_gpu,
                    args.multi_device,
                    args.model_attn_implementation,
                    trust_remote_code=args.trust_remote_code,
                )
        except torch.OutOfMemoryError as exception:
            if torch.cuda.device_count() <= 1:
                raise torch.OutOfMemoryError(
                    f"Out of memory error when loading the model {args.model_dir}. Only one device visible; this model does not fit on a single GPU."
                ) from exception
            elif not args.multi_gpu:
                raise torch.OutOfMemoryError(
                    f"Out of memory error when loading the model {args.model_dir}. Consider using `--multi_gpu` as {torch.cuda.device_count()} devices are available."
                ) from exception
            else:
                raise torch.OutOfMemoryError(
                    f"Out of memory error when loading the model {args.model_dir}. The model does not fit even with `--multi_gpu` across {torch.cuda.device_count()} devices. Consider using file-to-file quantization with `--file2file_quantization`, or make more GPU memory available."
                ) from exception

        # Check model compatibility with current Transformers version
        print("\n[INFO]: Checking model compatibility ...")
        check_compatibility_before_quantization(model, raise_on_error=False)

    if args.use_tp:
        TPDeviceManager.tp_mesh_init()

    # 2. (Optional) Reload quantized model
    if not _resumed and args.params_load:
        print("\nRestore quantized model from json and safetensors file ...")
        model = load_params(model, json_path=args.json_path, safetensors_path=args.safetensors_path)
        args.skip_quantization = True
    elif not _resumed and args.model_reload:
        # Use import_model_dir if provided (separate quantized checkpoint), otherwise model_dir is the checkpoint itself.
        reload_dir = args.import_model_dir or args.model_dir
        print("\nRestore quantized model from hf_format safetensors file ...")
        model = import_model_from_safetensors(
            model=model,
            model_dir=reload_dir,
            multi_device=args.multi_device,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.model_attn_implementation,
            device="cpu" if args.use_tp else args.device,
            multi_gpu=args.multi_gpu,
        )
        args.skip_quantization = True

    architectures = getattr(model.config, "architectures", None) or []
    model_type = (
        model.config.model_type
        if hasattr(model.config, "model_type")
        else (architectures[0] if architectures else None)
    )
    tokenizer = get_tokenizer(
        args.model_dir, max_seq_len=args.seq_len, model_type=model_type, trust_remote_code=args.trust_remote_code
    )

    # Detect multimodality from the model config's sub-modality keys instead of a
    # hardcoded model_type whitelist — every HF VLM/ALM config exposes one of these
    # (vision_config / audio_config / image_config / video_config).
    multimodal = any(
        getattr(model.config, k, None) is not None
        for k in ("vision_config", "audio_config", "image_config", "video_config")
    )

    if args.use_tp:
        if TPDeviceManager._tp_mesh is not None:
            _move_quantizer_to_dict(model.model)

            device = TPDeviceManager._device
            tp_mesh = TPDeviceManager._tp_mesh

            model.tensor_parallel(tp_mesh)
            model.to(device)
        else:
            warnings.warn(
                "Quark tensor parallelism is not initialized properly. Please check the torchrun settings.",
                UserWarning,
                stacklevel=2,
            )
            return

    # 3. Define calibration dataloader(still need this step for weight only and dynamic quantization in Quark for current version.)
    print("\n[INFO]: Loading dataset ...")

    # When the model is small, accelerate will place it on the last device
    main_device = model.device if args.multi_gpu or args.multi_device else args.device

    with profiler.scope(ProfileStep.DATASET_LOADING):
        calib_dataloader = get_calib_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_calib_data=args.num_calib_data,
            seqlen=args.seq_len,
            device=main_device,
        )

    # 4. Quantization
    if not args.skip_quantization:
        preprocess_for_quantization(model)

        architectures = getattr(model.config, "architectures", None) or []
        model_config_type = (
            model.config.model_type
            if hasattr(model.config, "model_type")
            else (architectures[0] if architectures else None)
        )

        quant_config = _build_quant_config(args, model_config_type)

        if getattr(args, "kv_cache_post_rope", False):
            if hasattr(quant_config, "kv_cache_post_rope"):
                quant_config.kv_cache_post_rope = True
            else:
                warnings.warn(
                    "--kv_cache_post_rope specified but quant_config has no 'kv_cache_post_rope' field; flag ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # In-place replacement of model modules with quantized versions
        quantizer = ModelQuantizer(quant_config, args.multi_device)
        _vis = getattr(getattr(model, "model", None), "visual", None); (_vis.to("cpu") if _vis is not None else None); torch.cuda.empty_cache(); print("[VISION-OFFLOAD] visual=" + ("found->CPU" if _vis is not None else "NONE"), flush=True)
        model = quantizer.quantize_model(model, calib_dataloader)
        args.exclude_layers = quantizer.config.exclude

        # After quantization, freeze models - moving from soft weights that are quantized on the fly
        # to e.g. `QuantLinear.weight` actually holding the fake quantized weights.
        runtime_options = None
        if args.enable_native_inference:
            runtime_options = RuntimeOptions(
                native_linear_mode=args.native_linear_mode,
            )
        model = quantizer.freeze(model, runtime_options=runtime_options)

    if args.model_export is not None:
        # Save pre-processors (tokenizer, image processor, etc.).
        export_dir = Path(args.output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        maybe_save_preprocessors(
            args.model_dir,
            export_dir,
            trust_remote_code=args.trust_remote_code,
        )

        if args.custom_mode != "quark" and args.export_weight_format == "fake_quantized":
            raise ValueError("Exporting with 'fake_quantized' only supports custom_mode=quark")

        # Export option 1: hugging-face safetensors format
        if "hf_format" in args.model_export:
            print("\n[INFO]: Exporting hugging face format safetensors...")
            # [EXPORT-GUARD-V2] snapshot frozen model for fast export retry
            # (skipped on --resume_guard: the guard already exists and is identical)
            if not _resumed:
                try:
                    import os as _os
                    # Fallback location: $QUARK_WORK_DIR (drive scripts set this), else cwd.
                    _gp = _os.environ.get(
                        "QUARK_EXPORT_GUARD_PATH",
                        _os.path.join(
                            _os.environ.get("QUARK_WORK_DIR", _os.getcwd()),
                            "quark_pre_export_model.pt"))
                    try:
                        import cloudpickle as _cp
                        with open(_gp, "wb") as _gf:
                            _cp.dump(model, _gf)
                        print("[EXPORT-GUARD] saved (cloudpickle) -> " + _gp, flush=True)
                    except ImportError:
                        torch.save(model, _gp)
                        print("[EXPORT-GUARD] saved (torch) -> " + _gp, flush=True)
                except Exception as _e:
                    print("[EXPORT-GUARD] save failed: " + repr(_e), flush=True)
            # [R19j VISION-PARITY] optionally restore vision-tower linears to bf16 (AMD parity)
            if getattr(args, "restore_vision_bf16", None):
                _restore_vision_bf16(model, args.restore_vision_bf16)
            # [R19j EXPORT-FIX] drop GDN _cpu_norm alias artifacts before save_pretrained
            _strip_gdn_cpu_norm_artifacts(model)
            with profiler.scope(ProfileStep.EXPORT_HF_SAFETENSORS), torch.no_grad():
                export_safetensors(
                    model=model,
                    output_dir=args.output_dir,
                    custom_mode=args.custom_mode,
                    weight_format=args.export_weight_format,
                    pack_method=args.pack_method,
                )
                # [R19j VISION-PARITY] sync config.json exclude with the bf16 vision linears
                # (parity model only; the vision-quantized resume leaves exclude=["lm_head"]).
                if getattr(args, "restore_vision_bf16", None):
                    _patch_config_exclude_for_bf16_linears(args.output_dir)

        # Export option 2: onnx
        if "onnx" in args.model_export:
            print("\n[INFO]: Exporting onnx graph...")
            with profiler.scope(ProfileStep.EXPORT_ONNX), torch.inference_mode():
                batch_iter = iter(calib_dataloader)
                input_args = next(batch_iter)
                if "uint4" in args.quant_scheme or "int4" in args.quant_scheme:
                    uint4_int4_flag = True
                else:
                    uint4_int4_flag = False

                export_onnx(
                    model=model, output_dir=args.output_dir, input_args=input_args, uint4_int4_flag=uint4_int4_flag
                )

        # Export option 3: gguf
        if "gguf" in args.model_export:
            print("\n[INFO]: Exporting gguf model...")
            with profiler.scope(ProfileStep.EXPORT_GGUF), torch.inference_mode():
                export_gguf(model, output_dir=args.output_dir, model_type=model_type, tokenizer_path=args.model_dir)

    if args.torch_compile:
        print("\n[INFO]: Calling PyTorch 2 torch.compile...")
        # Note: The model after torch.compile may not be able to export to other format
        model = torch.compile(model)

    if args.params_save:
        save_params(model, model_type=model_type, export_dir=args.save_dir)

    if not args.skip_evaluation:
        print("\n[INFO]: Evaluating ...")

        with profiler.scope(ProfileStep.MODEL_EVALUATION):
            args.use_ppl_eval_model = True
            eval_model(
                args,
                model,
                main_device,
                save_metrics_to_csv=args.save_metrics_to_csv,
                output_dir=args.metrics_output_dir,
                multimodal=multimodal,
            )

    if args.use_tp:
        TPDeviceManager.tp_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # Argument for model
    parser.add_argument(
        "--model_dir",
        help="Specify where the HuggingFace model is. This example support Llama, OPT models",
        required=True,
    )
    parser.add_argument(
        "--revision",
        help="HuggingFace Hub revision (branch, tag, or commit) to download when --model_dir is a Hub model ID. "
        "Triggers snapshot_download so all files come from the same revision.",
        default=None,
    )
    parser.add_argument("--device", help="Device for running the quantizer", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--multi_gpu",
        nargs="?",
        const="auto",
        default=None,
        choices=["auto", "balanced"],
        help="Enable multi-GPU mode. 'auto': default accelerate device map. "
        "'balanced': use auto-adjusted device map for better GPU memory balance.",
    )
    parser.add_argument(
        "--model_attn_implementation",
        help="The attention implementation to use in the model",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--multi_device",
        action="store_true",
        help="we allow you to use this mode to run a model quantization that exceeds the size of your gpu memory if you use args.multi_gpu and still run into OOM "
        "now it only supports thr common quantization without algorithms, please note that this can lead to very slow quantization.",
    )

    # Argument for calibration dataset
    parser.add_argument(
        "--dataset",
        help="Dataset for calibration",
        default="pileval",
        choices=[
            "pileval",
            "wikitext",
            "cnn_dailymail",
            "pileval_for_awq_benchmark",
            "wikitext_for_gptq_benchmark",
            "HuggingFaceH4/ultrachat_200k",
            "ScienceQA",
        ],
    )
    parser.add_argument(
        "--data_type", help="Datatype of the model", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument("--seq_len", type=int, help="Sequence length of data", default=512)
    parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument("--num_calib_data", help="Number of samples for calibration.", type=int, default=512)

    # Argument for quantization
    parser.add_argument("--skip_quantization", action="store_true")
    parser.add_argument(
        "--file2file_quantization",
        action="store_true",
        help="Enable file-to-file quantization mode. Quantizes safetensors shards directly without loading the full model into memory. "
        "Bypasses model loading, calibration, and standard quantization flow. Requires --model_export hf_format.",
    )

    parser.add_argument(
        "--quant_scheme",
        help="Quantization scheme to use. Supported schemes: all built-in schemes and custom schemes registered."
        "For the built-in schemes and their detailed configuration, see https://quark.docs.amd.com/latest/pytorch/user_guide_config_for_llm.html. "
        "To register custom schemes, please uncomment and modify the 'Custom Quantization Schemes' section at the top of this file.",
        choices=LLMTemplate.get_supported_schemes(),
        default=None,
        type=str,
    )

    parser.add_argument(
        "--layer_quant_scheme",
        action="append",
        nargs=2,
        metavar=("PATTERN", "QUANT_SCHEME"),
        help="Directly specify a quantization scheme for layers matching the given pattern. "
        "Can be repeated for multiple patterns. "
        "Example: --quant_scheme int4_wo_128 --layer_quant_scheme lm_head int8 "
        "(results in lm_head using int8 while other layers use int4_wo_128). "
        "Supports wildcards: --layer_quant_scheme '*down_proj' fp8",
    )

    parser.add_argument(
        "--kv_cache_dtype", "--kv_cache_quant_scheme", help="KV Cache dtype.", default=None, choices=["fp8", None]
    )

    parser.add_argument("--min_kv_scale", help="Minimum value of KV Cache scale.", type=float, default=0.0)
    parser.add_argument(
        "--kv_cache_post_rope",
        action="store_true",
        help="If set, quantize KV cache after RoPE (inside cache) instead of at k_proj/v_proj outputs.",
    )
    parser.add_argument(
        "--attention_dtype", help="The dtype of attention quantization.", type=str, default=None, choices=["fp8"]
    )
    parser.add_argument(
        "--quant_algo",
        default=None,
        type=lambda s: s.split(","),
        metavar="alg1,alg2",
        help="Comma-separated list of algorithms. Options include awq, gptq, smoothquant, rotation.",
    )
    parser.add_argument(
        "--quant_algo_config_file",
        action="append",
        nargs=2,
        metavar=("ALGO_NAME", "CONFIG_FILE"),
        help="Specify a configuration file for a specific quantization algorithm. "
        "Can be repeated for multiple algorithms. "
        "Example: --quant_algo_config_file awq ./awq_config.json --quant_algo_config_file gptq ./gptq_config.json "
        "(provides custom config files for AWQ and GPTQ algorithms).",
    )

    parser.add_argument(
        "--exclude_layers",
        type=str,
        nargs="*",  # Allows to pass a list of strings
        default=None,  # Default is None to allow model-specific layer exclusion
        help='List of layers to exclude from quantization. Default depends on model type. Usage: `--exclude_layers "*down_proj*" "*31.fc*" "*k_proj"`. To avoid excluding layers at all, simply use `--exclude_layers` without any argument.',
    )
    parser.add_argument(
        "--enable_native_inference",
        action="store_true",
        help="Enable native inference layer conversion during freeze().",
    )
    parser.add_argument(
        "--native_linear_mode",
        type=str,
        default="auto",
        choices=["auto", "fp8_per_tensor"],
        help="Native linear implementation mode used when native inference is enabled.",
    )

    # Argument for reloading
    parser.add_argument("--model_reload", help="safetensors or pth model reload", action="store_true")
    parser.add_argument(
        "--import_model_dir",
        help="[Deprecated: use --model_dir instead] directory of hf or quark model, override model directory for reload, if not provided, --model_dir is used.",
    )
    parser.add_argument("--params_load", help="Model parameters load", action="store_true")
    parser.add_argument(
        "--resume_guard",
        help="[R19j] Path to a pre-export model snapshot (cloudpickled fully-AWQ'd, frozen, "
        "QuantLinear-ready model). When set, skips model load / calibration / AWQ and goes "
        "straight to export + PPL eval. Fast retry after an export-stage crash.",
        default=None,
    )
    parser.add_argument(
        "--restore_vision_bf16",
        help="[R19j] Source HF checkpoint dir. With --resume_guard, restores the vision-tower "
        "linear layers to bf16 from this checkpoint before export, matching the AMD reference "
        "which excludes the vision tower from AWQ. Produces the ~19.8GB parity model.",
        default=None,
    )
    parser.add_argument("--json_path", help="Specify the path of saved json file")
    parser.add_argument("--safetensors_path", help="Specify the path of saved safetensors file")

    # Argument for export
    parser.add_argument(
        "--model_export",
        help="Model export format",
        default=None,
        action="append",
        choices=[None, "onnx", "hf_format", "gguf"],
    )
    parser.add_argument(
        "--custom_mode",
        help="When selecting `--custom_mode awq` or `--custom_mode fp8`, this legacy argument allows to export FP8 and AWQ models in the custom format they were exported with with quark<1.0, with custom config saved in the config.json, and config checkpoint format (AWQ uses `qzeros`, `qweight`, transposed `scales`).",
        default="quark",
        type=str,
        choices=["quark", "awq", "fp8"],
    )
    parser.add_argument("--torch_compile", help="Model torch compile", action="store_true")
    parser.add_argument(
        "--pack_method", type=str, help="Pack method for awq_export", default="reorder", choices=["order", "reorder"]
    )
    parser.add_argument("--output_dir", default="exported_model")
    parser.add_argument(
        "--export_weight_format",
        type=str,
        help="Whether to export weights compressed or uncompressed",
        default="real_quantized",
        choices=["fake_quantized", "real_quantized"],
    )
    parser.add_argument(
        "--no_keep_prequantized_layers",
        action="store_true",
        help="Force dequantization of excluded pre-quantized layers to bf16/fp16 on export. "
        "By default (flag omitted), such layers are preserved in their original quantized format "
        "(converted to Quark format); unsupported formats fall back to dequantization with a warning.",
    )
    parser.add_argument(
        "--keep_excluded_layers_as_original_model_state",
        action="store_true",
        help="File-to-file mode only: keep already-quantized excluded layers (e.g. FP8 attention "
        "in the official DeepSeek-V4 checkpoint) in their original on-disk format instead of "
        "dequantizing them to bf16/fp16. Off by default; only enable for source checkpoints whose "
        "quantization_config declares the excluded layers' format.",
    )

    # Argument for saving
    parser.add_argument("--params_save", help="Model parameters save", action="store_true")
    parser.add_argument(
        "--save_dir",
        help="Directory to save model parameters as safetensors or pth, in the case when --params_save is used.",
        default="model_params",
    )

    # Argument for evaluation
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument(
        "--evaluation_dataset",
        help="Dataset for evaluation",
        default="wikitext",
        choices=["wikitext", "wikitext_gpt_oss_120b", "wikitext_gpt_oss_20b"],
    )
    parser.add_argument("--use_ppl_eval_model", action="store_true")
    parser.add_argument("--save_metrics_to_csv", action="store_true")
    parser.add_argument("--metrics_output_dir", default="metrics_output_dir", help="Output path of csv with metrics.")
    parser.add_argument(
        "--tasks",
        default=None,
        type=str,
        metavar="task1,task2",
        help="Comma-separated list of task names or task groupings to evaluate on.",
    )
    parser.add_argument("--use_ppl_eval_for_kv_cache", action="store_true")
    parser.add_argument(
        "--ppl_eval_for_kv_cache_context_size",
        type=int,
        help="Context size used in PPL evaluation for KV cache.",
        default=1024,
    )
    parser.add_argument(
        "--ppl_eval_for_kv_cache_sample_size",
        type=int,
        help="Sample size used in PPL evaluation for KV cache.",
        default=512,
    )
    parser.add_argument(
        "--ppl_eval_for_kv_cache_patch_size",
        type=int,
        help="Patch size used in PPL evaluation for KV cache.",
        default=None,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Batch size used for evaluation. Acceptable values are 'auto', 'auto:N' or N, where N is a positive integer. Default is `1`.",
    )
    parser.add_argument(
        "--max_eval_batch_size",
        type=int,
        default=64,
        metavar="P",
        help="Maximal batch size to try with `--batch_size auto`.",
    )
    parser.add_argument(
        "--num_eval_data",
        help="Number of samples for evaluation. The default value is -1, which means the entire dataset is used for evaluation.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=None, metavar="N", help="Number of examples in few-shot context"
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt.",
    )
    parser.add_argument("--use_mlperf_rouge", action="store_true")
    parser.add_argument("--eval_data_dir", help="Dataset for evaluation", type=str, default=None)
    parser.add_argument(
        "--use_tp", action="store_true", help="Enable tensor parallelism exclusively for model evaluation."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--trust_remote_code",
        action="store_true",
        dest="trust_remote_code",
        help="Enable execution of custom model code from the Hub (use only with repositories you fully trust).",
    )
    group.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable execution of custom model code from the Hub (safer, recommended if unsure).",
    )
    parser.set_defaults(trust_remote_code=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.layer_quant_scheme is not None:
        for layer_info in args.layer_quant_scheme:
            if len(layer_info) != 2:
                raise ValueError(
                    f"Invalid --layer_quant_scheme argument: {layer_info}. "
                    f"Expected exactly 2 values (PATTERN, QUANT_SCHEME), but got {len(layer_info)}."
                )

    if args.quant_algo_config_file is not None:
        for algo_config in args.quant_algo_config_file:
            if len(algo_config) != 2:
                raise ValueError(
                    f"Invalid --quant_algo_config_file argument: {algo_config}. "
                    f"Expected exactly 2 values (ALGO_NAME, CONFIG_FILE), but got {len(algo_config)}."
                )
            algo_name, config_file = algo_config
            if not os.path.isfile(config_file):
                raise ValueError(
                    f"Configuration file '{config_file}' for algorithm '{algo_name}' does not exist. "
                    f"Please provide a valid config file path."
                )

    main(args)
