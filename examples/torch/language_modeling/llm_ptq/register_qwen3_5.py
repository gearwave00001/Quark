"""Custom Quark template for dense qwen3_5 (Huihui-Qwen3.8-27B-abliterated).

Mirrors the built-in qwen3_5_moe template minus MoE-specific exclusions/f2f
converters, with an explicit AWQConfig because qwen3_5 has no entry in AWQ_MAP.

Excludes match AMD's reference checkpoint (amd/Qwen3.8-27B-Quark-AWQ-MXFP4):
  lm_head + model.visual.* + mtp.*  (attention is NOT excluded -> MXFP4).
"""
from quark.torch.quantization.config.config import AWQConfig
from quark.torch.quantization.config.template import LLMTemplate

_QWEN3_5_AWQ = AWQConfig(
    model_decoder_layers="model.language_model.layers",
    scaling_layers=[
        {
            "prev_op": "input_layernorm",
            "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "inp": "self_attn.q_proj",
            "module2inspect": "self_attn",
        },
        {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
        {
            "prev_op": "post_attention_layernorm",
            "layers": ["mlp.gate_proj", "mlp.up_proj"],
            "inp": "mlp.gate_proj",
            "module2inspect": "mlp",
        },
        {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
    ],
)

template = LLMTemplate(
    model_type="qwen3_5",
    kv_layers_name=["k_proj", "v_proj"],
    q_layer_name="*q_proj",
    gate_up_layers_name=["gate_proj", "up_proj"],
    exclude_layers_name=["lm_head", "model.visual.", "mtp."],
    algorithm_configs={"awq": _QWEN3_5_AWQ},
    f2f_weight_converters=None,
)
LLMTemplate.register_template(template)
print("[register_qwen3_5] template registered:", template.model_type)


# --- Smoke-test support: dense qwen3 (e.g. Qwen/Qwen3-0.6B) -------------------
# Plain Qwen3ForCausalLM is a built-in template but has NO entry in AWQ_MAP, so
# `--quant_algo awq` raises NotImplementedError for it. Register an explicit
# AWQConfig mirroring the qwen3_5 recipe, tuned to the plain-qwen3 layout:
#   * decoder lives at `model.layers` (not `model.language_model.layers`)
#   * no visual tower / mtp -> only lm_head excluded
# This is inert for the real run (which uses qwen3_5); it only lets the tiny
# smoke model exercise the same sanitize->loss->scale-search->export path.
_QWEN3_AWQ = AWQConfig(
    model_decoder_layers="model.layers",
    scaling_layers=[
        {
            "prev_op": "input_layernorm",
            "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "inp": "self_attn.q_proj",
            "module2inspect": "self_attn",
        },
        {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
        {
            "prev_op": "post_attention_layernorm",
            "layers": ["mlp.gate_proj", "mlp.up_proj"],
            "inp": "mlp.gate_proj",
            "module2inspect": "mlp",
        },
        {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
    ],
)

_qwen3_template = LLMTemplate(
    model_type="qwen3",
    kv_layers_name=["*k_proj", "*v_proj"],
    q_layer_name="*q_proj",
    gate_up_layers_name=["gate_proj", "up_proj"],
    exclude_layers_name=["lm_head"],
    algorithm_configs={"awq": _QWEN3_AWQ},
    f2f_weight_converters=None,
)
LLMTemplate.register_template(_qwen3_template)
print("[register_qwen3_5] template registered:", _qwen3_template.model_type)
