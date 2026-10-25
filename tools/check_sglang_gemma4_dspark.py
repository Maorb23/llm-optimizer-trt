"""Config-only regression and architecture gate; no weights, GPU, or studies."""

import argparse
import tempfile
from pathlib import Path


def main():
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.utils.hf_transformers.config import get_config
    from transformers import AutoConfig, Gemma4TextConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft", required=True, help="Pinned draft snapshot directory"
    )
    args = parser.parse_args()

    # Real parser round trip with heterogeneous full/SWA shapes.
    config = Gemma4TextConfig(
        num_hidden_layers=6,
        head_dim=128,
        global_head_dim=256,
        num_key_value_heads=2,
        num_global_key_value_heads=4,
        attention_k_eq_v=True,
        architectures=["Gemma4ForCausalLM"],
    )
    expected = [
        (layer.head_dim, layer.num_key_value_heads) for layer in config.per_layer_config
    ]
    with tempfile.TemporaryDirectory() as directory:
        config.save_pretrained(directory)
        parsed = get_config(directory, trust_remote_code=False)
    actual = [
        (parsed.swa_head_dim, parsed.swa_num_key_value_heads)
        if kind == "sliding_attention"
        else (parsed.head_dim, parsed.num_key_value_heads)
        for kind in parsed.layer_types
    ]
    assert actual == expected, (actual, expected)
    assert not parsed.allow_global_per_layer_attribute_access
    print("PASS: mixed full/SWA shapes preserved")

    draft = str(Path(args.draft).resolve())
    original = AutoConfig.from_pretrained(draft, local_files_only=True)
    expected = [
        (layer.head_dim, layer.num_key_value_heads)
        for layer in original.per_layer_config
    ]
    parsed = get_config(draft, trust_remote_code=False, local_files_only=True)
    actual = [(parsed.head_dim, parsed.num_key_value_heads)] * parsed.num_hidden_layers
    assert parsed.layer_types == ["full_attention"] * 5
    assert actual == expected == [(512, 1)] * 5, (actual, expected)
    assert not parsed.allow_global_per_layer_attribute_access
    assert parsed.rope_parameters["full_attention"]["partial_rotary_factor"] == 0.25
    assert parsed.architectures == ["Gemma4DSparkModel"]
    print("PASS: draft shapes are 512/1; partial RoPE and architecture preserved")
    if not ModelRegistry.is_arch_supported("Gemma4DSparkModel"):
        raise SystemExit(
            "BLOCKED: installed SGLang has no Gemma4DSparkModel implementation. "
            "Parser repair is insufficient; do not start a study."
        )
    print("PASS: draft architecture registered (GPU inference still unverified)")


if __name__ == "__main__":
    main()
