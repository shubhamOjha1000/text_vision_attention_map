"""
Qwen2.5-VL adapter for the RAW attention-score tests.

Thin wrapper over the generic `probe_hf_vlm.make_hf_vlm_output` -- the exact same
eager-patch capture verified on SmolVLM2, pointed at a Qwen2.5-VL checkpoint.
Fills the SAME `ProbeOutput`, so every test in `test_raw_attention.py` runs
unchanged.

Default model: **Qwen2.5-VL-7B-Instruct in 4-bit** (~5 GB) so it fits a single
16 GB GPU (T4 / g4dn). Override with the `QWEN_VL_ID` env var or `model_id` arg,
e.g. `Qwen/Qwen2.5-VL-72B-Instruct` on a large multi-GPU node (there,
`load_in_4bit=True` + `device_map="auto"` shards ~40 GB across the GPUs).
"""

import importlib.util
import os
from typing import Optional

import torch

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "probe_hf_vlm", os.path.join(_here, "probe_hf_vlm.py"))
_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hf)

ProbeOutput = _hf.ProbeOutput
make_hf_vlm_output = _hf.make_hf_vlm_output

DEFAULT_QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"


def make_qwen_output(model_id: Optional[str] = None,
                     max_layers: int = 6,
                     load_in_4bit: bool = True) -> Optional[ProbeOutput]:
    """
    Load a Qwen2.5-VL VLM (default: 7B in 4-bit), run one forward, and return a
    ProbeOutput of RAW pre-softmax scores + post-softmax for the first
    `max_layers` decoder layers. Returns None (test skips) on any load failure.
    """
    model_id = model_id or os.environ.get("QWEN_VL_ID", DEFAULT_QWEN)
    hf_token = (os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return make_hf_vlm_output(
        model_id,
        name="qwen2.5-vl",
        model_class_names=("Qwen2_5_VLForConditionalGeneration",
                           "AutoModelForImageTextToText"),
        max_layers=max_layers,
        do_image_splitting=None,           # Qwen processor has no such flag
        device_map="auto",                  # single-GPU placement or sharding
        hf_token=hf_token,                  # None is fine (Qwen2.5-VL is open)
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
        extra_image_token_strings=("<|image_pad|>", "<image>"),
    )
