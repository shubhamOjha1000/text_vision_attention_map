"""
Llama-4 adapter for the RAW attention-score tests.

Thin wrapper over the generic `probe_hf_vlm.make_hf_vlm_output` -- the exact same
eager-patch capture verified on SmolVLM2, pointed at a Llama-4 checkpoint. Fills
the SAME `ProbeOutput`, so every test in `test_raw_attention.py` runs unchanged.

Default model: **Llama-4-Scout-17B-16E-Instruct** (the smaller Llama-4; Maverick
is ~400B and needs an even larger node). Both are:
  * **gated**  -> accept the license on the Hub and authenticate (HF token /
    `huggingface_hub.login()`), and
  * **multi-GPU** -> they do NOT fit on a single Colab GPU; loaded with
    `device_map="auto"` to shard across whatever accelerators are available.

Override the checkpoint with the `LLAMA4_ID` env var or the `model_id` arg
(e.g. `meta-llama/Llama-4-Maverick-17B-128E-Instruct`).
"""

import importlib.util
import os
from typing import Optional

import torch

# load the generic core by path (robust in Colab)
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "probe_hf_vlm", os.path.join(_here, "probe_hf_vlm.py"))
_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hf)

ProbeOutput = _hf.ProbeOutput
make_hf_vlm_output = _hf.make_hf_vlm_output

DEFAULT_LLAMA4 = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


def make_llama4_output(model_id: Optional[str] = None,
                       max_layers: int = 4) -> Optional[ProbeOutput]:
    """
    Load a Llama-4 VLM (default: Scout), run one forward, and return a
    ProbeOutput of RAW pre-softmax scores + post-softmax for the first
    `max_layers` decoder layers. Returns None (test skips) if the model can't be
    loaded (gated auth / not enough GPUs / offline).
    """
    model_id = model_id or os.environ.get("LLAMA4_ID", DEFAULT_LLAMA4)
    hf_token = (os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return make_hf_vlm_output(
        model_id,
        name="llama4",
        # Llama-4 exposes a dedicated conditional-generation class; fall back to Auto.
        model_class_names=("Llama4ForConditionalGeneration",
                           "AutoModelForImageTextToText"),
        max_layers=max_layers,
        do_image_splitting=None,          # Llama-4 processor has no such flag
        device_map="auto",                 # shard across GPUs
        hf_token=hf_token,                 # None is fine if already logged in
        dtype=torch.bfloat16,              # recommended for Llama-4
        extra_image_token_strings=("<|image|>", "<image>"),
    )
