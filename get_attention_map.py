"""
PaliGemma VLM inference + text->vision RAW attention scores for ALL decoder layers.

What it does
------------
1. Loads PaliGemma (SigLIP + Gemma) from `model_path`.
2. Runs ONE forward (prefill) over (image, prompt)  -> this is the inference step
   that computes the decoder self-attention scores S = Q Kᵀ / √d.
3. Hooks EVERY decoder layer and grabs each layer's RAW pre-softmax scores
   `_raw_attn_scores` [B, H, L, L]  (NOT softmax).
4. Slices them into the text->vision map with **text tokens in rows, vision tokens
   in columns**:
        P = S[text_positions][:, vision_positions]      # [L_t, L_v]

No SparseVLM, no pruning, no rater selection -> every text token is kept.

Use as a library
----------------
    from get_attention_map import get_attention_maps, load_paligemma
    model, processor, device = load_paligemma(model_path)
    out = get_attention_maps(model, processor, device,
                             prompt="What color is the car?",
                             image_path="car.png")
    P       = out["maps"][0]            # layer-0 map [L_t, L_v]  head-averaged (raw scores)
    P_heads = out["maps_per_head"][0]   # layer-0 per head [H, L_t, L_v]
    rows    = out["text_tokens"]        # row labels (len L_t)
    # out["maps"] has one entry per decoder layer: {0, 1, ..., num_hidden_layers-1}
"""

from typing import Dict, Optional

import torch
from PIL import Image

from modeling_gemma import KVCache, GemmaAttention
from processing_paligemma import PaliGemmaProcessor
from utils import load_hf_model


# The stripped PaliGemma.forward still accepts these (now inert) SparseVLM args;
# we always pass the "off" values so nothing is pruned and every token is kept.
#   order: (vision%, text%, Sparse_VLM, Diff_pruining_ratio_Decoder, dic)
_SPARSE_OFF = (0.0, 0.0, False, False, None)


def load_paligemma(model_path: str, device: Optional[str] = None):
    """Load PaliGemma + processor. Auto-picks cuda/mps/cpu unless `device` given."""
    if device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    return model, processor, device


@torch.no_grad()
def get_attention_maps(model, processor, device, prompt: str, image_path: str) -> Dict:
    """
    Returns text->vision RAW (pre-softmax) score maps for EVERY decoder layer.

    Output dict:
        maps          : {layer_idx: Tensor [L_t, L_v]}          head-averaged raw scores
        maps_per_head : {layer_idx: Tensor [num_heads, L_t, L_v]}
        text_tokens   : List[str]  length L_t   (row labels)
        vision_positions / text_positions : LongTensor indices in the full seq
    """
    # ---- capture buffer + hooks on EVERY decoder self-attention layer ----
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inp, _output):
            # _module._raw_attn_scores: [B, num_heads, q_len, kv_len]
            # == the RAW pre-softmax scores QKᵀ/√d (NOT the softmax distribution)
            captured[layer_idx] = _module._raw_attn_scores.detach().to("cpu")
        return hook

    for module in model.modules():
        if isinstance(module, GemmaAttention):
            handles.append(module.register_forward_hook(make_hook(module.layer_idx)))

    try:
        # ---- build inputs (same path as normal inference) ----
        image = Image.open(image_path).convert("RGB")
        inp = processor(text=[prompt], images=[image])
        inp = {k: v.to(device) for k, v in inp.items()}
        input_ids = inp["input_ids"]            # [1, L]
        attention_mask = inp["attention_mask"]  # [1, L]
        pixel_values = inp["pixel_values"]

        # ---- ONE forward (prefill). Sparse_VLM=False -> no pruning ----
        model(
            *_SPARSE_OFF,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=KVCache(),
        )
    finally:
        for h in handles:
            h.remove()

    if not captured:
        raise RuntimeError("No attention captured; no GemmaAttention layers were hooked.")

    # ---- text vs vision positions (the model's own rule) ----
    image_token_index = model.config.image_token_index
    pad_token_id = model.pad_token_id
    ids = input_ids[0]  # [L]
    text_mask = (ids != image_token_index) & (ids != pad_token_id)
    image_mask = ids == image_token_index
    text_positions = torch.nonzero(text_mask, as_tuple=False).squeeze(-1).cpu()
    vision_positions = torch.nonzero(image_mask, as_tuple=False).squeeze(-1).cpu()
    text_tokens = processor.tokenizer.convert_ids_to_tokens(ids[text_positions].tolist())

    # ---- slice each captured RAW-score layer: rows = text, cols = vision ----
    maps, maps_per_head = {}, {}
    for layer_idx, attn in captured.items():
        attn = attn[0]                                              # [H, L, L] raw pre-softmax
        p_heads = attn[:, text_positions][:, :, vision_positions]   # [H, L_t, L_v]
        maps_per_head[layer_idx] = p_heads
        maps[layer_idx] = p_heads.mean(dim=0)                       # [L_t, L_v]

    return {
        "maps": dict(sorted(maps.items())),
        "maps_per_head": dict(sorted(maps_per_head.items())),
        "text_tokens": text_tokens,
        "text_positions": text_positions,
        "vision_positions": vision_positions,
        "prompt": prompt,
    }
