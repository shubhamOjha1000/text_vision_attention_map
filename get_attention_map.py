"""
PaliGemma VLM inference + text->vision attention map at the decoder layer(s) YOU pick.

What it does
------------
1. Loads PaliGemma (SigLIP + Gemma) from `--model_path`.
2. Runs ONE forward (prefill) over (image, prompt)  -> this is the inference step
   that produces the decoder self-attention A = Softmax(Q Kᵀ / √d).
3. Hooks ONLY the decoder layer(s) you ask for (`--layers 0 5 17`) and grabs that
   layer's genuine attention `attn_weights` [B, H, L, L].
4. Slices it into the text->vision map with **text tokens in rows, vision tokens
   in columns**:
        P = A[text_positions][:, vision_positions]      # [L_t, L_v]

No SparseVLM, no pruning, no rater selection -> every text token is kept.

Run
---
python get_attention_map.py \
    --model_path /path/to/paligemma-3b-pt-224 \
    --image      /path/to/car.png \
    --prompt     "Color of car is Black right?" \
    --layers 0 \
    --save_dir   ./out          # optional: dumps map.pt (+ heatmap if matplotlib)

Import as a library
-------------------
    from get_attention_map import get_attention_maps, load_paligemma
    model, processor, device = load_paligemma(model_path)
    out = get_attention_maps(model, processor, device,
                             prompt="What color is the car?",
                             image_path="car.png", layers=[0])
    P       = out["maps"][0]            # [L_t, L_v]  head-averaged
    P_heads = out["maps_per_head"][0]   # [H, L_t, L_v]
    rows    = out["text_tokens"]        # row labels (len L_t)
"""

import argparse
from typing import Dict, List, Optional

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
def get_attention_maps(model, processor, device, prompt: str, image_path: str,
                       layers: List[int]) -> Dict:
    """
    Returns text->vision attention maps for the requested decoder `layers`.

    Output dict:
        maps          : {layer_idx: Tensor [L_t, L_v]}          head-averaged
        maps_per_head : {layer_idx: Tensor [num_heads, L_t, L_v]}
        text_tokens   : List[str]  length L_t   (row labels)
        vision_positions / text_positions : LongTensor indices in the full seq
    """
    # ---- capture buffer + hooks on ONLY the requested layers ----
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inp, output):
            # GemmaAttention.forward returns (attn_output, attn_weights)
            # attn_weights: [B, num_heads, q_len, kv_len]  <-- the genuine A
            captured[layer_idx] = output[1].detach().to("cpu")
        return hook

    wanted = set(layers)
    for module in model.modules():
        if isinstance(module, GemmaAttention) and module.layer_idx in wanted:
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

    missing = wanted - set(captured)
    if missing:
        raise RuntimeError(f"No attention captured for layer(s) {sorted(missing)}; "
                           f"model has {model.config.text_config.num_hidden_layers} layers.")

    # ---- text vs vision positions (the model's own rule) ----
    image_token_index = model.config.image_token_index
    pad_token_id = model.pad_token_id
    ids = input_ids[0]  # [L]
    text_mask = (ids != image_token_index) & (ids != pad_token_id)
    image_mask = ids == image_token_index
    text_positions = torch.nonzero(text_mask, as_tuple=False).squeeze(-1).cpu()
    vision_positions = torch.nonzero(image_mask, as_tuple=False).squeeze(-1).cpu()
    text_tokens = processor.tokenizer.convert_ids_to_tokens(ids[text_positions].tolist())

    # ---- slice each captured layer: rows = text, cols = vision ----
    maps, maps_per_head = {}, {}
    for layer_idx, attn in captured.items():
        attn = attn[0]                                              # [H, L, L]
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


def _maybe_plot(P, text_tokens, layer, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped] matplotlib unavailable: {e}")
        return
    fig, ax = plt.subplots(figsize=(12, max(4, len(text_tokens) * 0.35)))
    im = ax.imshow(P.numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(text_tokens)))
    ax.set_yticklabels(text_tokens, fontsize=8)
    ax.set_xlabel(f"vision tokens (L_v = {P.shape[1]})")
    ax.set_ylabel("text tokens (rows)")
    ax.set_title(f"Text -> Vision attention  P  (layer {layer})")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="google/paligemma-3b-pt-224",
                    help="HF Hub repo id (auto-downloaded) or a local weights dir")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[0],
                    help="decoder layer index/indices to extract, e.g. --layers 0 5 17")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--only_cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.only_cpu else None
    print("Loading model ...")
    model, processor, device = load_paligemma(args.model_path, device)
    print("Device:", device)

    out = get_attention_maps(model, processor, device,
                             args.prompt, args.image, args.layers)

    print("\n=== Text -> Vision attention map ===")
    print("prompt          :", out["prompt"])
    print("L_t (text rows) :", len(out["text_tokens"]))
    print("L_v (vision col):", out["vision_positions"].numel())
    print("row labels      :", out["text_tokens"])
    for layer in args.layers:
        P = out["maps"][layer]
        print(f"layer {layer}: P.shape = {tuple(P.shape)} (L_t, L_v) | "
              f"per-head = {tuple(out['maps_per_head'][layer].shape)} (H, L_t, L_v)")

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(out, os.path.join(args.save_dir, "map.pt"))
        print(f"[saved] {os.path.join(args.save_dir, 'map.pt')}")
        for layer in args.layers:
            _maybe_plot(out["maps"][layer], out["text_tokens"], layer,
                        os.path.join(args.save_dir, f"layer{layer}.png"))


if __name__ == "__main__":
    main()
