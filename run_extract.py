"""
Run the text -> vision attention-map extractor on one (image, prompt).

Example
-------
python run_extract.py \
    --model_path /path/to/paligemma-3b-pt-224 \
    --image /path/to/car.png \
    --prompt "Color of car is Black right?" \
    --layer 0 \
    --save_dir ./out

What you get
------------
For the chosen layer (and the all-layer average) it prints the shape of the
text->vision map  P  [L_t, L_v]  (text tokens in rows, vision tokens in columns)
and, if --save_dir is given, dumps:
    - maps.pt         : the full result (all layers, per-head + head-averaged)
    - layer{ L }.png  : a heatmap of P for the chosen layer (needs matplotlib)
"""

import argparse
import os

import torch

from attention_extractor import AttentionMapExtractor, load_paligemma


def maybe_plot(P: torch.Tensor, text_tokens, layer: int, save_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib optional
        print(f"[plot skipped] matplotlib unavailable: {e}")
        return

    fig_h = max(4, len(text_tokens) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(P.numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(text_tokens)))
    ax.set_yticklabels(text_tokens, fontsize=8)
    ax.set_xlabel(f"vision tokens (L_v = {P.shape[1]})")
    ax.set_ylabel("text tokens (rows)")
    ax.set_title(f"Text -> Vision attention map  P  (layer {layer})")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="google/paligemma-3b-pt-224", type=str,
                    help="HF Hub repo id (auto-downloaded) or a local weights dir")
    ap.add_argument("--image", required=True, type=str)
    ap.add_argument("--prompt", required=True, type=str)
    ap.add_argument("--layer", default=0, type=int,
                    help="which decoder layer's map to inspect/plot")
    ap.add_argument("--save_dir", default=None, type=str)
    ap.add_argument("--only_cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.only_cpu else None
    print("Loading model ...")
    model, processor, device = load_paligemma(args.model_path, device)
    print(f"Device: {device}")

    extractor = AttentionMapExtractor(model, processor, device)
    result = extractor.extract(args.prompt, args.image)

    print("\n=== Text -> Vision attention map ===")
    print(f"prompt          : {result.prompt}")
    print(f"L_t (text rows) : {result.L_t}")
    print(f"L_v (vision col): {result.L_v}")
    print(f"row labels      : {result.text_tokens}")
    print(f"num layers      : {len(result.maps)}")

    P = result.maps[args.layer]
    print(f"\nP[layer={args.layer}].shape = {tuple(P.shape)}   (L_t, L_v)")
    print(f"P_per_head[layer={args.layer}].shape = {tuple(result.maps_per_head[args.layer].shape)}   (H, L_t, L_v)")

    P_avg = result.mean_over_layers()
    print(f"P_avg-over-layers.shape = {tuple(P_avg.shape)}")

    # p_bar: per-visual-token significance (paper's Eq. 3), just for reference
    p_bar = result.visual_significance(layer=args.layer)
    print(f"p_bar[layer={args.layer}].shape = {tuple(p_bar.shape)}   (L_v,)  <- column-averaged score")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            {
                "maps": result.maps,
                "maps_per_head": result.maps_per_head,
                "text_tokens": result.text_tokens,
                "text_positions": result.text_positions,
                "vision_positions": result.vision_positions,
                "prompt": result.prompt,
            },
            os.path.join(args.save_dir, "maps.pt"),
        )
        print(f"[saved] {os.path.join(args.save_dir, 'maps.pt')}")
        maybe_plot(P, result.text_tokens, args.layer,
                   os.path.join(args.save_dir, f"layer{args.layer}.png"))


if __name__ == "__main__":
    main()
