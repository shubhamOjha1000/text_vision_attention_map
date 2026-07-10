# text_vision_attention_map

Extract the **text → vision attention map** from PaliGemma — the same decoder
self-attention that SparseVLM uses to build its priority matrix `P` — with
**text tokens in the rows and visual tokens in the columns**.

This folder is **not** about sparsification. No raters, no pruning. **Every text
token is kept in the rows.** We only read the attention map out and hand it back.

```
        P  =  A[ text_token_rows , vision_token_cols ]      shape [L_t, L_v]
```

## How the map is obtained (faithful to the paper's code)

The real decoder attention `A = Softmax(Q Kᵀ / √d)` is computed inside
`GemmaAttention.forward` (`../SparseVLM_code/modeling_gemma.py`, ~line 361) and
returned as the 2nd element of that module's output. We attach a PyTorch
`forward_hook` on every `GemmaAttention` block and capture that exact tensor
during a single prefill pass — so this is the genuine `A`, **not** a re-derived
`hidden @ hiddenᵀ` similarity. We then slice `rows = text positions`,
`cols = image positions`.

`text` / `image` positions use the model's own rule:
`image` = token id `== image_token_index` (`<image>`); `text` = everything else
that is not padding.

## Files

| file | purpose |
|------|---------|
| `attention_extractor.py` | `AttentionMapExtractor` — hooks + slicing; returns per-layer maps `[L_t, L_v]` and per-head `[H, L_t, L_v]`. |
| `run_extract.py` | CLI: load model, extract, print shapes, optionally save `maps.pt` + a heatmap PNG. |

The PaliGemma model itself (SigLIP + Gemma) is **reused as-is** from the sibling
`../SparseVLM_code/` folder (imported at runtime), so the map is identical to
what the paper's code sees. That folder must sit next to this one.

## Run

```bash
python run_extract.py \
    --model_path /path/to/paligemma-3b-pt-224 \
    --image      /path/to/car.png \
    --prompt     "Color of car is Black right?" \
    --layer 0 \
    --save_dir ./out
```

## Use as a library

```python
from attention_extractor import AttentionMapExtractor, load_paligemma

model, processor, device = load_paligemma(model_path)
extractor = AttentionMapExtractor(model, processor, device)

result = extractor.extract("What color is the car?", "car.png")

P        = result.maps[0]              # layer-0 map        [L_t, L_v]
P_heads  = result.maps_per_head[0]     # per head           [H, L_t, L_v]
P_avg    = result.mean_over_layers()   # averaged over layers [L_t, L_v]
rows     = result.text_tokens          # row labels (len L_t)

# (reference only) collapse rows -> per-visual-token score, the paper's p_bar:
p_bar    = result.visual_significance(layer=0)   # [L_v]
```

## Notes

- During prefill the code path uses a full (non-causal) attention mask, so text
  rows have well-defined weights over all vision columns.
- One `extract()` call = one forward pass; the hook captures each layer once.
- Head handling: `maps` are head-averaged; `maps_per_head` keeps all heads if you
  want a specific head or a different reduction.
