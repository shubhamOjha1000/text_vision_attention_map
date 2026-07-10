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
| `get_attention_map.py` | **Main entry.** Hooks only the decoder layer(s) you pick; returns head-averaged `[L_t, L_v]` and per-head `[H, L_t, L_v]` maps. |
| `attention_extractor.py` / `run_extract.py` | Library + CLI variants that capture **all** layers at once. |
| `inference.py` | Plain PaliGemma generation loop (sanity-check the model answers). |
| `modeling_gemma.py`, `modeling_siglip.py`, `processing_paligemma.py`, `utils.py` | Self-contained PaliGemma (SigLIP + Gemma) implementation, SparseVLM removed. |

The folder is self-contained — no dependency on any sibling folder.

## Weights

Weights load **straight from the HuggingFace Hub** by default
(`google/paligemma-3b-pt-224`); no manual download needed. You can also pass a
local directory (containing `config.json`, tokenizer files and `*.safetensors`)
to `--model_path`.

PaliGemma is a **gated** model, so accept its license on the Hub once and
authenticate before first download:

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_xxx
```

## Run

```bash
# --model_path defaults to google/paligemma-3b-pt-224 (auto-downloaded)
python get_attention_map.py \
    --image   /path/to/car.png \
    --prompt  "Color of car is Black right?" \
    --layers  0 \
    --save_dir ./out
```

Pick any decoder layer(s): `--layers 0 5 17`.

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
