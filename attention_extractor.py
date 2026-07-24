"""
Text -> Vision RAW attention-score extractor for PaliGemma.

Goal
----
Return the language-model decoder's **raw, pre-softmax** attention scores
`S = Q K^T / sqrt(d)` (NOT the post-softmax distribution `Softmax(S)`), sliced
into

        P  =  S[ text_token_rows , vision_token_cols ]        # [L_t, L_v]

so that **text tokens are the rows and visual tokens are the columns**.

Difference from SparseVLM
-------------------------
This module has NOTHING to do with pruning / sparsification. We do not select
"raters" and we do not drop any tokens. **ALL text tokens are kept in the rows.**
We only read the raw scores out of the model and return them.

How the map is obtained
-----------------------
Inside `GemmaAttention.forward` (modeling_gemma.py, ~line 331) the raw scores
`Q K^T / sqrt(d)` are stashed on the module as `_raw_attn_scores`, *before* the
attention mask is added and *before* softmax. We attach a `forward_hook` on every
`GemmaAttention` module and read that attribute during a single prefill pass --
these are the genuine raw scores, not a re-derived similarity and not the
softmax'd map. We then slice text-rows x vision-cols.

Note: raw scores are unnormalized logits, meaningful up to a per-row additive
constant (softmax is shift-invariant). Use them for ranking / differences /
custom-temperature re-softmaxing, not as absolute magnitudes.

The PaliGemma model (SigLIP + Gemma) lives in this same folder -- it is the
paper's implementation with the SparseVLM pruning stripped out (pure inference),
so the attention map is identical to what the original code sees.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from PIL import Image

from modeling_gemma import KVCache, GemmaAttention, PaliGemmaForConditionalGeneration
from processing_paligemma import PaliGemmaProcessor
from utils import load_hf_model


@dataclass
class AttentionMapResult:
    """Container for one (image, prompt) extraction."""
    # Per-layer text->vision RAW (pre-softmax) score maps, head-averaged. Each: [L_t, L_v]
    maps: Dict[int, torch.Tensor]
    # Per-layer RAW score maps with heads kept separately. Each: [num_heads, L_t, L_v]
    maps_per_head: Dict[int, torch.Tensor]
    # String tokens for the rows (text) and a simple index list for cols (vision)
    text_tokens: List[str]          # length L_t   -> row labels
    text_positions: torch.Tensor    # length L_t   -> their index in the full seq
    vision_positions: torch.Tensor  # length L_v   -> their index in the full seq
    prompt: str
    L_t: int = field(init=False)
    L_v: int = field(init=False)

    def __post_init__(self):
        self.L_t = len(self.text_tokens)
        self.L_v = int(self.vision_positions.numel())

    def mean_over_layers(self) -> torch.Tensor:
        """Average the text->vision map across all layers -> [L_t, L_v]."""
        stacked = torch.stack(list(self.maps.values()), dim=0)  # [num_layers, L_t, L_v]
        return stacked.mean(dim=0)

    def visual_significance(self, layer: Optional[int] = None) -> torch.Tensor:
        """
        Collapse rows -> one raw-score per visual token:
            p_bar = (1 / L_t) * sum_over_text_rows( P )      -> [L_v]
        NOTE: P here holds RAW pre-softmax scores, so this is a mean of raw
        logits (up to a per-row additive constant), not the paper's post-softmax
        `p_bar`. Use for ranking patches; renormalize if you need a distribution.
        `layer=None` uses the layer-averaged map.
        """
        p = self.mean_over_layers() if layer is None else self.maps[layer]
        return p.mean(dim=0)


class AttentionMapExtractor:
    """
    Wraps a loaded PaliGemma model and pulls out text->vision attention maps.

    Usage:
        model, tok = load_hf_model(model_path, device)
        extractor = AttentionMapExtractor(model, processor, device)
        result = extractor.extract(prompt, image_path)
        P = result.maps[0]            # layer-0 map, [L_t, L_v]
        P_avg = result.mean_over_layers()
    """

    def __init__(self, model: PaliGemmaForConditionalGeneration,
                 processor: PaliGemmaProcessor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.image_token_index = model.config.image_token_index
        self.pad_token_id = model.pad_token_id

        self._captured: Dict[int, torch.Tensor] = {}
        self._handles = []
        self._register_hooks()

    # ------------------------------------------------------------------ hooks
    def _register_hooks(self):
        """Attach a forward hook on every decoder self-attention block."""
        for module in self.model.modules():
            if isinstance(module, GemmaAttention):
                handle = module.register_forward_hook(self._make_hook(module.layer_idx))
                self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, _output):
            # _module._raw_attn_scores: [B, num_heads, q_len, kv_len]
            # == the RAW pre-softmax scores QKᵀ/√d (NOT the softmax distribution)
            self._captured[layer_idx] = _module._raw_attn_scores.detach().to("cpu")
        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # -------------------------------------------------------------- extraction
    @torch.no_grad()
    def extract(self, prompt: str, image_file_path: str) -> AttentionMapResult:
        """
        Run a single prefill pass over (prompt, image) and return the
        text->vision attention maps for every decoder layer.
        """
        self._captured.clear()

        # ---- build model inputs (same path as inference.py) ----
        image = Image.open(image_file_path).convert("RGB")
        model_inputs = self.processor(text=[prompt], images=[image])
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        input_ids = model_inputs["input_ids"]            # [1, L]
        attention_mask = model_inputs["attention_mask"]  # [1, L]
        pixel_values = model_inputs["pixel_values"]

        # ---- single forward (prefill). Sparse_VLM=False -> no pruning, all tokens kept ----
        kv_cache = KVCache()
        self.model(
            0.0,          # vision_token_pruning_percentage (unused when Sparse_VLM=False)
            0.0,          # text_token_pruning_percentage   (unused when Sparse_VLM=False)
            False,        # Sparse_VLM  -> keep every token
            False,        # Diff_pruining_ratio_Decoder
            None,         # dic (unused)
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        # ---- identify text vs vision positions (same rule as the model) ----
        ids = input_ids[0]  # [L]
        text_mask = (ids != self.image_token_index) & (ids != self.pad_token_id)
        image_mask = ids == self.image_token_index
        text_positions = torch.nonzero(text_mask, as_tuple=False).squeeze(-1).cpu()
        vision_positions = torch.nonzero(image_mask, as_tuple=False).squeeze(-1).cpu()

        text_tokens = self.processor.tokenizer.convert_ids_to_tokens(
            ids[text_positions].tolist()
        )

        # ---- slice each captured RAW-score map: rows = text, cols = vision ----
        maps: Dict[int, torch.Tensor] = {}
        maps_per_head: Dict[int, torch.Tensor] = {}
        for layer_idx, attn in self._captured.items():
            # attn: [1, H, L, L]  raw pre-softmax scores (prefill -> q_len == kv_len == L)
            attn = attn[0]                                   # [H, L, L]
            p_heads = attn[:, text_positions][:, :, vision_positions]  # [H, L_t, L_v]
            maps_per_head[layer_idx] = p_heads
            maps[layer_idx] = p_heads.mean(dim=0)            # [L_t, L_v]

        return AttentionMapResult(
            maps=dict(sorted(maps.items())),
            maps_per_head=dict(sorted(maps_per_head.items())),
            text_tokens=text_tokens,
            text_positions=text_positions,
            vision_positions=vision_positions,
            prompt=prompt,
        )


def load_paligemma(model_path: str, device: Optional[str] = None):
    """Convenience loader mirroring inference.py."""
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
