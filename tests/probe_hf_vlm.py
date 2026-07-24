"""
Generic RAW attention-score probe for ANY HuggingFace `transformers` VLM.

This is the model-agnostic core behind `probe_smolvlm.py` / `probe_llama4.py`.
It fills the SAME `ProbeOutput` used by `test_raw_attention.py`, so the invariant
tests run unchanged. Point it at a new model id and it just works (modulo gated
auth / hardware); that is the whole "generalises to any VLM" claim, in code.

Capture mechanism (identical to the verified SmolVLM path)
---------------------------------------------------------
We cannot edit a stock transformers model's source, so instead of stashing the
pre-softmax scores in the forward (as the in-repo PaliGemma does) we register a
raw-capturing **eager** attention by monkeypatching the module-level
`eager_attention_forward` in every loaded `transformers.models.*` file. With
`attn_implementation="eager"`, each attention module resolves that name from its
module globals at call time, so the model uses our version, which stashes:
    module._raw_attn_scores = Q Kᵀ*scaling (+mask)   # PRE-softmax
    module._post_attn        = softmax(...)          # POST-softmax
We then keep only the language-model self-attention layers (kv_len == q_len == L)
and drop the vision encoder's patch-to-patch attention.
"""

import importlib.util
import os
from typing import Optional, Sequence

import torch


def _shim_pil_ink():
    """Some `transformers` versions do `from PIL._typing import _Ink` at import
    time; older Pillow builds (e.g. on Colab) lack it. `_Ink` is only a type
    alias, so inject a harmless placeholder so the import succeeds with no
    restart. Must run BEFORE transformers is imported."""
    try:
        import typing
        import PIL._typing as _pt
        if not hasattr(_pt, "_Ink"):
            _pt._Ink = typing.Any
    except Exception:
        pass


_shim_pil_ink()

# --- import ProbeOutput from the sibling test module (by path; robust in Colab) ---
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "test_raw_attention", os.path.join(_here, "test_raw_attention.py"))
_T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_T)
ProbeOutput = _T.ProbeOutput


def _make_raw_capturing_eager(max_layers: int):
    """Stand-in for transformers' eager_attention_forward that stashes the
    pre-softmax scores (only for the first `max_layers` layers, to bound memory)."""
    import torch.nn.functional as F
    try:
        from transformers.models.llama.modeling_llama import repeat_kv
    except Exception:
        def repeat_kv(h, n):
            b, kv, s, d = h.shape
            if n == 1:
                return h
            return h[:, :, None, :, :].expand(b, kv, n, s, d).reshape(b, kv * n, s, d)

    def fn(module, query, key, value, attention_mask=None, scaling=None,
           dropout=0.0, **kwargs):
        if scaling is None:
            scaling = getattr(module, "scaling", 1.0)
        ng = getattr(module, "num_key_value_groups", 1)
        k = repeat_kv(key, ng)
        v = repeat_kv(value, ng)
        attn = torch.matmul(query, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn = attn + attention_mask[:, :, :, : k.shape[-2]]

        if getattr(module, "layer_idx", 10 ** 9) < max_layers:
            module._raw_attn_scores = attn.detach().float().cpu()   # PRE-softmax
        probs = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        if getattr(module, "layer_idx", 10 ** 9) < max_layers:
            module._post_attn = probs.detach().float().cpu()        # POST-softmax
        probs = F.dropout(probs, p=dropout, training=module.training)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous()
        return out, None

    return fn


def _patch_eager_globals(new_fn):
    """Replace module-level `eager_attention_forward` in every loaded
    transformers model file. Returns [(module, original_fn)] for restoration."""
    import sys
    patched = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("transformers.models."):
            continue
        if getattr(mod, "eager_attention_forward", None) is not None:
            patched.append((mod, mod.eager_attention_forward))
            mod.eager_attention_forward = new_fn
    return patched


def _unpatch_eager_globals(patched):
    for mod, orig in patched:
        mod.eager_attention_forward = orig


def _load_demo_image():
    """Fetch a demo image with plain PIL (+requests); blank fallback if offline.
    Avoids `transformers.image_utils.load_image` (its `_Ink` import breaks on
    some Colab Pillow builds)."""
    from PIL import Image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        import io
        import requests
        return Image.open(io.BytesIO(requests.get(url, timeout=30).content)).convert("RGB")
    except Exception:
        return Image.new("RGB", (384, 384), (127, 127, 127))


def _find_image_token_id(model, processor, extra: Sequence[str] = ()):
    for cfg in (model.config,
                getattr(model.config, "text_config", None),
                getattr(model.config, "vision_config", None)):
        if cfg is None:
            continue
        for attr in ("image_token_id", "image_token_index"):
            tid = getattr(cfg, attr, None)
            if isinstance(tid, int):
                return tid
    for tok in tuple(extra) + ("<image>", "<|image|>", "<fake_token_around_image>"):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != processor.tokenizer.unk_token_id:
            return tid
    raise RuntimeError("could not determine image_token_id for this model")


def _resolve_model_class(model_class_names):
    import transformers
    for cname in model_class_names:
        cls = getattr(transformers, cname, None)
        if cls is not None:
            return cls
    raise RuntimeError(f"none of {list(model_class_names)} exist in this transformers")


def make_hf_vlm_output(
    model_id: str,
    *,
    name: str,
    model_class_names: Sequence[str] = ("AutoModelForImageTextToText",
                                        "AutoModelForVision2Seq"),
    max_layers: int = 6,
    do_image_splitting: Optional[bool] = False,
    device_map: Optional[str] = None,
    hf_token: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = False,
    extra_image_token_strings: Sequence[str] = (),
    question: str = "How many cats are in the image?",
) -> Optional[ProbeOutput]:
    """
    Load a HF VLM, run one forward on (image, prompt), and return a ProbeOutput
    with RAW pre-softmax scores + post-softmax for the first `max_layers`
    language-model decoder layers. Returns None (test skips) on any failure
    (gated auth / OOM / unsupported version / offline).

    device_map="auto" shards a large model across GPUs (use for Llama-4 etc.);
    leave None to place a small model on a single device.
    """
    try:
        _shim_pil_ink()
        from transformers import AutoProcessor
        model_cls = _resolve_model_class(model_class_names)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        tok_kw = {"token": hf_token} if hf_token else {}
        processor = AutoProcessor.from_pretrained(model_id, **tok_kw)
        if do_image_splitting is not None:
            try:
                processor.image_processor.do_image_splitting = do_image_splitting
            except Exception:
                pass

        load_kw = dict(torch_dtype=dtype, attn_implementation="eager", **tok_kw)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            if device_map is None:
                device_map = "auto"   # quantized weights must be placed via device_map
        if device_map is not None:
            load_kw["device_map"] = device_map
        model = model_cls.from_pretrained(model_id, **load_kw)
        if device_map is None:
            model = model.to(device)
        model = model.eval()
        try:
            model_device = model.get_input_embeddings().weight.device
        except Exception:
            model_device = next(model.parameters()).device

        # ---- build one (image, prompt) input via the chat template ----
        image = _load_demo_image()
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inp = processor(text=prompt, images=[image], return_tensors="pt").to(model_device)
        assert "input_ids" in inp, "processor returned no input_ids"
        assert "pixel_values" in inp, ("processor returned no pixel_values -- image "
                                       "dropped; check processor call / chat template")

        # ---- swap in the raw-capturing eager attention for the forward ----
        patched = _patch_eager_globals(_make_raw_capturing_eager(max_layers))
        assert patched, ("no transformers `eager_attention_forward` found to patch; "
                         "this transformers version may use a different attention path.")
        try:
            with torch.no_grad():
                model(**inp)
        finally:
            _unpatch_eager_globals(patched)

        input_ids = inp["input_ids"][0].cpu()
        L = int(input_ids.shape[0])

        # ---- keep only language-model self-attention (kv_len == q_len == L) ----
        raw_scores, post_softmax = {}, {}
        for module in model.modules():
            raw = getattr(module, "_raw_attn_scores", None)
            post = getattr(module, "_post_attn", None)
            if raw is None or post is None:
                continue
            if raw.shape[-1] != L or raw.shape[-2] != L:
                continue  # vision-encoder attention -> skip
            li = int(getattr(module, "layer_idx", len(raw_scores)))
            raw_scores[li] = raw[0].float()
            post_softmax[li] = post[0].float()
            del module._raw_attn_scores, module._post_attn

        assert raw_scores, ("no language-model attention captured (kv_len==L). "
                            "Check the transformers version / attention path.")

        img_id = _find_image_token_id(model, processor, extra_image_token_strings)
        pad_id = processor.tokenizer.pad_token_id
        pad_id = pad_id if pad_id is not None else -(10 ** 9)
        image_mask = input_ids == img_id
        text_mask = (input_ids != img_id) & (input_ids != pad_id)

        return ProbeOutput(input_ids, image_mask, text_mask, raw_scores, post_softmax,
                           expected_num_image_tokens=None, name=name, question=question)
    except Exception as e:  # gated / OOM / unsupported / offline -> skip gracefully
        print(f"[{name} probe unavailable] {type(e).__name__}: {e}")
        return None
