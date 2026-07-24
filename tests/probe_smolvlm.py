"""
SmolVLM2 adapter for the RAW attention-score tests.

Fills the SAME `ProbeOutput` used by `test_raw_attention.py`, so every invariant
test in that file runs against SmolVLM unchanged. This is the concrete proof of
the "generalisable to any VLM" claim: one adapter per model, tests untouched.

How the raw scores are captured (model-agnostic, no source edit)
---------------------------------------------------------------
SmolVLM2 is a HuggingFace `transformers` model, so we cannot edit its forward the
way we did for the in-repo PaliGemma. Instead we register a custom **eager**
attention function into `transformers`' attention interface
(`ALL_ATTENTION_FUNCTIONS`). It is a faithful copy of the stock
`eager_attention_forward` that additionally stashes, on each attention module:
    module._raw_attn_scores  = Q Kᵀ * scaling (+ mask)   # PRE-softmax  <-- captured
    module._post_attn        = softmax(...)              # POST-softmax
exactly mirroring the PaliGemma pattern (pre-softmax at the matmul, post-softmax
after the softmax) but without touching the model's code.

We then keep only the LANGUAGE-model self-attention layers (kv_len == sequence
length L) and drop the vision encoder's patch-to-patch attention.

The same recipe works for any recent transformers VLM that uses the attention
interface (Idefics3, Qwen2-VL, LLaVA-NeXT, ...): only the model id and the
image-token lookup change.
"""

import importlib.util
import os
from typing import Optional

import torch


def _shim_pil_ink():
    """Some `transformers` versions do `from PIL._typing import _Ink` at import
    time (e.g. inside `transformers.image_utils`, which the processor pulls in).
    Older Pillow builds shipped on Colab lack `_Ink`, raising
    `ImportError: cannot import name '_Ink' from 'PIL._typing'`. `_Ink` is only a
    type alias, so injecting a harmless placeholder makes the import succeed with
    no restart / Pillow reinstall. Must run BEFORE transformers is imported."""
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

_CACHE = {"tried": False, "output": None}


def _make_raw_capturing_eager(max_layers: int):
    """A stand-in for transformers' eager_attention_forward that stashes the
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

        # stash only for the first few layers to keep CPU memory bounded
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
    """Replace the module-level `eager_attention_forward` in every loaded
    transformers model file with `new_fn`. With `attn_implementation="eager"`,
    each attention module resolves that name from its module globals at call
    time, so this makes the model use our raw-capturing version. Returns the list
    of (module, original_fn) so it can be restored. This is version-robust:
    recent transformers dispatch "eager" to this module-level function, NOT
    through ALL_ATTENTION_FUNCTIONS (which does not even contain an "eager" key)."""
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
    """Fetch a demo image with plain PIL (+requests). Avoids
    `transformers.image_utils.load_image`, whose `from PIL._typing import _Ink`
    breaks against the Pillow already loaded in some Colab kernels. Falls back to
    a blank image if there's no network."""
    from PIL import Image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        import io
        import requests
        data = requests.get(url, timeout=30).content
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return Image.new("RGB", (384, 384), (127, 127, 127))


def _find_image_token_id(model, processor):
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        tid = getattr(cfg, "image_token_id", None)
        if tid is not None:
            return tid
    for tok in ("<image>", "<fake_token_around_image>"):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != processor.tokenizer.unk_token_id:
            return tid
    raise RuntimeError("could not determine image_token_id for this model")


def make_smolvlm_output(model_id: Optional[str] = None,
                        max_layers: int = 6) -> Optional[ProbeOutput]:
    """
    Load SmolVLM2, run one forward on (image, prompt), and return a ProbeOutput
    with genuine RAW pre-softmax scores + the model's post-softmax, for the first
    `max_layers` language-model decoder layers. Returns None (test skips) if the
    model / transformers version / hardware can't support it.
    """
    if _CACHE["tried"]:
        return _CACHE["output"]
    _CACHE["tried"] = True

    model_id = model_id or os.environ.get("SMOLVLM_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    try:
        _shim_pil_ink()  # ensure PIL._typing._Ink exists before transformers imports
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except Exception:
            from transformers import AutoModelForVision2Seq as _AutoVLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(model_id)
        # one image tile -> shorter sequence -> less memory (best-effort)
        try:
            processor.image_processor.do_image_splitting = False
        except Exception:
            pass

        model = _AutoVLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="eager").to(device).eval()

        # ---- build one (image, prompt) input via the chat template ----
        image = _load_demo_image()
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "How many cats are in the image?"}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inp = processor(text=prompt, images=[image], return_tensors="pt").to(device)

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
            # free the stashed tensors off the modules
            del module._raw_attn_scores, module._post_attn

        assert raw_scores, ("no language-model attention captured (kv_len==L). "
                            "Check the transformers version / attention interface.")

        img_id = _find_image_token_id(model, processor)
        pad_id = processor.tokenizer.pad_token_id
        pad_id = pad_id if pad_id is not None else -(10 ** 9)
        image_mask = input_ids == img_id
        text_mask = (input_ids != img_id) & (input_ids != pad_id)

        out = ProbeOutput(input_ids, image_mask, text_mask, raw_scores, post_softmax,
                          expected_num_image_tokens=None, name="smolvlm")
    except Exception as e:  # unsupported version / OOM / offline -> skip gracefully
        print(f"[smolvlm probe unavailable] {type(e).__name__}: {e}")
        out = None

    _CACHE["output"] = out
    return out
