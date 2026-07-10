"""
Plain PaliGemma VLM inference (greedy / top-p generation).

This is the standard inference loop for the PaliGemma model in this folder, with
the SparseVLM pruning removed. Use it to sanity-check that the model loads and
answers questions before / independently of the attention-map extraction.

Example
-------
python inference.py \
    --model_path /path/to/paligemma-3b-pt-224 \
    --image /path/to/car.png \
    --prompt "Color of car is Black right?" \
    --max_tokens 100
"""

import argparse

import torch
from PIL import Image

from modeling_gemma import KVCache
from attention_extractor import load_paligemma


def _sample_top_p(probs: torch.Tensor, p: float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


# The stripped PaliGemma.forward still accepts these (now inert) SparseVLM args;
# we always pass the "off" values so no pruning happens and every token is kept.
_SPARSE_OFF = (0.0, 0.0, False, False, None)  # (vis%, txt%, Sparse_VLM, Diff, dic)


@torch.no_grad()
def generate(model, processor, device, prompt, image_file_path,
             max_tokens=100, temperature=0.8, top_p=0.9, do_sample=False):
    image = Image.open(image_file_path).convert("RGB")
    model_inputs = processor(text=[prompt], images=[image])
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens):
        outputs = model(
            *_SPARSE_OFF,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        if next_token.item() == stop_token:
            break
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--image", required=True, type=str)
    ap.add_argument("--prompt", required=True, type=str)
    ap.add_argument("--max_tokens", default=100, type=int)
    ap.add_argument("--temperature", default=0.8, type=float)
    ap.add_argument("--top_p", default=0.9, type=float)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--only_cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.only_cpu else None
    model, processor, device = load_paligemma(args.model_path, device)
    print("Device in use:", device)

    answer = generate(
        model, processor, device, args.prompt, args.image,
        max_tokens=args.max_tokens, temperature=args.temperature,
        top_p=args.top_p, do_sample=args.do_sample,
    )
    print("\nQuestion :-", args.prompt)
    print("Model Answer :-", answer)


if __name__ == "__main__":
    main()
