#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Qari OCR launcher (Qwen2-VL) with robust text cleaning used in the RĀBIṬA project.

The script performs page-level OCR using the
NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct model.

Main features:
- Natural sorting of page images
- Optional image pre-processing (autocontrast + sharpen)
- Safer generation defaults + repetition controls to reduce runaway loops
- Post-processing to strip HTML-like tags and normalize whitespace
- Compression of absurdly long digit runs (e.g., phone number loops)

Example:
    python qari_ocr.py \
        --images ./images \
        --out_txt output.txt \
        --out_jsonl output.jsonl
"""

__author__ = "Elisa Gugliotta"
__project__ = "RĀBIṬA"

import json
import argparse
import re
import unicodedata
from pathlib import Path
from typing import List

#import torch
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
#from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration

PROMPT_AR = "اقرأ النص العربي في الصورة وأعده كسلسلة نصية فقط، دون وصف."
DEFAULT_MODEL = "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct"

# Invisibili che spesso inquinano downstream
INVISIBLE = {"\u200b", "\u200c", "\u200d", "\ufeff"}
REPL = {
    "\u00a0": " ",  # NBSP
    "\u00ad": "",   # soft hyphen
}


def normalize_text(s: str, remove_invisible: bool = True) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFKC", s)
    if remove_invisible:
        for ch in INVISIBLE:
            s = s.replace(ch, "")
        for k, v in REPL.items():
            s = s.replace(k, v)
    return s.strip()


def strip_html_keep_text(s: str) -> str:
    # Convert common break/block tags to newlines
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</?(p|h[1-6]|div|li|ul|ol|section|article)\b[^>]*>", "\n", s)
    # Remove any remaining tags
    s = re.sub(r"<[^>]+>", "", s)
    # Collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def squash_long_digit_runs(s: str, max_len: int = 20) -> str:
    # Keep the first max_len digits and truncate the rest
    return re.sub(rf"(\d{{{max_len}}})\d+", r"\1…", s)


def natural_key(p: Path):
    # Sort by the first integer found in the stem, then by name
    m = re.search(r"(\d+)", p.stem)
    return (int(m.group(1)) if m else 10**9, p.name)


def collect_images(path: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    p = Path(path)
    if p.is_dir():
        imgs = [x for x in p.iterdir() if x.suffix.lower() in exts]
        imgs = sorted(imgs, key=natural_key)
        return [str(x) for x in imgs]
    if p.is_file() and p.suffix.lower() in exts:
        return [str(p)]
    return []


def preprocess_image(image: Image.Image, max_side: int = 1280, enhance: bool = True) -> Image.Image:
    # Resize to keep inference reasonable
    image = image.convert("RGB")
    w, h = image.size
    scale = max(w, h) / max_side
    if scale > 1:
        image = image.resize((int(w / scale), int(h / scale)))

    if not enhance:
        return image

    # Help faint / low-contrast scans
    g = image.convert("L")
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.SHARPEN)
    return g.convert("RGB")


def load_model(model_id: str, dtype: str):
    import torch
    from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    base_id = "unsloth/Qwen2-VL-2B-Instruct"

    def repo_has_adapter(repo_id: str) -> bool:
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id, filename="adapter_model.safetensors")
            return True
        except Exception:
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id, filename="adapter_config.json")
                return True
            except Exception:
                return False

    if repo_has_adapter(model_id):
        from peft import PeftModel

        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_id,
            device_map="auto",
            dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(base, model_id).eval()

        # tokenizer/processor dal base (più robusto)
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        processor = AutoProcessor.from_pretrained(base_id)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch_dtype,
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

    return model, tokenizer, processor


@torch.inference_mode()
def ocr_one_image(
    model,
    tokenizer,
    processor,
    image_path: str,
    max_new_tokens: int,
    remove_invisible: bool,
    prompt: str,
    do_preprocess: bool,
    strip_html: bool,
    digit_run_max: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    image = Image.open(image_path)
    image = preprocess_image(image, enhance=do_preprocess)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[chat],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    if repetition_penalty and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

    output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = normalize_text(text, remove_invisible=remove_invisible)

    if strip_html:
        text = strip_html_keep_text(text)

    if digit_run_max and digit_run_max > 0:
        text = squash_long_digit_runs(text, max_len=int(digit_run_max))

    return text


def main():
    ap = argparse.ArgumentParser(description="OCR con Qari (Qwen2-VL) + cleaning")
    ap.add_argument("--images", required=True, help="Cartella immagini o singola immagine")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out_txt", required=True, help="Output TXT concatenato")
    ap.add_argument("--out_jsonl", default=None, help="(opz.) Output JSONL per pagina")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    # Safer defaults (avoid runaway)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=6)

    ap.add_argument("--page_sep", default="marker", choices=["marker", "formfeed", "none"])
    ap.add_argument("--keep_invisible", action="store_true")
    ap.add_argument("--prompt", default=PROMPT_AR)

    # Cleaning toggles
    ap.add_argument("--no_preprocess", action="store_true", help="Disattiva autocontrast+sharpen")
    ap.add_argument("--no_strip_html", action="store_true", help="Non rimuovere tag tipo <br>, <h3>...")
    ap.add_argument("--digit_run_max", type=int, default=20, help="Tronca sequenze di cifre troppo lunghe (0=off)")

    args = ap.parse_args()

    imgs = collect_images(args.images)
    if not imgs:
        raise SystemExit("Nessuna immagine trovata (controlla estensioni: png/jpg/tif/...).")

    model, tok, proc = load_model(args.model, args.dtype)

    if args.page_sep == "marker":
        sep = "\n\n### PAGE ###\n\n"
    elif args.page_sep == "formfeed":
        sep = "\n\f\n"
    else:
        sep = "\n"

    out_jsonl = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

    texts = []
    for ip in tqdm(imgs, desc="Qari-OCR"):
        txt = ocr_one_image(
            model,
            tok,
            proc,
            ip,
            max_new_tokens=args.max_new_tokens,
            remove_invisible=not args.keep_invisible,
            prompt=args.prompt,
            do_preprocess=not args.no_preprocess,
            strip_html=not args.no_strip_html,
            digit_run_max=args.digit_run_max,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        texts.append(txt)
        if out_jsonl:
            out_jsonl.write(json.dumps({"image": ip, "text": txt}, ensure_ascii=False) + "\n")

    if out_jsonl:
        out_jsonl.close()

    Path(args.out_txt).write_text(sep.join(texts), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
