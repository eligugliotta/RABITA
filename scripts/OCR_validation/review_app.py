#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Human-in-the-loop OCR validation platform used in the RĀBIṬA project.

The app supports page-level validation of Arabic OCR outputs by displaying:
- the original page image;
- the OCR text to correct;
- an optional manually validated reference;
- automatic alerts for script mixing and OCR-sensitive patterns;
- optional neural TTS support using XTTS v2.

Run with:
    streamlit run review_app.py
"""

__author__ = "Elisa Gugliotta"
__project__ = "RĀBIṬA"

import streamlit as st
from pathlib import Path
import json
import re
import difflib
import unicodedata
#import streamlit.components.v1 as components
#import html
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

st.set_page_config(layout="wide")

st.sidebar.title("Configuration")

IMAGE_DIR = Path(st.sidebar.text_input("Image directory", "../WildFadhila_cropped"))
RAW_JSONL = Path(st.sidebar.text_input("Raw OCR JSONL", "../wildfadhila_qari_raw_full.jsonl"))
AUTO_TEXT = Path(st.sidebar.text_input("Post-processed OCR text", "qari_output_readable_manuallyREV.txt"))
MANUAL_VALIDATED = Path(st.sidebar.text_input("Manual reference text", "wildfadhila_ocr_clean2_valid.txt"))
ERROR_TSV = Path(st.sidebar.text_input("OCR mismatch TSV", "ocr_mismatch_first29_by_page.tsv"))

CORR_DIR = Path(st.sidebar.text_input("Corrections directory", "corrections"))
CORR_DIR.mkdir(exist_ok=True)

SPEAKER_WAV = Path(st.sidebar.text_input("Speaker WAV for TTS", "speaker.wav"))
TTS_CACHE_DIR = Path(st.sidebar.text_input("TTS cache directory", "tts_cache"))
TTS_CACHE_DIR.mkdir(exist_ok=True)

BOX_HEIGHT = st.sidebar.number_input("Text box height", min_value=300, max_value=1200, value=600, step=50)

st.set_page_config(layout="wide")

st.markdown("""
<style>
textarea {
    direction: rtl !important;
    unicode-bidi: plaintext !important;
    text-align: right !important;
    font-family: "Noto Naskh Arabic", "Scheherazade New", serif !important;
    font-size: 18px !important;
    line-height: 1.8 !important;
}

.stTextArea textarea {
    direction: rtl !important;
    unicode-bidi: plaintext !important;
    text-align: right !important;
}
</style>
""", unsafe_allow_html=True)

#import unicodedata

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")
TOKEN_RE = re.compile(r"\S+")
TUNISIAN_G = "ڨ"  # qāf con tre punti sopra, usata per /g/
ARABIC_QUESTION_MARK = "؟"

CONFUSABLE_MAP = {
    # cirillico/greco simili a lettere latine/arabe: da ampliare man mano
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
    "А": "A",
    "В": "B",
    "Е": "E",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "Х": "X",
}

@st.cache_resource
def load_xtts_model():
    torch.serialization.add_safe_globals([
        XttsConfig,
        XttsAudioConfig,
        XttsArgs,
        BaseDatasetConfig,
    ])

    use_gpu = torch.cuda.is_available()

    return TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        gpu=use_gpu
    )

def prepare_text_for_tts(text, max_chars=900):
    """
    Prepares text for TTS by removing page markers and limiting length.
    """
    text = re.sub(r"### PAGE\s+\d+\s+###", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars]

    return text

def script_of_char(ch):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"

    if "ARABIC" in name:
        return "ARABIC"
    if "LATIN" in name:
        return "LATIN"
    if ch.isdigit():
        return "DIGIT"
    if unicodedata.category(ch).startswith("P"):
        return "PUNCT"
    if ch.isspace():
        return "SPACE"
    return name.split()[0]

def detect_script_alerts(text):
    alerts = []

    for tok in TOKEN_RE.findall(text):
        has_ar = bool(ARABIC_RE.search(tok))
        has_lat = bool(LATIN_RE.search(tok))

        scripts = set()
        weird_chars = []

        for ch in tok:
            sc = script_of_char(ch)
            scripts.add(sc)

            if sc not in {"ARABIC", "LATIN", "DIGIT", "PUNCT", "SPACE"}:
                weird_chars.append(ch)

        if has_lat and not has_ar:
            alerts.append({
                "type": "latin_token",
                "token": tok,
                "message": "Token latino: verificare code-mixing e ordine della frase."
            })

        elif has_lat and has_ar:
            alerts.append({
                "type": "mixed_arabic_latin",
                "token": tok,
                "message": "Token misto arabo/latino: verificare se è code-mixing corretto o fusione OCR."
            })

        if weird_chars:
            proposed = "".join(CONFUSABLE_MAP.get(ch, ch) for ch in tok)
            alerts.append({
                "type": "non_arabic_non_latin",
                "token": tok,
                "message": f"Caratteri non arabi/non latini rilevati: {', '.join(sorted(set(weird_chars)))}",
                "proposal": proposed if proposed != tok else ""
            })

        # Tunisian /g/ orthographic variation:
        # OCR often reads ڨ as قّ or قَ
        if re.search(r"ق[\u064E\u0651\u0652]+|قّ|قَ|قْ", tok):

            suggestion = tok

            suggestion = suggestion.replace("قّ", "ڨ")
            suggestion = suggestion.replace("قَ", "ڨ")
            suggestion = suggestion.replace("قْ", "ڨ")

            suggestion = re.sub(
                r"ق[\u064E\u0651\u0652]+",
                "ڨ",
                suggestion
            )

            alerts.append({
                "type": "tunisian_g_candidate",
                "token": tok,
                "message": "Possible Tunisian /g/ spelling.",
                "proposal": suggestion
            })

        # Arabic punctuation:
        # OCR may preserve Latin question mark instead of Arabic question mark.
        if "?" in tok:
            alerts.append({
                "type": "latin_question_mark",
                "token": tok,
                "message": "Latin question mark detected. Consider replacing it with Arabic question mark.",
                "proposal": tok.replace("?", ARABIC_QUESTION_MARK)
            })

    return alerts

def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def split_pages(text):
    parts = re.split(r"(### PAGE\s+\d+\s+###)", text)
    pages = {}
    current = None
    buf = []

    for part in parts:
        m = re.match(r"### PAGE\s+(\d+)\s+###", part.strip())
        if m:
            if current is not None:
                pages[current] = "".join(buf).strip()
            current = int(m.group(1))
            buf = []
        else:
            buf.append(part)

    if current is not None:
        pages[current] = "".join(buf).strip()

    return pages


def make_diff(a, b):
    return "\n".join(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            fromfile="auto",
            tofile="corrected",
            lineterm=""
        )
    )

def load_error_suggestions(path):
    suggestions = {}
    if not path.exists():
        return suggestions

    import csv
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            page = int(row["page"])
            suggestions.setdefault(page, []).append(row)

    return suggestions

def protect_ltr_sequences(text):
    return text

def show_bidi_debug(text):
    lines = text.splitlines()
    debug_lines = []

    for line in lines:
        if "@" in line or re.search(r"[A-Za-z]", line):
            visible = (
                line
                .replace("\u2066", "[LRI]")
                .replace("\u2069", "[PDI]")
                .replace("\u200f", "[RLM]")
            )
            debug_lines.append(visible)

    return "\n".join(debug_lines)

raw_rows = load_jsonl(RAW_JSONL)
auto_pages = split_pages(AUTO_TEXT.read_text(encoding="utf-8"))
manual_pages = split_pages(MANUAL_VALIDATED.read_text(encoding="utf-8"))

error_suggestions = load_error_suggestions(ERROR_TSV)


images = sorted(
    [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
)

page_numbers = sorted(auto_pages.keys())
max_pages = min(len(images), len(raw_rows), len(page_numbers))

st.sidebar.title("Navigation")

st.sidebar.markdown("### Copy helpers")

st.sidebar.markdown("**Visible characters**")
st.sidebar.code("ڨ", language="text")
st.sidebar.caption("Tunisian /g/ letter")

st.sidebar.code("؟", language="text")
st.sidebar.caption("Arabic question mark")

st.sidebar.markdown("**Bidi control characters**")
st.sidebar.code("⁦", language="text")
st.sidebar.caption("LRI — Left-to-Right Isolate")

st.sidebar.code("⁩", language="text")
st.sidebar.caption("PDI — Pop Directional Isolate")

st.sidebar.code("‏", language="text")
st.sidebar.caption("RLM — Right-to-Left Mark")

st.sidebar.markdown("**Ready-to-copy templates**")

st.sidebar.code("⁦LATIN⁩", language="text")
st.sidebar.caption("Use for a Latin word inside Arabic text.")

st.sidebar.code("‏- ⁦LATIN⁩ النص العربي", language="text")
st.sidebar.caption("Dialogue turn with Latin name followed by Arabic text.")

st.sidebar.code("النص العربي ⁦LATIN⁩ -", language="text")
st.sidebar.caption("Visual RTL order: Arabic text + Latin element + dash.")

st.sidebar.divider()

if "idx" not in st.session_state:
    st.session_state.idx = 1

col_prev, col_next_nav = st.sidebar.columns(2)

with col_prev:
    if st.button("Previous"):
        if st.session_state.idx > 1:
            st.session_state.idx -= 1
            st.rerun()

with col_next_nav:
    if st.button("Next"):
        if st.session_state.idx < max_pages:
            st.session_state.idx += 1
            st.rerun()

idx = st.sidebar.number_input(
    "Image index",
    min_value=1,
    max_value=max_pages,
    value=st.session_state.idx,
    step=1
)


if idx != st.session_state.idx:
    st.session_state.idx = idx
    st.rerun()

st.session_state.idx = idx

page_numbers = sorted(auto_pages.keys())

page_num = page_numbers[idx - 1]
st.sidebar.caption(f"Current page: {page_num}")
image_path = images[idx - 1]
raw_text = raw_rows[idx - 1].get("text", "")
auto_text = auto_pages.get(page_num, "")

corr_path = CORR_DIR / f"page_{page_num:03d}.txt"

if corr_path.exists():
    initial_text = corr_path.read_text(encoding="utf-8")
else:
    initial_text = protect_ltr_sequences(auto_text)
    raw_text = protect_ltr_sequences(raw_text)

st.title(f"OCR Revision — page {page_num}")
st.caption(f"Image: {image_path.name}")

col1, col2, col3 = st.columns([1.0, 1.15, 1.15])

# -----------------------------
# COLONNA 1 — PDF / immagine
# -----------------------------
with col1:
    st.subheader("Original page")

    st.image(
        str(image_path),
        use_container_width=True
    )

# -----------------------------
# COLONNA 2 — TESTO DA CORREGGERE
# -----------------------------
with col2:
    st.subheader("QARI post-processed text to correct")

    corrected = st.text_area(
        "To validate",
        value=initial_text,
        height=BOX_HEIGHT
    )

    # -----------------------------
    # TEXT TO SPEECH
    # -----------------------------
    # -----------------------------
    # NEURAL TTS
    # -----------------------------

    tts_text = prepare_text_for_tts(corrected)

    audio_path = TTS_CACHE_DIR / f"page_{page_num:03d}.wav"

    st.caption("Audio is cached per page. Regenerate it after editing the text.")
    st.caption(f"TTS device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    col_tts1, col_tts2 = st.columns(2)

    with col_tts1:
        if st.button("Generate neural TTS"):

            if not SPEAKER_WAV.exists():
                st.error("speaker.wav not found in the validation folder.")

            elif not tts_text:
                st.error("No text available for TTS.")

            else:
                with st.spinner("Generating neural TTS..."):
                    tts = load_xtts_model()

                    tts.tts_to_file(
                        text=tts_text,
                        file_path=str(audio_path),
                        language="ar",
                        speaker_wav=str(SPEAKER_WAV)
                    )

                st.success(f"Audio generated: {audio_path}")

    with col_tts2:
        if audio_path.exists():
            st.audio(str(audio_path))
        else:
            st.info("No audio generated for this page yet.")

    col_save, col_next = st.columns(2)

    with col_save:
        if st.button("Save correction"):
            corr_path.write_text(
                corrected,
                encoding="utf-8"
            )

            st.success(f"Saved: {corr_path}")

    with col_next:
        if st.button("Save and continue"):

            corr_path.write_text(
                corrected,
                encoding="utf-8"
            )

            if st.session_state.idx < max_pages:
                st.session_state.idx += 1

            st.rerun()
# -----------------------------
# COLONNA 3 — VERSIONE MANUALE
# -----------------------------
with col3:
    st.subheader("Previously manually corrected version")

    if page_num in manual_pages and page_num <= 30:

        st.text_area(
            "Manually validated",
            value=manual_pages[page_num],
            height=BOX_HEIGHT,
            disabled=False
        )

    else:
        st.info(
            "No manually validated version available for this page."
        )

st.sidebar.divider()
st.sidebar.subheader("Automatic alerts")

alerts = detect_script_alerts(corrected)

if not alerts:
    st.sidebar.success("OK")

else:
    for a in alerts:

        if a["type"] == "latin_token":

            st.sidebar.warning(
                f"Latin/code-mixing:\n{a['token']}"
            )

        elif a["type"] == "mixed_arabic_latin":

            st.sidebar.error(
                f"Mixed token:\n{a['token']}"
            )

        elif a["type"] == "non_arabic_non_latin":

            st.sidebar.error(
                f"Unexpected characters:\n{a['token']}"
            )

            if a.get("proposal"):

                st.sidebar.info(
                    f"Suggested normalization:\n{a['proposal']}"
                )

        elif a["type"] == "tunisian_g_candidate":

            st.sidebar.warning(
                f"Tunisian /g/ candidate:\n{a['token']}"
            )

            if a.get("proposal"):

                st.sidebar.info(
                    f"Suggested replacement:\n{a['proposal']}"
                )

        elif a["type"] == "latin_question_mark":

            st.sidebar.warning(
                f"Latin question mark:\n{a['token']}"
            )

            if a.get("proposal"):

                st.sidebar.info(
                    f"Suggested replacement:\n{a['proposal']}"
                )

st.sidebar.divider()
st.sidebar.subheader("Historical mismatches")

page_errors = error_suggestions.get(page_num, [])

if not page_errors:
    st.sidebar.info("No known mismatches")
else:
    for e in page_errors[:15]:

        st.sidebar.markdown(
            f"""
**{e['category']}**

Manual:
`{e['gold']}`

QARI:
`{e['qari']}`
"""
        )

st.code(make_diff(auto_text, corrected), language="diff")

