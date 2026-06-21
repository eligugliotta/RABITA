#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAMeL Parser UD pipeline used in the RĀBIṬA project.

This script runs CAMeL Parser in Universal Dependencies mode on a plain-text
input file and exports the result as CoNLL-U.

If CAMeL Parser fails on a sentence-like unit, the script writes a valid
fallback CoNLL-U block using the UPOS tag X. This keeps problematic segments
inside the workflow so that they can later be inspected and corrected manually
in INCEpTION.

External dependency:
    CAMeL Parser: https://github.com/CAMeL-Lab/camel_parser

Example:
    python parse_text_ud.py \
        --input ../../corpus/weld_fadila/validated/WF_page_020_B2.txt \
        --output ../../corpus/weld_fadila/conllu_auto/WF_page_020_B2_auto.conllu \
        --camel-cli ./camel_parser/text_to_conll_cli.py
"""

__author__ = "Elisa Gugliotta"
__project__ = "RĀBIṬA"

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List


def run_camelparser_on_text(
    text: str,
    camel_cli: Path,
    disambig: str = "bert",
    mode: str = "ud",
) -> subprocess.CompletedProcess:
    """Run CAMeL Parser on one text unit."""
    cmd = [
        sys.executable,
        str(camel_cli),
        "-f",
        "text",
        "-d",
        disambig,
        "-m",
        mode,
        "-s",
        text,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def naive_tokenize(text: str) -> List[str]:
    """Fallback tokenizer used only when CAMeL Parser fails."""
    text = re.sub(r"([،؛:,.!?؟()\"«»“”])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def write_fallback_sentence(out_fh, sent_id: int, sent_text: str) -> None:
    """
    Write a minimal valid CoNLL-U block for an unparsed text unit.

    CoNLL-U columns:
    ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
    """
    tokens = naive_tokenize(sent_text)

    out_fh.write(f"# sent_id = fallback-{sent_id}\n")
    out_fh.write(f"# text = {sent_text.strip()}\n")

    for token_id, token in enumerate(tokens, start=1):
        out_fh.write(
            f"{token_id}\t{token}\t_\tX\t_\t_\t0\tdep\t_\tFallback=Yes\n"
        )

    out_fh.write("\n")


def split_into_units(text: str) -> List[str]:
    """
    Split text into robust sentence-like units.

    This intentionally simple splitter preserves the workflow on noisy OCR
    and non-standard Arabic texts. It first splits on non-empty lines, then
    on strong punctuation.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    units: List[str] = []
    for line in lines:
        parts = re.split(r"(?<=[\.\!\?؟\…])\s+", line)
        units.extend(part.strip() for part in parts if part.strip())

    return units


def parse_file(
    input_file: Path,
    output_file: Path,
    fail_log: Path,
    camel_cli: Path,
    disambig: str = "bert",
    mode: str = "ud",
) -> None:
    """Parse an input text file and write CoNLL-U output."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not camel_cli.exists():
        raise FileNotFoundError(
            f"CAMeL Parser CLI not found: {camel_cli}\n"
            "Clone or install CAMeL Parser and pass the path with --camel-cli."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fail_log.parent.mkdir(parents=True, exist_ok=True)

    raw_text = input_file.read_text(encoding="utf-8")
    units = split_into_units(raw_text)

    ok_count = 0
    fail_count = 0

    fail_log.write_text("", encoding="utf-8")

    with output_file.open("w", encoding="utf-8") as out_fh, fail_log.open(
        "a", encoding="utf-8"
    ) as log_fh:
        for idx, unit in enumerate(units, start=1):
            result = run_camelparser_on_text(
                unit,
                camel_cli=camel_cli,
                disambig=disambig,
                mode=mode,
            )

            if result.returncode == 0 and result.stdout.strip():
                out_fh.write(result.stdout.rstrip() + "\n\n")
                ok_count += 1
            else:
                write_fallback_sentence(out_fh, idx, unit)
                fail_count += 1

                error_message = (result.stderr or result.stdout or "").strip()
                log_fh.write(f"=== FAIL sent_id={idx} ===\n")
                log_fh.write(f"TEXT: {unit}\n")
                if error_message:
                    log_fh.write("ERROR:\n")
                    log_fh.write(error_message[:4000] + "\n")
                log_fh.write("\n")

    print(f"[OK] Written: {output_file}")
    print(f"[INFO] Parsed units: {ok_count} | fallback units: {fail_count}")
    if fail_count:
        print(f"[INFO] Failure log: {fail_log}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CAMeL Parser in UD mode and export resilient CoNLL-U."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input UTF-8 plain-text file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CoNLL-U file.",
    )
    parser.add_argument(
        "--camel-cli",
        default=Path("camel_parser/text_to_conll_cli.py"),
        type=Path,
        help="Path to CAMeL Parser text_to_conll_cli.py.",
    )
    parser.add_argument(
        "--fail-log",
        default=None,
        type=Path,
        help="Optional path for parsing failure log.",
    )
    parser.add_argument(
        "--disambig",
        default="bert",
        choices=["bert", "mle"],
        help="CAMeL Parser disambiguator.",
    )
    parser.add_argument(
        "--mode",
        default="ud",
        choices=["ud", "catib"],
        help="Parsing scheme.",
    )

    args = parser.parse_args()

    fail_log = args.fail_log
    if fail_log is None:
        fail_log = args.output.with_suffix(".failures.log")

    parse_file(
        input_file=args.input,
        output_file=args.output,
        fail_log=fail_log,
        camel_cli=args.camel_cli,
        disambig=args.disambig,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
