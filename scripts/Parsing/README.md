# CAMeL Parser UD Pipeline

This directory contains the scripts used in the RĀBIṬA project to generate automatic Universal Dependencies annotations for selected Arabic literary texts.

## External Dependency

The parsing workflow relies on **CAMeL Parser**, developed by the CAMeL Lab.
Repository: https://github.com/CAMeL-Lab/camel_parser
Please refer to the official documentation for installation instructions, model descriptions, and updates.

## Workflow

Validated text
- CAMeL Parser (UD mode)
- Automatic CoNLL-U output
- Resilient fallback strategy
- CoNLL-U post-processing
- INCEpTION import

## Scripts

### `parse_text_ud.py`

Runs CAMeL Parser in Universal Dependencies mode and exports CoNLL-U annotations.
The script implements a resilient parsing strategy:
* segments successfully parsed by CAMeL Parser are preserved unchanged;
* segments that cannot be parsed are automatically converted into valid CoNLL-U fallback structures;
* parsing failures are logged for subsequent inspection.
This approach prevents complete pipeline failure while preserving problematic passages for manual validation.

### `conllu_fixing.py`

Post-processes generated CoNLL-U files to improve compatibility with INCEpTION.
In particular, it repairs malformed empty values occurring in FEATS and MISC fields.

## Parsing Strategy

The pipeline uses CAMeL Parser with BERT-based disambiguation and Universal Dependencies output.
Potential parsing failures are retained as annotatable material rather than removed from the workflow. This design choice allows problematic linguistic phenomena, OCR artefacts, and non-standard forms to remain visible during subsequent annotation stages.

## Output

The resulting CoNLL-U files are intended as automatically generated annotations and constitute the starting point for manual validation in INCEpTION.
They should not be considered gold-standard linguistic annotations.

````bibtex
@inproceedings{gugliotta-tarquini-AIUCD, 
    title={Ça y est : Annotating Arabic Texts for Teaching}, 
    author={Gugliotta, Elisa and Tarquini, Maura}, 
    booktitle={ Digitale e Public Engagement - Pratiche e prospettive nelle DIgital Humanities (AIUCD 2026)}, 
    year={2026},
    url = {https://zenodo.org/records/20785399}
}
