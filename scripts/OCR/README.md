# QARI OCR

This directory contains the OCR workflow used in the RĀBIṬA project (*Resources for Arabic: Bridging Interoperable Texts and Annotation*).

## Model

The OCR pipeline relies on:

* **QARI OCR** (`NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct`)
* **Qwen2-VL** architecture

## Environment

The workflow was developed in a dedicated Python virtual environment (`venv_qari`).

Core dependencies are listed in `requirements.txt`.

A complete snapshot of the original environment is available in `requirements_full.txt`.

## Main Features

* Page-level OCR extraction from images
* JSONL export for page-aligned outputs
* Unicode normalization
* Optional image enhancement (autocontrast and sharpening)
* HTML and OCR artefact cleaning
* OCR post-processing and text normalization

## Input

Supported image formats include:

* PNG
* JPG / JPEG
* TIFF
* BMP
* WEBP

## Output

The script can generate:

* Plain text files (`.txt`)
* Page-level OCR outputs (`.jsonl`)

These outputs can subsequently be processed through the Human-in-the-Loop validation platform and the linguistic annotation pipeline.

## Related Publication

This workflow is part of the RĀBIṬA project, which explores the creation of reusable Arabic literary corpora combining OCR, manual validation, linguistic annotation, and pedagogical reuse.


### *Citation* 

Please cite this work as: 


````bibtex

@article{tarquini2025ocr,
  title={From OCR to Content Interpretation: Towards a Scalable Workflow for Arabic Literature in the Digital Humanities},
  author={Tarquini, Maura and Gugliotta, Elisa},
  journal={Testo e Senso},
  volume={1},
  number={29},
  pages={115--128},
  year={2025},
  url = {https://testoesenso.it/index.php/testoesenso/article/view/806}
}

@inproceedings{gugliotta-tarquini-AIUCD, 
    title={Ça y est : Annotating Arabic Texts for Teaching}, 
    author={Gugliotta, Elisa and Tarquini, Maura}, 
    booktitle={ Digitale e Public Engagement - Pratiche e prospettive nelle DIgital Humanities (AIUCD 2026)}, 
    year={2026},
    pages={658--664},
    url = {https://zenodo.org/records/20785399}
}



````
