# OCR Validation Platform

This directory contains the Human-in-the-Loop validation environment developed within the RĀBIṬA project.

The platform was designed to support the manual review and correction of OCR outputs generated from Arabic literary texts, with particular attention to non-standard varieties, orthographic variation, code-switching, and OCR-induced noise.

## Objectives

The platform aims to:

* facilitate page-by-page OCR validation;
* support the identification and correction of OCR errors;
* document recurring error patterns;
* assist the creation of reusable and FAIR textual resources.

## Main Features

* Side-by-side comparison between OCR output and validated text;
* Interactive correction interface;
* Character copy-helpers for Arabic and Tunisian-specific characters;
* Error highlighting and mismatch visualization;
* Page-level validation workflow;
* Export of validated texts.

## Human-in-the-Loop Approach

Rather than treating OCR errors as simple noise to be removed, the platform makes them visible and documentable.

This approach allows researchers to:

* improve OCR workflows;
* identify recurrent recognition patterns;
* study the interaction between linguistic variation and OCR performance;
* transform OCR errors into reusable annotation material.

## Technical Framework

The validation interface was implemented using:

* Streamlit
* Python
* JSONL-based OCR outputs

## Role within the RĀBIṬA Workflow

OCR → Post-processing → Human Validation → Linguistic Annotation → Pedagogical Reuse

The validated texts produced through this platform constitute the basis for subsequent linguistic annotation and educational activities.

````bibtex

@inproceedings{gugliotta-tarquini-AIUCD, 
    title={Ça y est : Annotating Arabic Texts for Teaching}, 
    author={Gugliotta, Elisa and Tarquini, Maura}, 
    booktitle={ Digitale e Public Engagement - Pratiche e prospettive nelle DIgital Humanities (AIUCD 2026)}, 
    year={2026},
    pages={658--664},
    url = {https://zenodo.org/records/20785399}
}

