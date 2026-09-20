# Local data layout

Source datasets and generated indexes are intentionally kept out of Git.

```text
data/
├── raw/                         Korean Basic Dictionary JSON exports
├── processed/
│   ├── grammar_entries.json     intermediate dictionary records (ignored)
│   └── grammar_clean.json       compact derived grammar data (ignored)
├── chroma/                      generated dictionary vector index (ignored)
└── topik/
    ├── sources.json             small source manifest (committed)
    ├── raw/                     user-provided TOPIK PDFs (ignored)
    ├── ocr_models/              downloaded EasyOCR weights (ignored)
    └── processed/               extracted text, crops, and OCR cache (ignored)
```

The repository does not redistribute Korean Basic Dictionary bulk exports or
TOPIK papers. See the root `README.md` and `DATA_NOTICE.md` for setup and source
information.
