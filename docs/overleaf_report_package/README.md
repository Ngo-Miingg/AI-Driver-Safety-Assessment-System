# Overleaf Quick Start

Use this package for the report on Overleaf.

Files:
- `main.tex`: the report source
- `ai_driver_safety_report_matrix_v2.pdf`: local reference render

Overleaf settings:
- Compiler: `pdfLaTeX`
- Main file: `main.tex`

Fast upload flow:
1. Create a new blank project on Overleaf.
2. Upload the contents of `overleaf_report_package.zip`.
3. Set compiler to `pdfLaTeX` and recompile.

Notes:
- The bibliography is already embedded with `thebibliography`, so no `.bib` file is required.
- The current version draws figures directly with TikZ, so no external image assets are required for this report build.
- If Overleaf shows the last page columns slightly uneven, that is normal for IEEEtran drafts. It can be balanced later as a final formatting pass.
