#!/bin/bash
# Exporta la memoria LaTeX a Word usando pandoc
# Uso: bash export_to_word.sh [output.docx]
PANDOC="/c/Users/jmmachado/AppData/Local/Microsoft/WinGet/Packages/JohnMacFarlane.Pandoc_Microsoft.Winget.Source_8wekyb3d8bbwe/pandoc-3.9.0.2/pandoc.exe"
OUTPUT="${1:-memoria/TFM_Entrega2.docx}"

cd "$(dirname "$0")"
"$PANDOC" memoria/main.tex \
  --from=latex \
  --to=docx \
  --output="$OUTPUT" \
  --wrap=none \
  && echo "Exportado: $OUTPUT" || echo "ERROR en la exportacion"
