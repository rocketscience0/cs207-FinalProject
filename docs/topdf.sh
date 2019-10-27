# Shell script to convert a markdown document to PDF
# Converts a document to word from markdown
FILENAME=${1%.*}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pandoc \
    --standalone \
    --output=$FILENAME.tex \
    --wrap=none \
    --include-in-header=default-header.tex \
    $1 \
    --filter=pandoc-minted.py \
    # --pdf-engine=/usr/local/texlive/2018/bin/x86_64-darwin/lualatex
    # --filter="pandoc-crossref" \
    # --filter="pandoc-citeproc" \
    # --reference-doc="$EXAMPLE_FILE" \
    # --metadata link-citations=true \
    # --number-sections \
    # --toc