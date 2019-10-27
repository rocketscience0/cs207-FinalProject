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

lualatex -shell-escape $FILENAME
lualatex -shell-escape $FILENAME