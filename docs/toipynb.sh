# Shell script to convert a markdown document to a jupyter notebook
# Converts a document to word from markdown
FILENAME=${1%.*}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pandoc \
    --standalone \
    --output=$FILENAME.ipynb \
    --wrap=none \
    --atx-headers \
    --include-in-header=default-header.tex \
    $1