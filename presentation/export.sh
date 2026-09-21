#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
python3 - "$build_dir/slides.md" <<'PY'
import sys,base64,mimetypes
from pathlib import Path
text=Path('sorting-presentation.md').read_text(encoding='utf-8')
for image in Path('assets').iterdir():
    data='data:'+mimetypes.guess_type(image.name)[0]+';base64,'+base64.b64encode(image.read_bytes()).decode()
    text=text.replace('assets/'+image.name,data)
Path(sys.argv[1]).write_text(text,encoding='utf-8')
PY
if [[ -n "${MARP_BIN:-}" ]]; then
    marp_command=("$MARP_BIN")
else
    marp_command=(npx --yes @marp-team/marp-cli@4.5.1)
fi
"${marp_command[@]}" "$build_dir/slides.md" --html --theme "$PWD/theme.css" -o "$PWD/sorting-presentation.html"
"${marp_command[@]}" "$build_dir/slides.md" --html --theme "$PWD/theme.css" --pdf -o "$PWD/sorting-presentation.pdf"
