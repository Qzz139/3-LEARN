#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import base64,mimetypes
from pathlib import Path
text=Path('slides.md').read_text()
for image in Path('public/assets').iterdir():
    text=text.replace('/assets/'+image.name,'data:'+mimetypes.guess_type(image.name)[0]+';base64,'+base64.b64encode(image.read_bytes()).decode())
Path('slides.build.md').write_text(text)
PY
trap 'rm -f slides.build.md' EXIT
slidev_cli=${SLIDEV_BIN:-./node_modules/.bin/slidev}
"$slidev_cli" build slides.build.md --base ./ --router-mode hash --out dist --without-notes
cp dist/index.html apple-basic.html
browser_options=()
if [[ -n "${CHROME_BIN:-}" ]]; then browser_options=(--executable-path "$CHROME_BIN"); fi
"$slidev_cli" export slides.build.md --format pdf --output apple-basic.pdf --wait 1000 "${browser_options[@]}"
