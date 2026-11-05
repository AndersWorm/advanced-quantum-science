#!/bin/bash
# Trin 1: cd ind i projektmappen
# Trin 2: bash setup.sh
# Trin 3: source .venv/bin/activate
# Virker det ikke i terminalen, så genstart VS Code
# cd .. går en mappe op

cd "$(dirname "$0")"

# Installer uv hvis det mangler (kræver ikke admin)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Lav environment med Python 3.12 (uv henter selv Python)
[ -d .venv ] || uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt -q

echo "Faerdig! Environment er klar."
.venv/bin/python --version