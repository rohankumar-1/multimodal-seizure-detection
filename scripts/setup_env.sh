#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install jupyterlab notebook ipykernel

if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

python -m ipykernel install --user --name repo_env --display-name "Python (repo_env)"

echo "Done. Activate with: source .venv/bin/activate"
echo "Then run: jupyter lab"

# run first: chmod +x scripts/setup_env.sh

# then run: ./scripts/setup_env.sh