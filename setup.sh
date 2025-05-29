#!/usr/bin/env bash
set -euo pipefail

# Install dependencies using uv and the provided lock file
uv pip install --system -e .
