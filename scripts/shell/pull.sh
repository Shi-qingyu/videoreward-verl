#!/bin/bash
set -euo pipefail

git pull
git submodule sync --recursive
git submodule update --init --recursive

