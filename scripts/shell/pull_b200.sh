#!/bin/bash
set -euo pipefail

export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export NO_PROXY=.byted.org
export UV_HTTP_TIMEOUT=1000000

git pull
git submodule sync --recursive
git submodule update --init --recursive

