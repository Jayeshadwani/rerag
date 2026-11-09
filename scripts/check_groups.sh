#!/bin/sh
# Installs ONLY the dependency groups an image gets and runs what that image runs (proposal 009).
#   build-image: the Dockerfile.api index stage (sync, quantize-onnx, build-index)
#   api-image:   the running API
# Usage: scripts/check_groups.sh [build-image|api-image]   (default: both)
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

install_only() {  # group
    uv venv -q "$work/$1"
    uv export --frozen --no-hashes --no-emit-project --only-group "$1" -o "$work/$1.txt" >/dev/null
    uv pip install -q --python "$work/$1/bin/python" -r "$work/$1.txt"
}

check_build_image() {
    install_only build-image
    for args in "--help" "sync --help" "quantize-onnx --help" "build-index --help"; do
        echo "build-image: python -m noterecall.cli $args"
        # shellcheck disable=SC2086
        PYTHONPATH="$root/src" "$work/build-image/bin/python" -m noterecall.cli $args >/dev/null
    done
}

check_api_image() {
    install_only api-image
    echo "api-image: import noterecall.api"
    PYTHONPATH="$root/src" "$work/api-image/bin/python" -c "import noterecall.api, noterecall.serve.runtime"
}

case "${1:-all}" in
    build-image) check_build_image ;;
    api-image) check_api_image ;;
    all) check_build_image; check_api_image ;;
    *) echo "unknown group $1" >&2; exit 2 ;;
esac
echo "groups: ok"
