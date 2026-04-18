#!/bin/bash
# upload all GGUF model repositories under models/gguf/ to huggingface.
#
# each subdirectory of models/gguf/ becomes an HF repo at
# <HF_ORG>/<dirname>. repos are created (private by default) if missing
# and uploaded non-recursively via `hf upload`.
#
# requirements: hf cli, authenticated via `hf auth login`
#
# repos are also added to a HuggingFace collection (default:
# <HF_ORG>/qwen3-tts titled "Qwen3-TTS"), creating it if missing.
#
# usage:
#   ./scripts/upload_gguf_to_hf.sh [--org ORG] [--public] [--dry-run]
#                                  [--collection SLUG] [--collection-title TITLE]
#                                  [--no-collection] [repo ...]
#
# examples:
#   ./scripts/upload_gguf_to_hf.sh
#   ./scripts/upload_gguf_to_hf.sh --org khimaros --public
#   ./scripts/upload_gguf_to_hf.sh Qwen3-TTS-12Hz-0.6B-Base-GGUF

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
GGUF_DIR="${REPO_DIR}/models/gguf"

HF_ORG="${HF_ORG:-khimaros}"
COLLECTION_TITLE="${COLLECTION_TITLE:-Qwen3-TTS}"
COLLECTION_SLUG="${COLLECTION_SLUG:-${HF_ORG}/qwen3-tts}"
PRIVATE="true"
DRY_RUN="false"
SELECTED=()

while [ $# -gt 0 ]; do
    case "$1" in
        --org) HF_ORG="$2"; shift 2 ;;
        --collection) COLLECTION_SLUG="$2"; shift 2 ;;
        --collection-title) COLLECTION_TITLE="$2"; shift 2 ;;
        --no-collection) COLLECTION_SLUG=""; shift ;;
        --public) PRIVATE="false"; shift ;;
        --private) PRIVATE="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) SELECTED+=("$1"); shift ;;
    esac
done

if ! command -v hf >/dev/null 2>&1; then
    echo "error: hf cli not found. install via 'pip install huggingface_hub'" >&2
    exit 1
fi

if [ ! -d "$GGUF_DIR" ]; then
    echo "error: $GGUF_DIR does not exist" >&2
    exit 1
fi

run() {
    echo "  \$ $*"
    if [ "$DRY_RUN" != "true" ]; then
        "$@"
    fi
}

if [ ${#SELECTED[@]} -eq 0 ]; then
    mapfile -t SELECTED < <(cd "$GGUF_DIR" && find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
fi

echo "=== uploading to https://huggingface.co/$HF_ORG ==="
echo "  private=$PRIVATE dry_run=$DRY_RUN collection=${COLLECTION_SLUG:-<none>}"
echo ""

if [ -n "$COLLECTION_SLUG" ]; then
    echo "--- ensuring collection $COLLECTION_SLUG ---"
    run hf collections create "$COLLECTION_TITLE" \
        --namespace "$HF_ORG" \
        --exists-ok || true
    echo ""
fi

for name in "${SELECTED[@]}"; do
    src="$GGUF_DIR/$name"
    if [ ! -d "$src" ]; then
        echo "warning: $src not found, skipping"
        continue
    fi

    repo="$HF_ORG/$name"
    echo "--- $repo ---"

    create_args=(--repo-type model --exist-ok)
    if [ "$PRIVATE" = "true" ]; then
        create_args+=(--private)
    fi
    run hf repos create "$repo" "${create_args[@]}"

    run hf upload "$repo" "$src" . \
        --repo-type model \
        --commit-message "upload GGUF artifacts"

    if [ -n "$COLLECTION_SLUG" ]; then
        run hf collections add-item "$COLLECTION_SLUG" "$repo" model --exists-ok || true
    fi

    echo ""
done

echo "=== done ==="
