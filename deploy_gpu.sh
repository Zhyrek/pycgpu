#!/bin/bash
#
# deploy_gpu.sh — Sync GPU-enabled pycalphad files to a remote cluster.
#
# Reads the remote target from path.txt (e.g., user@server:~/path/to/pycalphad)
# and uses a single rsync invocation (one SSH connection, one credential prompt).
#
# Usage:
#   ./deploy_gpu.sh                  # Deploy library files
#   ./deploy_gpu.sh --dry-run        # Preview what would be sent
#   ./deploy_gpu.sh --with-tests     # Also sync important_tests/
#   ./deploy_gpu.sh --dry-run --with-tests
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATH_FILE="$SCRIPT_DIR/../PYCALPHAD_path.txt"
PYCALPHAD_SRC="$SCRIPT_DIR/pycalphad"
TESTS_SRC="$SCRIPT_DIR/important_tests"

# --- Parse arguments ---
DRY_RUN=""
WITH_TESTS=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN="--dry-run" ;;
        --with-tests) WITH_TESTS=true ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--with-tests]"
            echo ""
            echo "  --dry-run      Preview what would be transferred (no actual transfer)"
            echo "  --with-tests   Also sync important_tests/ alongside the pycalphad package"
            echo ""
            echo "Reads remote target from path.txt in the same directory as this script."
            echo "path.txt should contain a single line like:"
            echo "  user@server.com:~/miniconda3/envs/myenv/lib/python3.x/site-packages/pycalphad"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Run $0 --help for usage."
            exit 1
            ;;
    esac
done

# --- Validate path.txt ---
if [[ ! -f "$PATH_FILE" ]]; then
    echo "ERROR: $PATH_FILE not found."
    echo "Create it with a single line containing the remote pycalphad package path, e.g.:"
    echo "  user@server.com:~/miniconda3/envs/myenv/lib/python3.x/site-packages/pycalphad"
    exit 1
fi

REMOTE=$(head -n1 "$PATH_FILE" | tr -d '[:space:]')

if [[ -z "$REMOTE" ]]; then
    echo "ERROR: path.txt is empty."
    exit 1
fi

# Validate format: should contain user@host:path
if [[ ! "$REMOTE" == *@*:* ]]; then
    echo "ERROR: path.txt doesn't look like a valid remote path."
    echo "Expected format: user@server.com:/path/to/pycalphad"
    echo "Got: $REMOTE"
    exit 1
fi

# --- Build file list ---
# This list is relative to pycalphad/ (the package root, not the repo root).
# We dynamically find all files under gpu/, excluding __pycache__ and .pyc,
# and include core/equilibrium.py which has the gpu=True entrypoint.

FILELIST=$(mktemp)
trap "rm -f $FILELIST" EXIT

# All gpu/ files (working tree), excluding caches
(cd "$PYCALPHAD_SRC" && find gpu -type f \
    ! -path '*__pycache__*' \
    ! -name '*.pyc' \
    ! -name '*.pyo' \
) >> "$FILELIST"

# The modified equilibrium.py with gpu=True support
echo "core/equilibrium.py" >> "$FILELIST"

FILE_COUNT=$(wc -l < "$FILELIST")

echo "=== pycalphad GPU deploy ==="
echo "Remote target: $REMOTE"
echo "Files to sync: $FILE_COUNT"
if [[ -n "$DRY_RUN" ]]; then
    echo "Mode: DRY RUN (no files will be transferred)"
fi
echo ""

# --- Deploy library files ---
echo "--- Syncing pycalphad library files ---"
rsync -avz $DRY_RUN \
    --files-from="$FILELIST" \
    "$PYCALPHAD_SRC/" \
    "$REMOTE/"

# --- Optionally deploy tests ---
if $WITH_TESTS; then
    echo ""
    echo "--- Syncing important_tests/ ---"

    if [[ ! -d "$TESTS_SRC" ]]; then
        echo "WARNING: important_tests/ directory not found, skipping."
    else
        # Extract user@host and base path from REMOTE
        REMOTE_USERHOST="${REMOTE%%:*}"
        REMOTE_PATH="${REMOTE#*:}"
        # Place tests as a sibling directory: ../important_tests/
        REMOTE_TESTS="${REMOTE_USERHOST}:${REMOTE_PATH}/../important_tests/"

        TESTS_FILELIST=$(mktemp)
        trap "rm -f $FILELIST $TESTS_FILELIST" EXIT

        # Sync test scripts and data files, excluding caches and generated kernels
        (cd "$TESTS_SRC" && find . -type f \
            ! -path '*__pycache__*' \
            ! -path '*.pycgpu_kernels*' \
            ! -name '*.pyc' \
            ! -name '*.pyo' \
        ) >> "$TESTS_FILELIST"

        TESTS_COUNT=$(wc -l < "$TESTS_FILELIST")
        echo "Test files to sync: $TESTS_COUNT"
        echo "Test target: $REMOTE_TESTS"

        rsync -avz $DRY_RUN \
            --files-from="$TESTS_FILELIST" \
            "$TESTS_SRC/" \
            "$REMOTE_TESTS"
    fi
fi

echo ""
if [[ -n "$DRY_RUN" ]]; then
    echo "Dry run complete. Re-run without --dry-run to transfer files."
else
    echo "Deploy complete."
fi
