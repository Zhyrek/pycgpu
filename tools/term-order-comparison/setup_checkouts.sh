#!/usr/bin/env bash
# Build three side-by-side checkouts of pycalphad, as git worktrees of the current repository:
#   DIR/develop    upstream develop
#   DIR/739        develop + PR #739's two files (pycalphad/pycalphad#739, head ce030350)
#   DIR/bocklund   develop + bocklund/pycalphad claude/bit-identical-phase-records (9043dd3a), cherry-picked
# The fixes are left as uncommitted changes. Neither touches Cython, so the extensions are built once,
# in DIR/develop, and copied into the other two.
#
# usage: tools/term-order-comparison/setup_checkouts.sh DIR [BASE]
#   BASE defaults to f11ec4c9, the develop commit the results in results/ were measured on.
set -euo pipefail
DIR=${1:?usage: setup_checkouts.sh DIR [BASE]}
BASE=${2:-f11ec4c9921220942fde2239b5b1efeec3bb74a1}
PR739=ce030350a6aad34c135a6fa5d49c58d86ebb6ddb
BOCKLUND=9043dd3ac5ff109fb33eb3e797e6f9a5b1ccc3d3
PYTHON=${PYTHON:-python}

git fetch https://github.com/pycalphad/pycalphad.git develop pull/739/head
git fetch https://github.com/bocklund/pycalphad.git claude/bit-identical-phase-records
mkdir -p "$DIR"

git worktree add --detach "$DIR/develop" "$BASE"
git worktree add --detach "$DIR/739" "$BASE"
git worktree add --detach "$DIR/bocklund" "$BASE"

# PR #739 predates #741, which also edits build_functions' decorator: take the PR's version of its
# two files, then restore #741's lru_cache(maxsize=512) if the base has it.
(
  cd "$DIR/739"
  git checkout "$PR739" -- pycalphad/codegen/sympydiff_utils.py pycalphad/tests/test_codegen.py
  if git show "$BASE:pycalphad/codegen/sympydiff_utils.py" | grep -q '@lru_cache(maxsize=512)'; then
    "$PYTHON" - <<'EOF'
path = "pycalphad/codegen/sympydiff_utils.py"
src = open(path).read()
src = src.replace("from pycalphad.core.cache import cacheit\n", "from pycalphad.core.cache import cacheit, lru_cache\n", 1)
src = src.replace("@cacheit\ndef build_functions(", "@lru_cache(maxsize=512)\ndef build_functions(", 1)
open(path, "w").write(src)
EOF
  fi
)

(cd "$DIR/bocklund" && git cherry-pick --no-commit "$BOCKLUND")

(cd "$DIR/develop" && "$PYTHON" setup.py build_ext --inplace > build.log 2>&1)
(cd "$DIR/develop" && find pycalphad \( -name '*.so' -o -name '*.pyd' \)) | while read -r ext; do
  cp "$DIR/develop/$ext" "$DIR/739/$ext"
  cp "$DIR/develop/$ext" "$DIR/bocklund/$ext"
done
echo "checkouts ready in $DIR; run each with PYTHONPATH=$DIR/<name>"
