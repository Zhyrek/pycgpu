#!/usr/bin/env bash
# Regenerate results/ from the checkouts made by setup_checkouts.sh.
# usage: tools/term-order-comparison/run_all.sh DIR [MPEA_TDB]
#   Without MPEA_TDB only the scramble test runs. The MPEA database needs its "Lin+" reference
#   renamed (e.g. to "Lin5"): pycalphad's TDB parser only accepts alphanumeric reference names.
# Each run is one process at a time; the runs are interleaved across branches to spread machine noise.
set -euo pipefail
DIR=${1:?usage: run_all.sh DIR [MPEA_TDB]}
MPEA=${2:-}
PYTHON=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/results"
mkdir -p "$OUT"
export PYTHONHASHSEED=0

for mode in add addmul; do
  for b in develop 739 bocklund; do
    PYTHONPATH="$DIR/$b" "$PYTHON" "$HERE/scramble.py" "$mode" "$OUT/scramble_${mode}_$b.jsonl"
  done
done

if [ -n "$MPEA" ]; then
  : > "$OUT/timing_stages.jsonl"
  : > "$OUT/timing_eq.jsonl"
  for r in 1 2 3 4; do
    for b in develop 739 bocklund; do
      t=$(PYTHONPATH="$DIR/$b" "$PYTHON" "$HERE/timing.py" "$MPEA" stages)
      echo "{\"branch\":\"$b\",\"r\":$r,\"t\":$t}" >> "$OUT/timing_stages.jsonl"
    done
  done
  for r in 1 2 3; do
    for b in develop 739 bocklund; do
      t=$(PYTHONPATH="$DIR/$b" "$PYTHON" "$HERE/timing.py" "$MPEA" eq)
      echo "{\"branch\":\"$b\",\"r\":$r,\"t\":$t}" >> "$OUT/timing_eq.jsonl"
    done
  done
fi
"$PYTHON" "$HERE/summarize.py" "$OUT"
