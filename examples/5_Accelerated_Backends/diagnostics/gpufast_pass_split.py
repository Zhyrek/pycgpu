"""gpu vs gpu-fast pass-split decision table.

Runs profile_pipeline.py for each requested system on the plain 'gpu'
backend and on 'gpu-fast', and reduces the stage breakdowns to the one
question the gpu-fast roadmap hangs on:

    On THIS card, is the solve dominated by the lockstep pass-1 kernel
    (everyone iterating together) or by the pass-2 faithful re-solve of
    the stragglers (conditions that hit the pass-1 iteration cap)?

How to read the table:
  - "pass 2" row dominating the gpu-fast solve  ->  straggler serial
    chains bound you. The next algorithmic step is a straggler solver
    (semismooth-Newton formulation, see project notes) or more pass-1
    budget; raw kernel-rate improvements won't help much.
  - "pass 1" row dominating  ->  bulk evaluation throughput bounds you.
    The next step is evaluation-rate work (tensorized/GEMM evaluation,
    memory-coalescing) rather than a smarter straggler algorithm.
  - 'gpu' column vs 'gpu-fast' column: sanity check that the two-pass
    split actually beats (or at least matches) the faithful backend
    before reading anything into its internals.

Usage:
    python gpufast_pass_split.py                          # all three systems
    python gpufast_pass_split.py --systems ternary        # AlCuFe only
    python gpufast_pass_split.py --budget 120             # seconds per run

Each cell is a separate profile_pipeline.py subprocess run under the
given wall-clock budget (grid sized to fill it), so a full 3-system
matrix costs about 6x the budget plus compilation on the first touch of
each system.
"""
import argparse
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

_ROW_RE = re.compile(r'^  (.+?)\s{2,}([\d.]+)s\s+\(\s*([\d.]+)%\)\s*$')
_HDR_RE = re.compile(r'^=== .+ — (\d+) conditions, ([\d.]+)s wall')


def run_profile(system, backend, budget):
    cmd = [sys.executable, os.path.join(_HERE, 'profile_pipeline.py'),
           system, '--backend', backend, '--budget', str(budget)]
    print(f'[run] {system} / {backend} (budget {budget:.0f}s)...', flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(f'[run] FAILED (exit {proc.returncode}); last output:')
        for line in out.strip().splitlines()[-15:]:
            print('   ', line)
        return None
    rows = {}
    n_conds = total = None
    for line in out.splitlines():
        mh = _HDR_RE.match(line)
        if mh:
            n_conds, total = int(mh.group(1)), float(mh.group(2))
        mr = _ROW_RE.match(line)
        if mr:
            rows[mr.group(1).strip()] = float(mr.group(2))
    if n_conds is None:
        print('[run] could not parse profile output; raw tail:')
        for line in out.strip().splitlines()[-15:]:
            print('   ', line)
        return None
    return {'rows': rows, 'n': n_conds, 'total': total}


def _find(rows, *needles):
    for label, secs in rows.items():
        if all(n in label for n in needles):
            return label, secs
    return None, 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--systems', nargs='+', default=['binary', 'ternary', 'quaternary'],
                    choices=['binary', 'ternary', 'quaternary'])
    ap.add_argument('--budget', type=float, default=60.0,
                    help='wall-clock budget (s) per profile run (default 60)')
    args = ap.parse_args()

    results = {}
    for system in args.systems:
        for backend in ('gpu', 'gpu-fast'):
            results[(system, backend)] = run_profile(system, backend, args.budget)

    print('\n' + '=' * 72)
    print('PASS-SPLIT DECISION TABLE')
    print('=' * 72)
    for system in args.systems:
        gpu = results[(system, 'gpu')]
        fast = results[(system, 'gpu-fast')]
        print(f'\n--- {system} ---')
        if gpu:
            _, solve = _find(gpu['rows'], 'kernel')
            print(f"  gpu       : {gpu['total']:8.2f}s total, {1000*gpu['total']/gpu['n']:7.3f} ms/cond "
                  f"({gpu['n']} conds; solve {solve:.2f}s)")
        if fast:
            _, p1 = _find(fast['rows'], 'pass 1')
            lbl2, p2 = _find(fast['rows'], 'pass 2')
            m = re.search(r'(\d+) conds', lbl2 or '')
            n_strag = int(m.group(1)) if m else 0
            solve = p1 + p2
            print(f"  gpu-fast  : {fast['total']:8.2f}s total, {1000*fast['total']/fast['n']:7.3f} ms/cond "
                  f"({fast['n']} conds)")
            if solve > 0:
                print(f"      pass 1 (lockstep kernel)  : {p1:8.2f}s  ({100*p1/solve:5.1f}% of solve)")
                print(f"      pass 2 (faithful rerun)   : {p2:8.2f}s  ({100*p2/solve:5.1f}% of solve, "
                      f"{n_strag} stragglers = {100*n_strag/fast['n']:.1f}% of conditions)")
                if p2 > p1:
                    print('      -> PASS 2 DOMINATES: straggler serial chains bound this card.')
                    print('         Next lever: straggler solver (semismooth) / pass-1 budget tuning.')
                else:
                    print('      -> PASS 1 DOMINATES: bulk evaluation throughput bounds this card.')
                    print('         Next lever: evaluation rate (tensorized GEMM eval, coalescing).')
            else:
                print('      (no two-pass split rows found — PYCGPU_PASS1_ITERS may be'
                      ' disabled, or all conditions converged inside pass 1)')
        if gpu and fast and fast['total'] and gpu['total']:
            r_gpu = gpu['total'] / gpu['n']
            r_fast = fast['total'] / fast['n']
            print(f'  gpu-fast per-condition speedup vs gpu: {r_gpu / r_fast:5.2f}x')


if __name__ == '__main__':
    main()
