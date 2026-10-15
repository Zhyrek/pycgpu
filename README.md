# Mapping: develop vs the column-generation backend ('cg')

57 binary and ternary maps (`BinaryStrategy` / `TernaryStrategy`) drawn with upstream develop
(1a1732b7) and with the fork's `cg` backend (branch `cg`, b15b25ba). Each map was checked against
point equilibria.

- Top-level images: the 24 systems where something was flagged.
- `agree/`: the 33 where nothing was flagged.
- `summary.md`: per system, the checked disagreement regions, the share of each map's boundary missing
  from the other, line counts and timings.
- Systems: the test databases in `pycalphad/tests/databases`, and the eight binaries added on
  upstream PR 722's branch (AlTe-19Shi, AuCu-98Sun-LB, BaCa-86Alc, CaMg-06Zho, CrPt-98Spe-LB,
  CrTa-93Dup-LB, CrV-92Lee-LB, GaLa-11Idb). The other systems in PR 722's comparison images come from
  the NIMS CPDDB collection (registration required) and are not included.

## Panels

1. develop's map.
2. cg's map.
3. Both maps' boundaries over a grid of point equilibria.
   - **Lines:** develop's in blue, cg's in red dashed.
   - **Background:** each grid point coloured by the phase assemblage of the lower-energy result
     (develop at pdens 500, or cg).
   - **Small grey ×:** a grid point where develop's point equilibrium is more than 1 J/mol too high.

Tie triangles in ternaries are drawn with solid edges.

## Checks

### Local check of every disagreement (large markers)

- **Finding regions.** Boundary points of one map that are farther than 0.02 from the other map's
  boundaries are grouped into regions.
- **Local truth.** Around up to four points per region, a fine grid of point equilibria is computed with
  both develop (pdens 1000) and cg, taking the lower-energy result.
- **Equivalent phases.** Equivalent descriptions of one phase (equal energies at disordered site
  fractions, e.g. B2_BCC / BCC_VA / BCC_4SL / BCC_NOB in Sundman's Al-Fe, or FCC_4SL / FCC_A1) are
  treated as the same phase.
- **When a boundary counts as real.** Only when it can be a ZPF line, i.e. the phase assemblage changes
  next to it in one of these ways:
  - it borders a single-phase region;
  - in a binary, it separates two two-phase regions across composition (a stoichiometric compound's
    line);
  - in a ternary, it separates two two-phase regions.

  These do not count:
  - in a binary, a change between two two-phase regions across temperature (an invariant);
  - in a ternary, a change between a two-phase and a three-phase region (a tie-triangle edge).
- **Markers.**
  - × blue: develop boundary that is not there.
  - ○ green: real boundary develop misses.
  - ◆ red: where cg is wrong.

### Full-grid check (small markers)

The same comparison against the whole background grid. It also catches errors both maps share.
- **P marker:** a true boundary with no map line within 0.03.
- **X marker:** a map line with no change of phase assemblage within 0.05.

Markers are blue for develop and red for cg. Points within 0.006 of the diagram edges are not judged,
because the grid starts 0.005 from each edge.

## Result

| | Regions |
|---|---|
| develop boundary spurious | 22 |
| real boundary develop misses | 13 |
| cg wrong | 1 |

The one cg region is a short line (8 points) along a tie-triangle edge in Al-Cr-Ni 1200 K (alcrni.tdb).

Full-grid findings that both maps share:
- the BCC_B2 + liquid field above 2200 K in Co-Cr (a database artifact, far above the melting point of
  Co);
- a small region near x(Co) = 0.95 at 1150 K in Al-Co;
- lines in Ti-V near x = 0.71, 600 K and in Cr-Ta at x = 0.333.

Flags on Fe-Ni's sliver at x = 0.403 and on single grid points in Al-Fe are the grid's resolution: the
sliver is narrower than the 0.01 grid spacing, and the Al-Fe flags come from isolated noisy points.

### What changed in cg's mapping (commits 67abee16 and b15b25ba on `cg`)

- **Ordering transitions:** first-order constitution jumps and composition-basin switches count as
  failed steps, so mapping steps over neither. A set that turns into a better ordering variant keeps
  its line going. Ni-Pt's L1_0 + L1_2 band now reaches 920 K, and Co-V's FCC + FCC region is drawn.
- **Critical points:** miscibility gaps close at their critical point (Al-Zn).
- **Equivalent descriptions:** descriptions of one phase with different names are recognised as the
  same state. Al-Fe's B2 + liquid boundary now runs to the melting point of Fe, with no zero-width
  A2/B2 line.
- **Doubled lines:** a loop check in mixed units (K against mole fraction) ended lines early, which
  were then traced twice; this caused the doubled C15 + liquid line in Cr-Nb.
- **Ternary exits:** exits near diagram edges try the other axis, and metastable nodes are rebuilt with
  the right candidate (Al-Cr-Ni B2 + B2 lines).

The mapping-level changes (step retries, loop check, variant switches, node retry, edge exits) are in
the shared mapping code, so they also apply to the fork's other backends. The develop maps here are
upstream develop, unchanged; the doubled Cr-Nb line, for example, is still visible in develop's panel.

## Limits

- "Truth" is the lower-energy result of two solvers on a grid: about 0.005 spacing for the local check,
  0.01–0.02 for the full grid. A two-phase field narrower than that can be missed by the check.
- Databases with several descriptions of one phase can still give overlapping lines for one boundary.
  Example: Sundman's Al-Fe traces its BCC + liquid boundary with both BCC_VA and B2_BCC.
