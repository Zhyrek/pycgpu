# Mapping: develop vs the column-generation backend ('cg')

57 binary and ternary maps (`BinaryStrategy` / `TernaryStrategy`) drawn with upstream develop
(1a1732b7) and with the fork's `cg` backend, and every place where the two disagree checked against
point equilibria.

- Top-level images: the 18 systems where the maps disagree somewhere. `agree/`: the 39 where they
  agree (no boundary of one map farther than 0.02 from the other's).
- `summary.md`: per system, the checked disagreement regions, the share of each map's boundary
  missing from the other, line counts and timings.
- Systems: the test databases in `pycalphad/tests/databases`, and the eight binaries added on
  upstream PR 722's branch (AlTe-19Shi, AuCu-98Sun-LB, BaCa-86Alc, CaMg-06Zho, CrPt-98Spe-LB,
  CrTa-93Dup-LB, CrV-92Lee-LB, GaLa-11Idb). The other systems in PR 722's comparison images come from
  the NIMS CPDDB collection (registration required) and are not included.

## Panels

1. develop's map.
2. cg's map.
3. Both maps' boundaries (develop blue, cg red dashed) over a grid of point equilibria. Each grid
   point is coloured by the phase assemblage of the lower-energy result: develop at pdens 500, or cg.
   A small grey × marks a point where develop's point equilibrium is more than 1 J/mol too high; a
   black + would mark the same for cg.

## How disagreements were checked

- **Finding regions.** Boundary points of one map that are farther than 0.02 from the other map's
  boundaries (in composition, and temperature as a fraction of the plotted range) are grouped into
  regions.
- **Local truth.** Around up to four points per region, a local grid of point equilibria is computed:
  11×7 for a binary, 11×11 for a ternary, ±0.025, with both develop (pdens 1000) and cg. At each grid
  point the lower-energy result is taken.
- **Equivalent phases.** Equivalent descriptions of one phase (equal energies at disordered site
  fractions, e.g. B2_BCC / BCC_VA / BCC_4SL / BCC_NOB in Sundman's Al-Fe, or FCC_4SL / FCC_A1) are
  treated as the same phase.
- **Verdict.** A boundary is real if the phase assemblage changes within 0.012 of the point. Large
  markers in panel 3:
  - × blue: a develop boundary that is not there.
  - ○ green: a real boundary that develop misses.
  - ◆ red: where cg is wrong. None remain.

## Result

| | Regions |
|---|---|
| develop boundary spurious | 22 |
| real boundary develop misses | 13 |
| cg wrong | 0 |

Before the fixes below, cg missed 5 real boundaries:
- the low-temperature ordering field in Ni-Pt;
- two B2 + B2 regions in Al-Cr-Ni at 1200 K (alcrni.tdb);
- two regions that turned out to be equivalent-phase naming artifacts in the verifier.

### What changed to get there

- **cg's local solver (mapping):** a composition set whose site fractions jump far more than its
  composition in one step (an ordered set falling into the disordered state across a first-order
  transition) now counts as not converged. Mapping then takes a smaller step instead of stepping over
  the two-phase field. Two composition sets of one phase converging onto each other also count as not
  converged.
- **Ternary mapping:** a node exit whose stepping axis immediately runs off the diagram (a line close to
  an edge) now tries the other axis variable.
- **Node finding:** when a new node is metastable to another candidate phase, the node is rebuilt from
  the same step point with that candidate. Before, the whole ZPF line was discarded.

The last two are in the shared mapping code, so they apply to every backend.

## Limits

- Only places where the two maps disagree were checked. A boundary that both maps miss, or both draw
  wrongly, is not flagged.
- "Truth" is the lower-energy result of two solvers on a local grid with a spacing of about 0.005. A
  two-phase field narrower than that can be missed by the check.
