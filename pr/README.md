# Images for the mapping fix PRs

Each image: upstream develop (6f9e2ce6) | develop with the PR's change | both on a grid of point equilibria, with
the checked disagreements marked (same method as the top-level README, with "cg" replaced by the PR branch).
Both maps use upstream's own solver; the runs are deterministic (PYTHONHASHSEED=0).

## mapping-fixes/near-invariants

Census figures: upstream develop | this PR | this PR's mapping with the cg backend's global-min search (the only
difference from the PR) | point equilibria (cg) on a 0.005 grid, refined where neighbouring points cannot be adjacent.
Circled: regions of the truth grid (more than 20 points) that the map does not cover, each confirmed by point
equilibria with both develop (pdens 2000) and cg. Red X: lines or tie triangles whose area holds other phase sets. The sections are 1 K above or below four-phase invariants found
from the databases, or cases from the 82-case set; chosen as the largest remaining misses and spurious items.
