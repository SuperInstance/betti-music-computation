# betti-music-computation

Persistent homology computation for musical traditions — computes Betti numbers, persistence diagrams, and statistical significance across cultural distance matrices.

## What This Gives You

- **Betti curves** (β₀, β₁, β₂) computed from cultural distance matrices via Vietoris-Rips complexes
- **Persistence diagrams** with visual output for each homology dimension (H₀, H₁, H₂)
- **Null model comparison** — p-values testing whether observed topology exceeds random expectation
- **3D tradition visualization** — spatial embedding of cultural traditions with topological structure

## Quick Start

```bash
pip install numpy scipy matplotlib
python compute_betti.py
```

Outputs:
- `betti_curves.png` / `betti_curves.csv` — Betti number evolution across filtration
- `persistence_diagrams.png` / `persistence_H{0,1,2}.png` — Persistence diagrams per dimension
- `null_comparison.png` — Observed vs. null model comparison
- `p_values.json` — Statistical significance of topological features
- `distance_matrix.csv` — Computed distance matrix

## How It Fits

Part of the SuperInstance conservation-spectral ecosystem. Computes the topological invariants that `conservation-tomography` and `dial-space-explorer` visualize. The Betti numbers feed into `conservation-protocol` for cultural heritage preservation decisions.

## License

MIT
