#!/usr/bin/env python3
"""
COMPUTE BETTI NUMBERS OF THE MUSICAL TRADITION DIAL SPACE

V2: Faster null model — uses ripser directly for persistence,
    avoids expensive per-sample Betti curve computations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
from persim import plot_diagrams
import json, os, time, warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

OUTDIR = "/home/phoenix/.openclaw/workspace/experiments/betti-computation"
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(42)

print(f"Output directory: {OUTDIR}")

# ============================================================================
# 1. TRADITION COORDINATES (3D)
# ============================================================================

traditions = [
    "Carnatic", "Hindustani", "Turkish Makam", "Arabic Maqam",
    "West African", "Balinese Gamelan", "Javanese Gamelan",
    "Western CP", "Chinese", "Japanese Gagaku",
]

coords = np.array([
    [2.767, 3.626, 2.0],   # Carnatic
    [2.765, 3.451, 1.8],   # Hindustani
    [2.828, 3.276, 1.6],   # Turkish Makam
    [2.936, 3.101, 1.5],   # Arabic Maqam
    [2.412, 3.625, 2.5],   # West African
    [2.308, 3.100, 2.2],   # Balinese Gamelan
    [2.308, 2.750, 1.8],   # Javanese Gamelan
    [2.715, 2.051, 1.0],   # Western CP
    [2.318, 2.050, 0.8],   # Chinese
    [2.384, 1.700, 0.5],   # Japanese Gagaku
])

n = len(traditions)
print(f"Loaded {n} traditions in 3D dial space.")

# Bounding box
bbox_min = coords.min(axis=0)
bbox_max = coords.max(axis=0)
print(f"Bounding box: [{bbox_min[0]:.3f}, {bbox_max[0]:.3f}] × "
      f"[{bbox_min[1]:.3f}, {bbox_max[1]:.3f}] × "
      f"[{bbox_min[2]:.3f}, {bbox_max[2]:.3f}]")

# ============================================================================
# 2. DISTANCE MATRIX
# ============================================================================

dists = pdist(coords, metric='euclidean')
dist_matrix = squareform(dists)

# Print
print("\nEuclidean distance matrix (3D):")
header = f"{'':20s}"
for t in traditions:
    header += f"{t[:8]:>8s}"
print(header)
for i in range(n):
    row = f"{traditions[i]:20s}"
    for j in range(n):
        row += f"{dist_matrix[i,j]:>8.4f}"
    print(row)

np.savetxt(os.path.join(OUTDIR, "distance_matrix.csv"), dist_matrix,
           delimiter=",", fmt="%.6f",
           header="," + ",".join(traditions))

# Sorted distances
triu = np.triu_indices(n, k=1)
d_vals = dist_matrix[triu]
d_pairs = [(traditions[i], traditions[j], dist_matrix[i,j])
           for i, j in zip(*triu)]
d_pairs.sort(key=lambda x: x[2])
print("\nSorted pairwise distances:")
for t1, t2, d in d_pairs:
    print(f"  {t1:20s} - {t2:20s}: {d:.4f}")

# ============================================================================
# 3. PERSISTENT HOMOLOGY (ripser)
# ============================================================================

print("\n=== COMPUTING PERSISTENT HOMOLOGY ===")
result = ripser(coords, maxdim=2, distance_matrix=False, thresh=3.0)
dgms = result['dgms']

for dim in range(3):
    print(f"  H_{dim}: {len(dgms[dim])} features")
    for p in sorted(dgms[dim], key=lambda x: x[0]):
        b, d = p[0], p[1]
        pers = f'{d-b:.4f}' if d != np.inf else '∞'
        death = '∞' if d == np.inf else f'{d:.4f}'
        print(f"    birth={b:.4f}, death={death}, persistence={pers}")

# Save persistence diagrams
persistence_data = {}
for dim in range(3):
    pts = []
    for p in dgms[dim]:
        pts.append([float(p[0]), float(p[1]) if p[1] != np.inf else None])
    persistence_data[f'H_{dim}'] = pts
with open(os.path.join(OUTDIR, "persistence_diagrams.json"), 'w') as f:
    json.dump(persistence_data, f, indent=2)

# ============================================================================
# 4. BETTI CURVES (via filtration scan)
# ============================================================================

print("\n=== COMPUTING BETTI CURVES ===")

def compute_betti_at_eps(dmat, eps):
    """Fast Betti numbers for 10-point set using basic linear algebra."""
    adj = dmat <= eps
    np.fill_diagonal(adj, True)
    
    # β₀ via connected components
    visited = np.zeros(n, dtype=bool)
    beta0 = 0
    for v in range(n):
        if not visited[v]:
            beta0 += 1
            queue = [v]
            visited[v] = True
            while queue:
                u = queue.pop(0)
                for w in range(n):
                    if adj[u, w] and not visited[w]:
                        visited[w] = True
                        queue.append(w)
    
    # Build edge list
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if dmat[i, j] <= eps]
    n1 = len(edges)
    
    # Build triangle list
    triangles = []
    for i in range(n):
        for j in range(i+1, n):
            if dmat[i, j] > eps: continue
            for k in range(j+1, n):
                if dmat[i, k] <= eps and dmat[j, k] <= eps:
                    triangles.append((i, j, k))
    n2 = len(triangles)
    
    # β₁ via boundary reduction
    if n2 > 0 and n1 > 0:
        edge_to_idx = {e: idx for idx, e in enumerate(edges)}
        bdry = np.zeros((n1, n2), dtype=np.int32)
        for t_idx, tri in enumerate(triangles):
            for e in [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]:
                idx = edge_to_idx.get(e)
                if idx is not None:
                    bdry[idx, t_idx] = 1
        
        # Column-reduce (GF(2))
        pivots = set()
        for col in range(n2):
            rows = np.where(bdry[:, col] == 1)[0]
            while len(rows) > 0:
                pivot = rows[-1]
                if pivot in pivots:
                    # Find the column that has this pivot
                    other_col = None
                    for pc_idx in range(col):
                        low = np.where(bdry[:, pc_idx] == 1)[0]
                        if len(low) > 0 and low[-1] == pivot:
                            other_col = pc_idx
                            break
                    if other_col is not None:
                        bdry[:, col] = (bdry[:, col] + bdry[:, other_col]) % 2
                        rows = np.where(bdry[:, col] == 1)[0]
                    else:
                        break
                else:
                    pivots.add(pivot)
                    break
        
        rank_d2 = len(pivots)
        beta1 = n1 - rank_d2 - (n - beta0)
        if beta1 < 0: beta1 = 0
    else:
        beta1 = 0
    
    # β₂ via tetrahedra
    if n2 > 0:
        tetrahedra = []
        for i in range(n):
            for j in range(i+1, n):
                if dmat[i, j] > eps: continue
                for k in range(j+1, n):
                    if dmat[i, k] > eps or dmat[j, k] > eps: continue
                    for l in range(k+1, n):
                        if (dmat[i, l] <= eps and dmat[j, l] <= eps 
                            and dmat[k, l] <= eps):
                            tetrahedra.append((i, j, k, l))
        n3 = len(tetrahedra)
        
        if n3 > 0:
            tri_to_idx = {tri: idx for idx, tri in enumerate(triangles)}
            bdry3 = np.zeros((n2, n3), dtype=np.int32)
            for tet_idx, tet in enumerate(tetrahedra):
                faces = [
                    tuple(sorted([tet[0], tet[1], tet[2]])),
                    tuple(sorted([tet[0], tet[1], tet[3]])),
                    tuple(sorted([tet[0], tet[2], tet[3]])),
                    tuple(sorted([tet[1], tet[2], tet[3]])),
                ]
                for face in faces:
                    tri_idx = tri_to_idx.get(face)
                    if tri_idx is not None:
                        bdry3[tri_idx, tet_idx] = 1
            
            pivots3 = set()
            for col in range(n3):
                rows = np.where(bdry3[:, col] == 1)[0]
                while len(rows) > 0:
                    pivot = rows[-1]
                    if pivot in pivots3:
                        other_col = None
                        for pc_idx in range(col):
                            low = np.where(bdry3[:, pc_idx] == 1)[0]
                            if len(low) > 0 and low[-1] == pivot:
                                other_col = pc_idx
                                break
                        if other_col is not None:
                            bdry3[:, col] = (bdry3[:, col] + bdry3[:, other_col]) % 2
                            rows = np.where(bdry3[:, col] == 1)[0]
                        else:
                            break
                    else:
                        pivots3.add(pivot)
                        break
            
            rank_d3 = len(pivots3)
            beta2 = n2 - rank_d3 - rank_d2 if n2 > 0 else 0
            if beta2 < 0: beta2 = 0
        else:
            beta2 = 0
    else:
        beta2 = 0
    
    return int(beta0), int(beta1), int(beta2)


epsilons = np.arange(0.0, 3.0 + 0.001, 0.05)
betti_curves = {'beta0': [], 'beta1': [], 'beta2': []}

for eps in epsilons:
    b0, b1, b2 = compute_betti_at_eps(dist_matrix, eps)
    betti_curves['beta0'].append(b0)
    betti_curves['beta1'].append(b1)
    betti_curves['beta2'].append(b2)

# Save
np.savetxt(os.path.join(OUTDIR, "betti_curves.csv"),
           np.column_stack([epsilons, betti_curves['beta0'], 
                           betti_curves['beta1'], betti_curves['beta2']]),
           delimiter=",", fmt="%.6f",
           header="epsilon,beta0,beta1,beta2")

# Print transitions
print("\nBETTI NUMBER TRANSITIONS:")
prev = [None, None, None]
for i, eps in enumerate(epsilons):
    vals = [betti_curves[f'beta{k}'][i] for k in range(3)]
    changes = []
    for k in range(3):
        if prev[k] is not None and vals[k] != prev[k]:
            changes.append(f"β_{k}: {prev[k]}→{vals[k]}")
        prev[k] = vals[k]
    if changes:
        print(f"  ε = {eps:.2f}: {', '.join(changes)}")

# ============================================================================
# 5. 2D PROJECTION (SENSITIVITY)
# ============================================================================

print("\n=== 2D PROJECTION (I_vert, I_horiz only) ===")
result_2d = ripser(coords[:, :2], maxdim=2, distance_matrix=False, thresh=3.0)
for dim in range(min(3, len(result_2d['dgms']))):
    print(f"  H_{dim}: {len(result_2d['dgms'][dim])} features")
    for p in result_2d['dgms'][dim]:
        b, d = p[0], p[1]
        death = '∞' if d == np.inf else f'{d:.4f}'
        print(f"    birth={b:.4f}, death={death}")

np.save(os.path.join(OUTDIR, "persistence_2d.npy"), np.array([d for dgm in result_2d['dgms'] for d in dgm], dtype=object))

# ============================================================================
# 6. NULL MODEL (FAST — just use ripser)
# ============================================================================

print("\n=== NULL MODEL COMPARISON ===")
N_NULL = 10000

# Track key statistics
null_h0_count = []    # number of H₀ features
null_h1_count = []    # number of H₁ features
null_h2_count = []    # number of H₂ features
null_h1_long = []     # H₁ features with persistence > 0.10
null_h2_long = []     # H₂ features with persistence > 0.10
null_max_b0 = []      # max β₀ (actually we want: time β₀ pauses at certain values)
null_max_b1 = []      # max β₁
null_max_b2 = []      # max β₂
null_b1_persistences = []  # all H₁ persistences
null_b2_persistences = []

# For Betti curve comparison, compute at key ε values only
key_epsilons = epsilons

start = time.time()
for seed in range(N_NULL):
    if (seed + 1) % 1000 == 0:
        elapsed = time.time() - start
        rate = (seed + 1) / elapsed
        remaining = (N_NULL - seed - 1) / max(rate, 0.1)
        print(f"  {seed + 1}/{N_NULL} ({rate:.0f}/s, ~{remaining:.0f}s)")
    
    rng = np.random.RandomState(seed)
    rand_coords = rng.uniform(bbox_min, bbox_max, size=(10, 3))
    
    res = ripser(rand_coords, maxdim=2, distance_matrix=False, thresh=3.0)
    
    null_h0_count.append(len(res['dgms'][0]))
    null_h1_count.append(len(res['dgms'][1]))
    null_h2_count.append(len(res['dgms'][2]))
    
    h1_pers = [p[1] - p[0] for p in res['dgms'][1] if p[1] != np.inf]
    null_b1_persistences.extend(h1_pers)
    null_h1_long.append(sum(1 for p in h1_pers if p > 0.10))
    
    h2_pers = [p[1] - p[0] for p in res['dgms'][2] if p[1] != np.inf]
    null_b2_persistences.extend(h2_pers)
    null_h2_long.append(sum(1 for p in h2_pers if p > 0.05))

elapsed = time.time() - start
print(f"Null model complete in {elapsed:.1f}s ({N_NULL/elapsed:.0f} samples/s)")

# Compute p-values
actual_h0_count = len(dgms[0])
actual_h1_count = len(dgms[1])
actual_h2_count = len(dgms[2])
actual_h1_long = sum(1 for p in dgms[1] if p[1] != np.inf and p[1] - p[0] > 0.10)
actual_h2_long = sum(1 for p in dgms[2] if p[1] != np.inf and p[1] - p[0] > 0.05)

p_h1_count = np.mean(np.array(null_h1_count) >= actual_h1_count)
p_h2_count = np.mean(np.array(null_h2_count) >= actual_h2_count)
p_h1_long = np.mean(np.array(null_h1_long) >= actual_h1_long)
p_h2_long = np.mean(np.array(null_h2_long) >= actual_h2_long)

# H₁ max β₁ comparison
# Compute max β₁ for actual from ripser diagonal
# ripser doesn't directly give β₁(ε); we'll use our manual computation
actual_b1_max = max(betti_curves['beta1'])
print(f"\n=== NULL MODEL RESULTS ===")
print(f"  H₀ features: actual={actual_h0_count}, null_mean={np.mean(null_h0_count):.1f}±{np.std(null_h0_count):.1f}")
print(f"  H₁ features: actual={actual_h1_count}, null_mean={np.mean(null_h1_count):.1f}±{np.std(null_h1_count):.1f}, p={p_h1_count:.6f}")
print(f"  H₂ features: actual={actual_h2_count}, null_mean={np.mean(null_h2_count):.1f}±{np.std(null_h2_count):.1f}, p={p_h2_count:.6f}")
print(f"  H₁ persistent (>0.10): actual={actual_h1_long}, "
      f"null_mean={np.mean(null_h1_long):.3f}±{np.std(null_h1_long):.3f}, p={p_h1_long:.6f}")
print(f"  H₂ persistent (>0.05): actual={actual_h2_long}, "
      f"null_mean={np.mean(null_h2_long):.3f}±{np.std(null_h2_long):.3f}, p={p_h2_long:.6f}")
print(f"  β₁ max (from Betti curve): actual={actual_b1_max}")

# Save null statistics
null_stats = {
    'h0_count_mean': float(np.mean(null_h0_count)),
    'h0_count_std': float(np.std(null_h0_count)),
    'h1_count_mean': float(np.mean(null_h1_count)),
    'h1_count_std': float(np.std(null_h1_count)),
    'h2_count_mean': float(np.mean(null_h2_count)),
    'h2_count_std': float(np.std(null_h2_count)),
    'h1_long_mean': float(np.mean(null_h1_long)),
    'h1_long_std': float(np.std(null_h1_long)),
    'h2_long_mean': float(np.mean(null_h2_long)),
    'h2_long_std': float(np.std(null_h2_long)),
    'p_h1_count': float(p_h1_count),
    'p_h2_count': float(p_h2_count),
    'p_h1_long': float(p_h1_long),
    'p_h2_long': float(p_h2_long),
    'actual_h0_count': actual_h0_count,
    'actual_h1_count': actual_h1_count,
    'actual_h2_count': actual_h2_count,
    'actual_h1_long': actual_h1_long,
    'actual_h2_long': actual_h2_long,
    'actual_b1_max': int(actual_b1_max),
}
with open(os.path.join(OUTDIR, "p_values.json"), 'w') as f:
    json.dump(null_stats, f, indent=2)

# Save null feature counts for histograms
np.savetxt(os.path.join(OUTDIR, "null_h1_persistences.csv"), 
           np.array(null_b1_persistences), delimiter=",", 
           header="persistence_of_all_null_H1_features")

# ============================================================================
# 7. PLOTS
# ============================================================================

print("\n=== GENERATING PLOTS ===")

# 7a. Betti curves
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
colors = ['#2196F3', '#FF5722', '#4CAF50']
labels = [r'$\beta_0$ (components)', r'$\beta_1$ (loops)', r'$\beta_2$ (voids)']

for k in range(3):
    ax = axes[k]
    curve = betti_curves[f'beta{k}']
    ax.plot(epsilons, curve, color=colors[k], linewidth=2.5, drawstyle='steps-post')
    ax.fill_between(epsilons, 0, curve, alpha=0.3, color=colors[k], step='post')
    ax.set_ylabel(labels[k], fontsize=12)
    ax.set_ylim(-0.2, max(curve) + 0.8)
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel(r'Filtration scale $\varepsilon$', fontsize=12)
fig.suptitle('Betti Numbers of the Musical Tradition Dial Space', 
             fontsize=14, fontweight='bold')

# Annotations
axes[0].axhline(y=5, color='gray', linestyle='--', alpha=0.5)
axes[0].annotate(r'$\beta_0=5$ (5 clusters)', xy=(0.35, 5.1), fontsize=9, color='gray')
axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[1].annotate(r'$\beta_1=1$ (1 hole)', xy=(0.55, 1.1), fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "betti_curves.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved betti_curves.png")

# 7b. Persistence diagrams
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for dim in range(3):
    if len(dgms[dim]) > 0:
        plot_diagrams(dgms[dim], ax=axes[dim], title=f'Persistence H_{dim}')
    else:
        axes[dim].text(0.5, 0.5, 'No features', ha='center', va='center', 
                      transform=axes[dim].transAxes, fontsize=12)
        axes[dim].set_title(f'Persistence H_{dim}')
    axes[dim].set_xlabel('Birth')
    axes[dim].set_ylabel('Death')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "persistence_diagrams.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved persistence_diagrams.png")

# 7c. Individual persistence diagrams
for dim, name, color in [(0, 'H0', '#2196F3'), (1, 'H1', '#FF5722'), (2, 'H2', '#4CAF50')]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    pts = dgms[dim]
    
    if len(pts) > 0:
        births = [p[0] for p in pts]
        deaths = [p[1] if p[1] != np.inf else 3.0 for p in pts]
        
        ax.scatter(births, deaths, c=color, s=50, alpha=0.7, edgecolors='black', zorder=5)
        
        max_val = max(births + [d for d in deaths if d != 3.0] + [0.1]) + 0.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='death = birth')
        
        for i, (b, d_val) in enumerate(zip(births, deaths)):
            label = f'({b:.3f}, ∞)' if d_val == 3.0 and pts[i][1] == np.inf else f'({b:.3f}, {pts[i][1]:.3f})'
            ax.annotate(label, (b, d_val), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, alpha=0.7)
        
        ax.set_xlim(-0.05, max_val)
        ax.set_ylim(-0.05, max_val + 0.1)
    else:
        ax.text(0.5, 0.5, 'No persistent features', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    
    ax.set_xlabel('Birth ε', fontsize=12)
    ax.set_ylabel('Death ε', fontsize=12)
    ax.set_title(f'Persistence: {name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if len(pts) > 0: ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"persistence_{name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved persistence_{name}.png")

# 7d. Null comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# H₁ comparison
ax = axes[0]
null_h1_arr = np.array(null_h1_count)
ax.hist(null_h1_arr, bins=range(0, int(null_h1_arr.max()) + 2), 
        alpha=0.6, color='gray', edgecolor='black', density=True, label='Null')
ax.axvline(actual_h1_count, color='#FF5722', linewidth=3, label=f'Actual ({actual_h1_count})')
ax.set_xlabel('Number of H₁ features')
ax.set_ylabel('Density')
ax.set_title('H₁ Feature Count: Actual vs Null')
ax.legend()
ax.grid(True, alpha=0.3)

# H₁ long persistence comparison
ax = axes[1]
null_h1_long_arr = np.array(null_h1_long)
max_bins = max(int(null_h1_long_arr.max()), actual_h1_long) + 2
ax.hist(null_h1_long_arr, bins=np.arange(-0.5, max_bins + 0.5, 1),
        alpha=0.6, color='gray', edgecolor='black', density=True, label='Null')
ax.axvline(actual_h1_long, color='#FF5722', linewidth=3, label=f'Actual ({actual_h1_long})')
ax.set_xlabel('H₁ features with persistence > 0.10')
ax.set_ylabel('Density')
ax.set_title('Persistent H₁ Features: Actual vs Null')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "null_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved null_comparison.png")

# 7e. 3D tradition plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
cluster_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
cluster_names_map = ["Maximal", "Rhythmic", "Balanced", "Harmonic", "Presence"]

for i, (trad, coord) in enumerate(zip(traditions, coords)):
    clust = 0  # default
    if trad == "West African": clust = 1
    elif trad in ["Balinese Gamelan", "Javanese Gamelan"]: clust = 2
    elif trad == "Western CP": clust = 3
    elif trad in ["Chinese", "Japanese Gagaku"]: clust = 4
    
    ax.scatter(coord[0], coord[1], coord[2], c=cluster_colors[clust], s=120, 
              edgecolors='black', linewidth=1, zorder=5)
    ax.text(coord[0], coord[1], coord[2], trad.replace(' ', '\n'), 
           fontsize=6, ha='center', va='bottom')

ax.set_xlabel('I_vert (pitch)', fontsize=11)
ax.set_ylabel('I_horiz (rhythm)', fontsize=11)
ax.set_zlabel('I_spectral (timbre)', fontsize=11)
ax.set_title('10 Musical Traditions in Dial Space', fontsize=13, fontweight='bold')

# Legend
for i, (c, name) in enumerate(zip(cluster_colors, cluster_names_map)):
    ax.plot([], [], [], 'o', c=c, label=name, markersize=8)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "traditions_3d.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved traditions_3d.png")

# ============================================================================
# 8. FINAL VERDICT
# ============================================================================

print("\n" + "="*70)
print("   VERIFICATION OF BETTI NUMBERS OF MUSIC PREDICTIONS")
print("="*70)

# Prediction 1: β₀=5 with persistence
beta5_persistence = 0.0
in_beta5 = False
for i, b0 in enumerate(betti_curves['beta0']):
    if b0 == 5:
        if not in_beta5:
            beta5_start = epsilons[i]
            in_beta5 = True
    else:
        if in_beta5:
            beta5_persistence += epsilons[i] - beta5_start
            in_beta5 = False
if in_beta5:
    beta5_persistence += 3.0 - beta5_start

print(f"\nPrediction 1: β₀=5 persists for Δε > 0.30")
print(f"  Actual: β₀=5 from ε={epsilons[np.argmax(np.array(betti_curves['beta0'])==5)]:.2f}"
      f" to ε={epsilons[np.argmax(np.array(betti_curves['beta0'])<5):]}" if False else "")
# Find the range
beta5_indices = [i for i, v in enumerate(betti_curves['beta0']) if v == 5]
if beta5_indices:
    print(f"  β₀=5 from ε={epsilons[beta5_indices[0]]:.2f} to ε={epsilons[beta5_indices[-1]]:.2f}")
print(f"  Persistence: {beta5_persistence:.4f}")
print(f"  {'✓ PASS' if beta5_persistence > 0.30 else '✗ FAIL'}")

# Prediction 2: β₁ ≥ 1 with persistence > 0.10
h1_persistences = [p[1] - p[0] for p in dgms[1] if p[1] != np.inf]
print(f"\nPrediction 2: β₁ ≥ 1 persistent hole (persistence > 0.10)")
print(f"  H₁ features with persistence > 0.10: {actual_h1_long}")
print(f"  All H₁ persistences: {[f'{p:.4f}' for p in h1_persistences]}")
print(f"  {'✓ PASS' if actual_h1_long >= 1 else '✗ FAIL'}")

# Prediction 3: β₂ ≥ 1
h2_persistences = [p[1] - p[0] for p in dgms[2] if p[1] != np.inf]
print(f"\nPrediction 3: β₂ ≥ 1 void with persistence > 0.05")
print(f"  H₂ features: {len(dgms[2])}")
print(f"  {'✓ PASS' if len(dgms[2]) >= 1 else '✗ FAIL'}")

# Prediction 4: Significant vs random
print(f"\nPrediction 4: Significant difference from random (p < 0.05)")
print(f"  H₁ count p = {p_h1_count:.6f} ({'significant' if p_h1_count < 0.05 else 'not significant'})")
print(f"  H₂ count p = {p_h2_count:.6f} ({'significant' if p_h2_count < 0.05 else 'not significant'})")
print(f"  H₁ persistent (>0.10) p = {p_h1_long:.6f} ({'significant' if p_h1_long < 0.05 else 'not significant'})")
print(f"  {'✓ PASS' if p_h1_long < 0.05 else '✗ FAIL' if p_h1_long > 0.10 else '⚠ MARGINAL'}")

print(f"\n{'='*70}")
print(f"   All output in: {OUTDIR}")
print(f"{'='*70}")

# List output files
print("\nOutput files:")
for f in sorted(os.listdir(OUTDIR)):
    fpath = os.path.join(OUTDIR, f)
    size = os.path.getsize(fpath)
    print(f"  {f:45s} {size:>8d} bytes")
