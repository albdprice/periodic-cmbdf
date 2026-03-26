"""
Generate learning curve figures from existing results.
For Anatole: MP formation energy + band gap + GPU crossover.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = '/root/cMBDF_cc/figures'
import os
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Fig 1: MP Formation Energy Learning Curve
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# Data from our benchmarks
N_train = [200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 20000, 30000]
MAE_eform = [0.505, 0.493, 0.448, 0.408, 0.366, 0.350, 0.335, 0.330, 0.293, 0.271]

ax.loglog(N_train, MAE_eform, 'o-', color='#2171B5', linewidth=2, markersize=8,
          label='p-cMBDF (40 dim, elem-specific)')

# REMatch point
ax.plot(500, 0.263, 's', color='#E6550D', markersize=10, zorder=5,
        label='p-cMBDF + REMatch kernel (40 dim)')

# Reference lines
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Composition-only baseline (~0.3-0.5)')
ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('Training set size N', fontsize=13)
ax.set_ylabel('MAE (eV/atom)', fontsize=13)
ax.set_title('Materials Project Formation Energy\n(matbench_mp_e_form, ≤30 atoms, 84 elements)', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(100, 50000)
ax.set_ylim(0.2, 0.6)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig1_mp_eform.png'), dpi=150)
plt.savefig(os.path.join(OUTDIR, 'fig1_mp_eform.pdf'))
print("Fig 1 saved", flush=True)

# ============================================================
# Fig 2: Band Gap Learning Curve
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

N_gap = [200, 500, 1000, 2000, 4000]
MAE_gap = [0.771, 0.775, 0.750, 0.714, 0.696]

ax.semilogx(N_gap, MAE_gap, 'o-', color='#6A51A3', linewidth=2, markersize=8,
            label='p-cMBDF (40 dim, elem-specific)')

ax.set_xlabel('Training set size N', fontsize=13)
ax.set_ylabel('MAE (eV)', fontsize=13)
ax.set_title('Materials Project Band Gap\n(matbench_mp_gap, ≤30 atoms)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig2_mp_gap.png'), dpi=150)
plt.savefig(os.path.join(OUTDIR, 'fig2_mp_gap.pdf'))
print("Fig 2 saved", flush=True)

# ============================================================
# Fig 3: GPU Crossover
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Per-structure
atoms_ps = [6, 11, 19, 32, 51, 77, 130]
numba_ps = [13.2, 0.5, 1.1, 1.5, 2.9, 4.2, 8.6]
gpu_ps = [2.2, 1.7, 1.8, 2.2, 4.7, 3.4, 4.9]
speedup_ps = [n/g for n, g in zip(numba_ps, gpu_ps)]

ax1.plot(atoms_ps, speedup_ps, 'o-', color='#2171B5', linewidth=2, markersize=8)
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Parity (GPU = Numba)')
ax1.fill_between(atoms_ps, speedup_ps, 1.0,
                  where=[s > 1 for s in speedup_ps],
                  alpha=0.15, color='green', label='GPU faster')
ax1.fill_between(atoms_ps, speedup_ps, 1.0,
                  where=[s <= 1 for s in speedup_ps],
                  alpha=0.15, color='red', label='Numba faster')
ax1.set_xlabel('Avg atoms per structure', fontsize=12)
ax1.set_ylabel('GPU speedup vs Numba', fontsize=12)
ax1.set_title('Per-structure processing', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 2.5)
ax1.annotate('Crossover\n~60-70 atoms', xy=(70, 1.0), fontsize=10,
             ha='center', va='bottom', color='red')

# Batched
atoms_b = [8, 12, 16]
numba_b = [10.9, 2.5, 2.6]
gpu_b = [5.9, 3.5, 2.8]
speedup_b = [n/g for n, g in zip(numba_b, gpu_b)]

ax2.bar(range(len(atoms_b)), speedup_b, color=['#2CA02C', '#D62728', '#FF7F0E'],
        alpha=0.8, width=0.6)
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
ax2.set_xticks(range(len(atoms_b)))
ax2.set_xticklabels(['≤15 (avg 8)\nN=500', '≤25 (avg 12)\nN=300', '≤40 (avg 16)\nN=200'])
ax2.set_ylabel('Batched GPU speedup vs Numba', fontsize=12)
ax2.set_title('Batched processing', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
for i, s in enumerate(speedup_b):
    ax2.text(i, s + 0.05, '%.2fx' % s, ha='center', fontsize=11, fontweight='bold')

plt.suptitle('GPU Performance: periodic cMBDF on RTX 4090', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig3_gpu_crossover.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTDIR, 'fig3_gpu_crossover.pdf'), bbox_inches='tight')
print("Fig 3 saved", flush=True)

# ============================================================
# Fig 4: Feature Pruning Comparison (molecules vs solids)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Molecular (QM9) — leave-one-out deltas
mol_labels = ['R:m=4,e⁻⁵ʳ', 'A:m=0,cos θ', 'A:m=1,cos 4θ',
              'R:m=1,e⁻⁵ʳ', 'R:m=3,e⁻¹·⁵ʳ']
mol_deltas = [170.6, 121.8, 112.3, 92.9, 78.8]
mol_labels_neg = ['A:m=0,cos 4θ', 'A:m=0,cos 3θ', 'A:m=1,cos 2θ',
                  'A:m=3,cos 2θ', 'A:m=0,cos 2θ']
mol_deltas_neg = [-30.4, -25.5, -23.5, -23.5, -22.9]

colors_pos = ['#2171B5'] * 5
colors_neg = ['#CB181D'] * 5

all_labels = mol_labels + mol_labels_neg
all_deltas = mol_deltas + mol_deltas_neg
all_colors = colors_pos + colors_neg

y_pos = range(len(all_labels))
ax1.barh(y_pos, all_deltas, color=all_colors, alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(all_labels, fontsize=9)
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.set_xlabel('ΔMAE (kcal/mol)', fontsize=11)
ax1.set_title('Molecules (QM9)\n17/40 removable', fontsize=12)
ax1.invert_yaxis()

# Solid-state (MP) — all positive
sol_labels = ['R:m=4,e⁻¹·⁵ʳ', 'A:m=2,cos 2θ', 'A:m=1,cos 3θ',
              'A:m=4,cos θ', 'R:m=0,1/(r+1)³']
sol_deltas = [0.0149, 0.0145, 0.0142, 0.0140, 0.0139]
sol_labels2 = ['A:m=0,cos 2θ', 'A:m=1,cos θ', 'A:m=0,cos 4θ',
               'R:m=2,e⁻⁵ʳ', 'A:m=1,cos 2θ']
sol_deltas2 = [0.0123, 0.0123, 0.0124, 0.0125, 0.0125]

sl = sol_labels + sol_labels2
sd = sol_deltas + sol_deltas2
ax2.barh(range(len(sl)), sd, color='#2171B5', alpha=0.8)
ax2.set_yticks(range(len(sl)))
ax2.set_yticklabels(sl, fontsize=9)
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.set_xlabel('ΔMAE (eV/atom)', fontsize=11)
ax2.set_title('Solids (MP)\n0/40 removable — ALL needed', fontsize=12)
ax2.invert_yaxis()

plt.suptitle('Feature Importance: Molecules vs Solids', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig4_feature_pruning.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTDIR, 'fig4_feature_pruning.pdf'), bbox_inches='tight')
print("Fig 4 saved", flush=True)

print("\nAll figures saved to %s" % OUTDIR, flush=True)
