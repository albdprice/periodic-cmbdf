"""
p-cMBDF + GNN: Use p-cMBDF (40-dim) as node features in a simple
message-passing GNN. Train on matbench tasks.

Architecture:
- Node features: p-cMBDF (40 dim per atom)
- Edge features: distance + cutoff
- 3 message-passing layers with 128-dim hidden
- Global sum pooling -> MLP head -> scalar prediction
- Train with Adam, MSE loss, 200 epochs

This combines physical feature quality with GNN learning power.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os, sys, time, gc
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

DATA_DIR = '/home/albd/projects/cmbdf/data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device, flush=True)

# ============================================================
# Simple Message-Passing GNN
# ============================================================
class SimpleGNN(nn.Module):
    """
    Message-passing GNN with p-cMBDF node features.
    No explicit edge features beyond distance-weighted messages.
    """
    def __init__(self, in_dim=40, hidden_dim=128, n_layers=3, cutoff=6.0):
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Linear(in_dim, hidden_dim)

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.message_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            self.update_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features, edge_index, edge_dist, n_atoms_list):
        """
        node_features: (total_atoms, in_dim) — p-cMBDF features
        edge_index: (2, n_edges) — [source, target]
        edge_dist: (n_edges,) — pairwise distances
        n_atoms_list: list of atom counts per structure
        """
        h = self.embed(node_features)

        # Distance-based envelope
        envelope = 0.5 * (torch.cos(torch.pi * edge_dist / self.cutoff) + 1.0)

        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers):
            # Message: transform neighbor features, weight by distance
            src, tgt = edge_index[0], edge_index[1]
            messages = msg_layer(h[src]) * envelope.unsqueeze(-1)

            # Aggregate messages per target atom
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, tgt.unsqueeze(1).expand(-1, h.shape[1]), messages)

            # Update: combine current features with aggregated messages
            h = h + upd_layer(torch.cat([h, agg], dim=-1))

        # Per-atom prediction, then sum per structure (energy is extensive)
        atom_out = self.output(h).squeeze(-1)

        # Sum pooling per structure
        predictions = []
        offset = 0
        for n in n_atoms_list:
            predictions.append(atom_out[offset:offset+n].sum())
            offset += n

        return torch.stack(predictions)


def prepare_batch(indices, all_reps, all_charges, all_targets, all_edges, device):
    """Prepare a batched graph from structure indices."""
    node_feats = []
    edge_indices = []
    edge_dists = []
    targets = []
    n_atoms_list = []
    atom_offset = 0

    for i in indices:
        n_at = len(all_charges[i])
        feat = all_reps[i, :n_at, :]
        node_feats.append(torch.tensor(feat, dtype=torch.float32))
        n_atoms_list.append(n_at)
        targets.append(all_targets[i])

        # Edges from pre-built edge list
        if all_edges[i] is not None:
            src, tgt, dist = all_edges[i]
            edge_indices.append(torch.stack([
                torch.tensor(src, dtype=torch.long) + atom_offset,
                torch.tensor(tgt, dtype=torch.long) + atom_offset
            ]))
            edge_dists.append(torch.tensor(dist, dtype=torch.float32))

        atom_offset += n_at

    node_feats = torch.cat(node_feats).to(device)
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1).to(device)
        edge_dist = torch.cat(edge_dists).to(device)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_dist = torch.zeros(0, dtype=torch.float32, device=device)

    targets = torch.tensor(targets, dtype=torch.float32, device=device)
    return node_feats, edge_index, edge_dist, n_atoms_list, targets


def build_edges(charges, coords, cells, cutoff=6.0):
    """Build edge lists for all structures using ASE."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list as ase_nl

    all_edges = []
    for i in range(len(charges)):
        try:
            a = Atoms(numbers=charges[i].astype(int), positions=coords[i],
                     cell=cells[i], pbc=True)
            ii, jj, dd = ase_nl('ijd', a, cutoff)
            all_edges.append((ii, jj, dd))
        except:
            all_edges.append(None)
    return all_edges


def train_and_eval(train_idx, test_idx, all_reps, all_charges, all_targets,
                   all_edges, n_epochs=200, batch_size=64, lr=1e-3):
    """Train GNN and return test MAE."""
    model = SimpleGNN(in_dim=40, hidden_dim=128, n_layers=3, cutoff=6.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(train_idx)
        epoch_loss = 0
        n_batches = 0

        for b_start in range(0, len(train_idx), batch_size):
            b_idx = train_idx[b_start:b_start + batch_size]
            node_feats, edge_index, edge_dist, n_atoms, targets = \
                prepare_batch(b_idx, all_reps, all_charges, all_targets, all_edges, device)

            optimizer.zero_grad()
            pred = model(node_feats, edge_index, edge_dist, n_atoms)
            loss = nn.MSELoss()(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step(epoch_loss / n_batches)

        if (epoch + 1) % 50 == 0:
            print("    Epoch %d: loss=%.4f" % (epoch + 1, epoch_loss / n_batches), flush=True)

    # Evaluate
    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for b_start in range(0, len(test_idx), batch_size):
            b_idx = test_idx[b_start:b_start + batch_size]
            node_feats, edge_index, edge_dist, n_atoms, targets = \
                prepare_batch(b_idx, all_reps, all_charges, all_targets, all_edges, device)
            pred = model(node_feats, edge_index, edge_dist, n_atoms)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(targets.cpu().numpy())

    return mean_absolute_error(all_true, all_pred)


# ============================================================
# Run on matbench tasks
# ============================================================
from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element

tasks = [
    ('matbench_perovskites', 'eV/atom'),  # Start with smaller tasks
    ('matbench_phonons', '1/cm'),
    ('matbench_dielectric', 'unitless'),
    ('matbench_mp_e_form', 'eV/atom'),  # Large task last
]

print("=" * 70, flush=True)
print("p-cMBDF + GNN (3-layer message passing)", flush=True)
print("=" * 70, flush=True)

for task_name, unit in tasks:
    print("\n" + "=" * 70, flush=True)
    print("TASK: %s [%s]" % (task_name, unit), flush=True)
    print("=" * 70, flush=True)

    # Load data
    cache_data = os.path.join(DATA_DIR, 'matbench_%s.npz' % task_name.replace('matbench_', ''))
    if not os.path.exists(cache_data):
        from matminer.datasets import load_dataset
        print("  Fetching...", flush=True)
        df = load_dataset(task_name)
        target_col = [c for c in df.columns if c != 'structure'][0]
        charges, coords, cells, targets, natoms = [], [], [], [], []
        for _, row in df.iterrows():
            try:
                s = row['structure']
                charges.append(np.array([sp.Z for sp in s.species], dtype=np.float64))
                coords.append(np.array(s.cart_coords, dtype=np.float64))
                cells.append(np.array(s.lattice.matrix, dtype=np.float64))
                targets.append(row[target_col])
                natoms.append(len(s))
            except:
                pass
        charges = np.array(charges, dtype=object)
        coords = np.array(coords, dtype=object)
        cells = np.array(cells, dtype=object)
        targets = np.array(targets, dtype=np.float64)
        natoms = np.array(natoms, dtype=np.int64)
        np.savez(cache_data, charges=charges, coords=coords, cells=cells,
                 targets=targets, n_atoms=natoms)
    else:
        d = np.load(cache_data, allow_pickle=True)
        charges = d['charges']
        coords = d['coords']
        cells = d['cells']
        targets = d['targets']
        natoms = d['n_atoms']

    # Filter
    mask = natoms <= 30
    valid = np.where(mask)[0]
    N = len(valid)

    # For mp_e_form, subsample to 20k for GNN tractability
    if N > 20000:
        np.random.seed(42)
        valid = np.random.choice(valid, 20000, replace=False)
        valid.sort()
        N = 20000

    charges = charges[valid]
    coords = coords[valid]
    cells = cells[valid]
    targets = targets[valid]
    print("  Using %d structures" % N, flush=True)

    # Generate reps
    cache_reps = os.path.join(DATA_DIR, 'pcmbdf_gnn_%s.npz' % task_name.replace('matbench_', ''))
    if os.path.exists(cache_reps):
        reps = np.load(cache_reps)['reps']
    else:
        print("  Generating p-cMBDF...", flush=True)
        reps = generate_mbdf_periodic(
            list(charges), list(coords), list(cells),
            pbc=(True,True,True), rcut=6.0, n_atm=2.0,
            n_jobs=-1, progress_bar=True, elem_specific=True)
        np.savez_compressed(cache_reps, reps=reps)

    reps_norm, _ = normalize_per_element(reps, charges, mode='mean')

    # Build edges
    print("  Building edge lists...", flush=True)
    all_edges = build_edges(charges, coords, cells, cutoff=6.0)

    # 5-fold CV (or 3-fold for large datasets)
    n_folds = 3 if N > 10000 else 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=18012019)

    fold_maes = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(N))):
        print("  Fold %d/%d (train=%d, test=%d):" % (fold_idx+1, n_folds, len(train_idx), len(test_idx)), flush=True)

        train_idx_list = train_idx.tolist()
        test_idx_list = test_idx.tolist()

        mae = train_and_eval(train_idx_list, test_idx_list,
                             reps_norm, charges, targets, all_edges,
                             n_epochs=200, batch_size=64, lr=1e-3)
        fold_maes.append(mae)
        print("    Fold %d MAE: %.4f %s" % (fold_idx+1, mae, unit), flush=True)

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    print("\n  === %s: %.4f ± %.4f %s ===" % (task_name, mean_mae, std_mae, unit), flush=True)

print("\nAll tasks complete!", flush=True)
