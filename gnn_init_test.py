"""
Proof of concept: p-cMBDF as initial node embeddings for a GNN
that refines them through message passing.

Compare:
1. Learned embeddings (random init, like CGCNN) + GNN
2. p-cMBDF embeddings (frozen) + GNN  [what we already tested]
3. p-cMBDF embeddings (trainable, refined by GNN) [the new idea]

On matbench_perovskites (18928 structures, 5 atoms each).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, time, gc
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_nl

DATA_DIR = '/home/albd/projects/cmbdf/data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device, flush=True)

# ============================================================
# GNN with configurable initial embeddings
# ============================================================
class FlexGNN(nn.Module):
    def __init__(self, init_mode='learned', n_species=100, pcmbdf_dim=40,
                 hidden=128, n_layers=3, cutoff=6.0):
        super().__init__()
        self.init_mode = init_mode
        self.cutoff = cutoff

        if init_mode == 'learned':
            # Random learnable embedding per species (like CGCNN)
            self.species_embed = nn.Embedding(n_species, hidden)
            self.input_proj = None
        elif init_mode == 'frozen':
            # p-cMBDF features, projected but not backpropagated through
            self.input_proj = nn.Linear(pcmbdf_dim, hidden)
        elif init_mode == 'refine':
            # p-cMBDF features as trainable initial state, refined by GNN
            self.input_proj = nn.Linear(pcmbdf_dim, hidden)

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.message_layers.append(nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)))
            self.update_layers.append(nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)))

        # Distance-based filter (like SchNet continuous filter)
        self.dist_filter = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

        self.output = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, species, pcmbdf_feats, edge_index, edge_dist, n_atoms_list):
        # Initial embeddings
        if self.init_mode == 'learned':
            h = self.species_embed(species)
        elif self.init_mode == 'frozen':
            with torch.no_grad():
                h = self.input_proj(pcmbdf_feats)
        elif self.init_mode == 'refine':
            h = self.input_proj(pcmbdf_feats)

        # Distance envelope + filter
        envelope = 0.5 * (torch.cos(torch.pi * edge_dist / self.cutoff) + 1.0)
        dist_feat = self.dist_filter(edge_dist.unsqueeze(-1)) * envelope.unsqueeze(-1)

        # Message passing
        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers):
            src, tgt = edge_index[0], edge_index[1]
            messages = msg_layer(h[src]) * dist_feat
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, tgt.unsqueeze(1).expand(-1, h.shape[1]), messages)
            h = h + upd_layer(torch.cat([h, agg], dim=-1))

        # Per-atom prediction, sum per structure
        atom_out = self.output(h).squeeze(-1)
        predictions = []
        offset = 0
        for n in n_atoms_list:
            predictions.append(atom_out[offset:offset+n].sum())
            offset += n
        return torch.stack(predictions)


# ============================================================
# Data loading
# ============================================================
print("Loading perovskites...", flush=True)
cache = os.path.join(DATA_DIR, 'matbench_perovskites.npz')
if not os.path.exists(cache):
    cache = os.path.join(DATA_DIR, 'matbench_matbench_perovskites.npz')
d = np.load(cache, allow_pickle=True)
charges = d['charges']
coords = d['coords']
cells = d['cells']
targets = d['targets']
natoms = d['n_atoms']

mask = natoms <= 30
valid = np.where(mask)[0]
N = len(valid)
charges = charges[valid]
coords = coords[valid]
cells = cells[valid]
targets = targets[valid]
print("Structures: %d" % N, flush=True)

# Load p-cMBDF reps
from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element
rep_cache = os.path.join(DATA_DIR, 'pcmbdf_gnn_init_perov.npz')
if os.path.exists(rep_cache):
    reps = np.load(rep_cache)['reps']
else:
    print("Generating p-cMBDF...", flush=True)
    reps = generate_mbdf_periodic(
        list(charges), list(coords), list(cells),
        pbc=(True,True,True), rcut=6.0, n_atm=2.0,
        n_jobs=-1, progress_bar=True, elem_specific=True)
    np.savez_compressed(rep_cache, reps=reps)
reps_norm, _ = normalize_per_element(reps, charges, mode='mean')
print("Reps: %s" % str(reps.shape), flush=True)

# Build edges
print("Building edges...", flush=True)
all_edges = []
for i in range(N):
    try:
        a = Atoms(numbers=charges[i].astype(int), positions=coords[i],
                 cell=cells[i], pbc=True)
        ii, jj, dd = ase_nl('ijd', a, 6.0)
        all_edges.append((ii, jj, dd))
    except:
        all_edges.append(None)

# Species mapping
all_species = sorted(set(int(z) for q in charges for z in q))
sp_to_idx = {z: i for i, z in enumerate(all_species)}
n_species = len(all_species)
print("Species: %d" % n_species, flush=True)


def prepare_batch(indices):
    node_feats, species_list, edge_indices, edge_dists = [], [], [], []
    tgts, n_atoms_list = [], []
    atom_offset = 0
    for i in indices:
        n_at = len(charges[i])
        node_feats.append(torch.tensor(reps_norm[i, :n_at, :], dtype=torch.float32))
        sp = torch.tensor([sp_to_idx[int(z)] for z in charges[i]], dtype=torch.long)
        species_list.append(sp)
        n_atoms_list.append(n_at)
        tgts.append(targets[i])
        if all_edges[i] is not None:
            src, tgt, dist = all_edges[i]
            edge_indices.append(torch.stack([
                torch.tensor(src, dtype=torch.long) + atom_offset,
                torch.tensor(tgt, dtype=torch.long) + atom_offset]))
            edge_dists.append(torch.tensor(dist, dtype=torch.float32))
        atom_offset += n_at

    node_feats = torch.cat(node_feats).to(device)
    species = torch.cat(species_list).to(device)
    edge_index = torch.cat(edge_indices, dim=1).to(device) if edge_indices else torch.zeros(2,0,dtype=torch.long,device=device)
    edge_dist = torch.cat(edge_dists).to(device) if edge_dists else torch.zeros(0,dtype=torch.float32,device=device)
    tgts = torch.tensor(tgts, dtype=torch.float32, device=device)
    return species, node_feats, edge_index, edge_dist, n_atoms_list, tgts


def train_eval(init_mode, train_idx, test_idx, n_epochs=200, batch_size=64, lr=1e-3):
    model = FlexGNN(init_mode=init_mode, n_species=n_species+1, pcmbdf_dim=40,
                    hidden=128, n_layers=3, cutoff=6.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-5)

    train_list = train_idx.tolist()
    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(train_list)
        epoch_loss = 0
        n_b = 0
        for b in range(0, len(train_list), batch_size):
            bi = train_list[b:b+batch_size]
            sp, nf, ei, ed, na, tgt = prepare_batch(bi)
            optimizer.zero_grad()
            pred = model(sp, nf, ei, ed, na)
            loss = nn.MSELoss()(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        scheduler.step(epoch_loss / n_b)

    model.eval()
    all_pred, all_true = [], []
    test_list = test_idx.tolist()
    with torch.no_grad():
        for b in range(0, len(test_list), batch_size):
            bi = test_list[b:b+batch_size]
            sp, nf, ei, ed, na, tgt = prepare_batch(bi)
            pred = model(sp, nf, ei, ed, na)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(tgt.cpu().numpy())
    return mean_absolute_error(all_true, all_pred)


# ============================================================
# 3-fold CV comparison
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Proof of Concept: p-cMBDF as GNN Initial Embeddings", flush=True)
print("=" * 60, flush=True)

kf = KFold(n_splits=3, shuffle=True, random_state=42)

for mode, label in [('learned', 'Random embeddings (CGCNN-like)'),
                     ('frozen', 'p-cMBDF frozen (not refined)'),
                     ('refine', 'p-cMBDF refined by GNN')]:
    print("\n--- %s ---" % label, flush=True)
    fold_maes = []
    for fold_idx, (tr, te) in enumerate(kf.split(np.arange(N))):
        mae = train_eval(mode, tr, te, n_epochs=200)
        fold_maes.append(mae)
        print("  Fold %d: %.4f eV/atom" % (fold_idx+1, mae), flush=True)
    mean = np.mean(fold_maes)
    print("  === %s: %.4f +/- %.4f ===" % (label, mean, np.std(fold_maes)), flush=True)

print("\nContext: ALIGNN on perovskites = 0.033 eV/atom (5-fold, ~5M params)", flush=True)
print("         p-cMBDF + KRR = 0.218 eV/atom (5-fold)", flush=True)
print("Done!", flush=True)
