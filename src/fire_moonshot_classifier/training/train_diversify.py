"""
DIVERSIFY-based drone autopilot classifier  (PX4 vs ArduPilot)

Input : build_features.py cache (X_seq), converted to 2-second sliding
        windows at 50 Hz with shape
        (N_FEAT=len(config.TARGET_FEATURES), WIN_LEN=100).
        Current features: XY-Accel, XY-Jerk, Curvature.

Class y : 0 = ArduPilot,  1 = PX4
Latent domain d' : auto-discovered (K=5) by DIVERSIFY cosine k-means
Strategy A : Sim2Real barrier — GRL adversarial domain confusion
Strategy B : time-scale / agility invariance — kinematic features already handle it
"""

import os
import sys
import json
import argparse
import random
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.decomposition import PCA
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_moonshot_classifier.datamanager import config

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "processor"))
from trajectory_processor import (
    process_px4_flight_data, process_ardu_flight_data, process_rosbag_flight_data)

# ── hyper-parameters ──────────────────────────────────────────────────────────
FEAT_HZ         = 50
WIN_SEC         = 2.0
WIN_LEN         = int(WIN_SEC * FEAT_HZ)   # 100 samples
HOP_LEN         = WIN_LEN // 2             # 50 samples = 50 % overlap
# Legacy raw-log training uses seven kinematic channels. main() replaces this
# with the X_seq channel count when training from the project feature cache.
N_FEAT          = 7
NUM_CLASSES     = 2                        # 0=ArduPilot, 1=PX4
LATENT_DOMAIN_N = 5
BOTTLENECK_DIM  = 32
DIS_HIDDEN      = 64
CNN_CH          = 128                      # final CNN channel count
ATTN_HIDDEN     = 64                       # temporal attention hidden dim
ALPHA           = 1.0
ALPHA1          = 1.0
LAM             = 0.0
LOCAL_EPOCH     = 3
MAX_EPOCH       = 100
# CKPT_INTERVAL   = 10   # save a checkpoint every N rounds (for double-descent curve)
LR              = 1e-3
LR_DECAY1       = 0.1
LR_DECAY2       = 1.0
WEIGHT_DECAY    = 5e-4
BETA1           = 0.9
BATCH_SIZE      = 128
SEED            = 42
MIN_WIN         = 30
REALFLIGHT_DIR  = PROJECT_ROOT / "data"
OOD_PCTILE = 95   # 95th-percentile of SITL-val kNN distances → threshold
KNN_K      = 5    # k-th nearest neighbor for OOD scoring

# ── Multiple Instance Learning (count-based MIL, Weidmann 2003; Foulds&Frank 2010)
# A flight (bag) is classified only if enough windows (instances) pass the OOD gate
# (= in-distribution / "valid features"); otherwise → Unknown.
# Two AND-ed conditions:
#   MIL_MIN_VALID : small absolute floor — need ≥N windows for any statistical decision
#   MIL_MIN_FRAC  : elastic quorum that scales with flight length (the main criterion;
#                   prevents short flights from being unfairly rejected by a fixed count)
MIL_MIN_VALID = 3     # statistical floor (keep small; fraction does the real gating)
MIL_MIN_FRAC  = 0.15  # require n_valid/n_total ≥ this (length-elastic quorum)

# ── Dual-path BN (AdaBN, Li et al. 2017) ──────────────────────────────────────
# OOD path always uses original SITL BatchNorm stats (eval mode) so the kNN-OOD
# gate keeps its SITL reference. Classification path can adapt BN to the real
# domain (unsupervised, label-free) to bridge the Sim2Real shift before classifying.
#   "off"        — classification also uses SITL BN (baseline)
#   "offline"    — BN stats estimated ONCE from a MIXED-class pool of real windows
#                  (works: pool contains both firmwares so class shift is preserved)
#   "per_flight" — BN stats from each flight's own windows (FAILS here: a flight is
#                  single-class, so BN centers away the class signal — kept for study)
ADABN_MODE    = "off"   # experiments showed AdaBN doesn't recover ArduPilot here;
                        # "offline"/"per_flight" kept available for study
ADABN_MIN_WIN = 4       # need ≥ this many windows for stable batch BN stats, else fall back

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

PX4_FOLDER  = PROJECT_ROOT / "data/px4_logs"
ARDU_FOLDER = PROJECT_ROOT / "data/ardu_logs"
TEST_RATIO  = 0.2

WANDB_PROJECT = "drone-firmware-classifier"
GIT_SHA       = None   # set by sweep_diversify._apply_sweep_config before each trial


def _filename_label(fname: str) -> str | None:
    """Derive ground-truth autopilot label from CSV filename, or None if unknown."""
    fl = fname.lower()
    if "px4" in fl:
        return "PX4"
    if "ardu" in fl:
        return "ArduPilot"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Network modules
# ══════════════════════════════════════════════════════════════════════════════

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class TemporalAttentionPooling(nn.Module):
    """
    Learnable attention pooling over the temporal dimension.
    Input: (B, C, T) → Output: (B, C)
    """
    def __init__(self, in_channels, hidden_dim= ATTN_HIDDEN):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C)
        attn_weights = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T, 1)

        out = (x * attn_weights).sum(dim=1)  # (B, C)
        
        return out, attn_weights
    
class FlightFeaturizer(nn.Module):
    """
    1D-CNN on fixed (B, N_FEAT, WIN_LEN) input.
      Conv1(N_FEAT→32, k=7) + MaxPool(2) → (B, 32, 50)
      Conv2(32→64, k=5) + MaxPool(2) → (B, 64, 25)
      Conv3(64→128,k=3)              → (B, 128, 25)
      AdaptiveAvgPool1d(1)           → (B, 128)
    """
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(N_FEAT, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, CNN_CH, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(CNN_CH), nn.ReLU(),
        )
        self.attn_pool = TemporalAttentionPooling(CNN_CH)
        self.in_features = CNN_CH

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))  # (B, CNN_CH, L')
        h, _ = self.attn_pool(x)                      # Attention Applied Pooling to get (B, CNN_CH)
        return h                                      # (B, CNN_CH)

    def forward_features(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        h, attn_weights = self.attn_pool(f3)          
        return f1, f2, h


class FeatBottleneck(nn.Module):
    def __init__(self, in_dim, out_dim=BOTTLENECK_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return self.bn(self.fc(x))


class FeatClassifier(nn.Module):
    def __init__(self, n_classes, in_dim=BOTTLENECK_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class PrototypeClassifier(nn.Module):
    """
    Distance-based classifier: logit = -dist(x, prototype_k).
    Low max-logit (= large min-distance) → OOD signal.
    """
    def __init__(self, n_classes, in_dim=BOTTLENECK_DIM):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, in_dim))

    def forward(self, x):
        dist = torch.cdist(x, self.prototypes)  # (B, n_classes)
        return -dist


class Discriminator(nn.Module):
    def __init__(self, in_dim=BOTTLENECK_DIM, hidden=DIS_HIDDEN, n_out=LATENT_DOMAIN_N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class FlightDataset:
    """
    Each item: (x, ctarget, dtarget, pctarget, pdtarget, index)
      x       : (N_FEAT, WIN_LEN) float32 tensor
      ctarget : class label  0=ArduPilot  1=PX4
      dtarget : original domain id (per-file, unused by algorithm but kept for parity)
      pctarget: -1 (unused)
      pdtarget: pseudo-domain, initialised 0, updated by DiversifyFlight.set_dlabel
      index   : integer index (needed by set_dlabel)
    """
    def __init__(self, x: list, labels: np.ndarray, dlabels: np.ndarray):
        self.x        = x                              # list of (N_FEAT, WIN_LEN) tensors
        self.labels   = labels.astype(np.int64)
        self.dlabels  = dlabels.astype(np.int64)
        self.pclabels = np.full(len(labels), -1, np.int64)
        self.pdlabels = np.zeros(len(labels), np.int64)

    def set_labels_by_index(self, tlabels, tindex, label_type):
        if label_type == 'pdlabel':
            self.pdlabels[tindex] = tlabels

    def __getitem__(self, idx):
        return (self.x[idx],
                int(self.labels[idx]),
                int(self.dlabels[idx]),
                int(self.pclabels[idx]),
                int(self.pdlabels[idx]),
                idx)

    def __len__(self):
        return len(self.labels)

    @classmethod
    def subset(cls, ds, indices):
        d = cls.__new__(cls)
        d.x        = [ds.x[i] for i in indices]
        d.labels   = ds.labels[indices]
        d.dlabels  = ds.dlabels[indices]
        d.pclabels = ds.pclabels[indices]
        d.pdlabels = ds.pdlabels[indices]
        return d


def _collate(batch):
    xs, cls_l, dom_l, pc_l, pd_l, idxs = zip(*batch)
    return (torch.stack(xs),
            torch.tensor(cls_l, dtype=torch.long),
            torch.tensor(dom_l, dtype=torch.long),
            torch.tensor(pc_l,  dtype=torch.long),
            torch.tensor(pd_l,  dtype=torch.long),
            np.array(idxs))


def _make_loader(ds, shuffle, batch_size=BATCH_SIZE):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=0, collate_fn=_collate)


# ══════════════════════════════════════════════════════════════════════════════
# DIVERSIFY algorithm (self-contained, CPU-friendly)
# ══════════════════════════════════════════════════════════════════════════════

def _entropy_logits(logits):
    p = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5), dim=1))


class DiversifyFlight(nn.Module):
    def __init__(self):
        super().__init__()
        fea_dim = FlightFeaturizer().in_features   # CNN_CH = 128

        self.featurizer     = FlightFeaturizer()
        # d-branch: domain-discovery (adversarial class confusion + pseudo-domain clf)
        self.dbottleneck    = FeatBottleneck(fea_dim)
        self.ddiscriminator = Discriminator(BOTTLENECK_DIM, DIS_HIDDEN, NUM_CLASSES)
        self.dclassifier    = FeatClassifier(LATENT_DOMAIN_N)
        # main branch: prototype class clf + GRL pseudo-domain adversary
        self.bottleneck     = FeatBottleneck(fea_dim)
        self.classifier     = PrototypeClassifier(NUM_CLASSES)
        self.discriminator  = Discriminator(BOTTLENECK_DIM, DIS_HIDDEN, LATENT_DOMAIN_N)
        # auxiliary branch: joint (pseudo-domain × class) classifier
        self.abottleneck    = FeatBottleneck(fea_dim)
        self.aclassifier    = FeatClassifier(NUM_CLASSES * LATENT_DOMAIN_N)
        
        self.l1_bottleneck    = FeatBottleneck(32)
        self.l1_classifier    = FeatClassifier(NUM_CLASSES)

    # Phase 1 — auxiliary branch
    def update_a(self, batch, opt):
        x  = batch[0].to(DEVICE).float()
        c  = batch[1].to(DEVICE).long()   # class label
        pd = batch[4].to(DEVICE).long()   # pseudo-domain
        y  = pd * NUM_CLASSES + c
        # z  = self.abottleneck(self.featurizer(x))
        # loss = F.cross_entropy(self.aclassifier(z), y)
        # opt.zero_grad(); loss.backward(); opt.step()

        z_full = self.abottleneck(self.featurizer(x))
        loss_main = F.cross_entropy(self.aclassifier(z_full), y)
        
        # 2. 보조 학습: L1 층에서만 기종 차이를 억지로 학습하도록 멱살 잡기
        f1, _, _ = self.featurizer.forward_features(x)
        z_l1 = f1.mean(dim=-1)
        z_l1_out = self.l1_bottleneck(z_l1)
        loss_l1 = F.cross_entropy(self.l1_classifier(z_l1_out), c)
        
        # 3. 두 Loss를 더해서 역전파 (네트워크 전체 + L1 집중 학습)
        loss = loss_main + loss_l1
        
        opt.zero_grad(); loss.backward(); opt.step()

        return loss.item()

    # Phase 2 — d-branch: adversarial class-confusion + pseudo-domain classification
    def update_d(self, batch, opt):
        x  = batch[0].to(DEVICE).float()
        c  = batch[1].to(DEVICE).long()   # class label  → GRL adversary target
        pd = batch[4].to(DEVICE).long()   # pseudo-domain → dclassifier target
        z  = self.dbottleneck(self.featurizer(x))
        rev       = ReverseLayerF.apply(z, ALPHA1)
        disc_loss = F.cross_entropy(self.ddiscriminator(rev), c)
        cd        = self.dclassifier(z)
        cls_loss  = F.cross_entropy(cd, pd) + LAM * _entropy_logits(cd)
        loss = disc_loss + cls_loss
        opt.zero_grad(); loss.backward(); opt.step()
        return {'total': loss.item(), 'dis': disc_loss.item(), 'cls': cls_loss.item()}

    # Pseudo-domain label update via cosine k-means
    @torch.no_grad()
    def set_dlabel(self, loader):
        self.eval()
        feats, outputs, indices = [], [], []
        for batch in loader:
            x   = batch[0].to(DEVICE).float()
            idx = batch[5]
            z   = self.dbottleneck(self.featurizer(x))
            o   = self.dclassifier(z)
            feats.append(z.cpu())
            outputs.append(o.cpu())
            indices.append(idx if isinstance(idx, np.ndarray) else np.array(idx))

        all_fea = torch.cat(feats, 0)
        all_out = torch.cat(outputs, 0)
        all_idx = np.hstack(indices)

        all_fea = torch.cat([all_fea, torch.ones(len(all_fea), 1)], 1)
        all_fea = (all_fea.T / all_fea.norm(p=2, dim=1)).T.numpy()
        aff     = F.softmax(all_out, dim=1).numpy()
        K       = aff.shape[1]
        initc   = aff.T @ all_fea / (1e-8 + aff.sum(0)[:, None])
        pred    = cdist(all_fea, initc, 'cosine').argmin(1)
        aff2    = np.eye(K)[pred]
        initc   = aff2.T @ all_fea / (1e-8 + aff2.sum(0)[:, None])
        pred    = cdist(all_fea, initc, 'cosine').argmin(1)

        loader.dataset.set_labels_by_index(pred, all_idx, 'pdlabel')
        print(f"    Pseudo-domain dist: {dict(Counter(pred.tolist()))}")
        self.train()

    # Phase 3 — main branch: class clf + GRL pseudo-domain adversary
    def update(self, batch, opt):
        x  = batch[0].to(DEVICE).float()
        y  = batch[1].to(DEVICE).long()   # class label
        pd = batch[4].to(DEVICE).long()   # pseudo-domain
        z  = self.bottleneck(self.featurizer(x))
        cls_loss  = F.cross_entropy(self.classifier(z), y)
        rev       = ReverseLayerF.apply(z, ALPHA)
        disc_loss = F.cross_entropy(self.discriminator(rev), pd)
        loss = cls_loss + disc_loss
        opt.zero_grad(); loss.backward(); opt.step()
        return {'total': loss.item(), 'cls': cls_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))

    def extract_z(self, x):
        """Bottleneck features (B, BOTTLENECK_DIM) — used by classifier."""
        return self.bottleneck(self.featurizer(x))

    def extract_ood_features(self, x):
        """
        Single-pass extraction of two kNN OOD levels:
          [0] block1 avg-pooled (B, 32) — pre-GRL, retains firmware micro-noise → Near-OOD (Cognipilot)
          [1] bottleneck        (B, 32) — post-GRL → Far-OOD + classification
        """
        f1, f2, h = self.featurizer.forward_features(x)
        z_l1 = f1.mean(dim=-1)       # (B, 32)
        z_bn  = self.bottleneck(h)   # (B, 32)
        return [z_l1, z_bn]
    
    def extract_multi_z(self, x):
        """
        Multi-level features for Mahalanobis OOD (Lee et al. 2018, NeurIPS).
          L0 block1     (B, 32)   shallow — least affected by GRL,
                                   retains firmware-specific micro-noise → Near-OOD
          L1 block2     (B, 64)   mid
          L2 bottleneck (B, 32)   deep, post-GRL → Far-OOD (physical abnormality)
        Spatial dim is globally-average-pooled (Lee 2018 convention).
        """
        f1, f2, h = self.featurizer.forward_features(x)
        z0 = f1.mean(dim=-1)               # (B, 32)
        z1 = f2.mean(dim=-1)               # (B, 64)
        z2 = self.bottleneck(h)            # (B, 32)
        return [z0, z1, z2]


def _make_optimizers(model):
    opta = torch.optim.Adam([
        {'params': model.featurizer.parameters(),  'lr': LR_DECAY1 * LR},
        {'params': model.abottleneck.parameters(), 'lr': LR_DECAY2 * LR},
        {'params': model.aclassifier.parameters(), 'lr': LR_DECAY2 * LR},
        {'params': model.l1_bottleneck.parameters(), 'lr': LR_DECAY2 * LR},
        {'params': model.l1_classifier.parameters(), 'lr': LR_DECAY2 * LR},
    ], lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.9))
    optd = torch.optim.Adam([
        {'params': model.dbottleneck.parameters(),    'lr': LR_DECAY2 * LR},
        {'params': model.dclassifier.parameters(),    'lr': LR_DECAY2 * LR},
        {'params': model.ddiscriminator.parameters(), 'lr': LR_DECAY2 * LR},
    ], lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.9))
    opt = torch.optim.Adam([
        {'params': model.bottleneck.parameters(),    'lr': LR_DECAY2 * LR},
        {'params': model.classifier.parameters(),    'lr': LR_DECAY2 * LR},
        {'params': model.discriminator.parameters(), 'lr': LR_DECAY2 * LR},
    ], lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.9))
    return opta, optd, opt


# ══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════════════════

def _zscore(feat):
    """Z-score each channel. feat: (N_FEAT, WIN_LEN)"""
    mean = feat.mean(axis=1, keepdims=True)
    std  = feat.std(axis=1,  keepdims=True)
    return (feat - mean) / (std + 1e-8)


def _slide_windows(feat_extracted):
    """
    Slide WIN_LEN-sample windows with HOP_LEN hop over feat_extracted (T, N_FEAT).
    Returns list of (N_FEAT, WIN_LEN) float32 tensors.
    """
    T = len(feat_extracted)
    if T < MIN_WIN:
        return []
    wins = []
    for s in range(0, T - WIN_LEN + 1, HOP_LEN):
        seg = feat_extracted[s:s + WIN_LEN].T.copy().astype(np.float32)  # (N_FEAT, WIN_LEN)
        seg = _zscore(seg)
        if np.any(~np.isfinite(seg)):
            continue
        wins.append(torch.tensor(seg, dtype=torch.float32))
    return wins


def load_sitl_windows(file_list):
    """
    Load SITL logs, slide windows over feat_extracted (already 50Hz, 7 features).
    file_list : list of (path, label)  label: 0=ArduPilot, 1=PX4
    Returns FlightDataset.
    """
    all_x, all_y, all_d = [], [], []
    for domain_id, (path, label) in enumerate(file_list):
        fn = process_px4_flight_data if label == 1 else process_ardu_flight_data
        result = fn(str(path))
        if result is None:
            continue
        feat_extracted = result[4]   # (T, 7) at 50Hz
        if feat_extracted is None or len(feat_extracted) < MIN_WIN:
            continue
        wins = _slide_windows(feat_extracted)
        for w in wins:
            all_x.append(w)
            all_y.append(label)
            all_d.append(domain_id)

    return FlightDataset(all_x,
                         np.array(all_y, dtype=np.int64),
                         np.array(all_d, dtype=np.int64))


def _sequence_length(sequence):
    """Return the unpadded length of an (L, F) sequence from X_seq."""
    non_padding = np.any(np.abs(sequence) > 1e-12, axis=1)
    if not np.any(non_padding):
        return 0
    return int(np.flatnonzero(non_padding)[-1] + 1)


def _windows_from_sequence(sequence):
    """Convert one cached (L, F) turn segment to fixed CNN windows."""
    length = _sequence_length(sequence)
    if length < MIN_WIN:
        return []

    sequence = np.asarray(sequence[:length], dtype=np.float32)
    if length < WIN_LEN:
        sequence = np.pad(
            sequence,
            ((0, WIN_LEN - length), (0, 0)),
            mode="edge",
        )
    return _slide_windows(sequence)


def load_cached_windows(cache_path, segment_indices=None):
    """
    Load the project's build_features.py cache.

    The project cache uses 0=PX4 and 1=ArduPilot, while the existing Diversify
    implementation uses 0=ArduPilot and 1=PX4. Labels are converted here so the
    model and its evaluation code retain their original convention.

    Splitting must happen before this function is called. That keeps windows
    from the same turn segment in only one of train/validation/test.
    """
    cached = np.load(cache_path)
    if "X_seq" not in cached or "y" not in cached:
        raise KeyError(f"{cache_path} must contain 'X_seq' and 'y'")

    sequences = cached["X_seq"]
    labels = cached["y"].astype(np.int64)
    if sequences.ndim != 3:
        raise ValueError(f"Expected X_seq shape (N, L, F), got {sequences.shape}")
    if sequences.shape[2] != N_FEAT:
        raise ValueError(
            f"Cache has {sequences.shape[2]} features, but model expects {N_FEAT}. "
            "Rebuild the feature cache after checking config.TARGET_FEATURES."
        )

    if segment_indices is None:
        segment_indices = np.arange(len(labels))

    all_x, all_y, all_d = [], [], []
    for segment_idx in np.asarray(segment_indices, dtype=np.int64):
        project_label = int(labels[segment_idx])
        if project_label not in (0, 1):
            continue
        diversify_label = 1 - project_label
        for window in _windows_from_sequence(sequences[segment_idx]):
            all_x.append(window)
            all_y.append(diversify_label)
            all_d.append(int(segment_idx))

    return FlightDataset(
        all_x,
        np.asarray(all_y, dtype=np.int64),
        np.asarray(all_d, dtype=np.int64),
    )


def split_cached_segments(labels, groups=None, test_ratio=TEST_RATIO, val_ratio=0.15):
    """Split at run level when cache groups exist, otherwise by segment."""
    labels = np.asarray(labels)
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        np.random.shuffle(unique_groups)
        n_test = max(1, int(len(unique_groups) * test_ratio))
        remaining = unique_groups[n_test:]
        n_val = max(1, int(len(remaining) * val_ratio))
        test_groups = unique_groups[:n_test]
        val_groups = remaining[:n_val]
        train_groups = remaining[n_val:]
        return (
            np.flatnonzero(np.isin(groups, train_groups)),
            np.flatnonzero(np.isin(groups, val_groups)),
            np.flatnonzero(np.isin(groups, test_groups)),
        )

    train_idx, val_idx, test_idx = [], [], []

    for cls in (0, 1):
        cls_idx = np.flatnonzero(labels == cls)
        np.random.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_ratio)) if len(cls_idx) > 2 else 0
        remaining = cls_idx[n_test:]
        n_val = max(1, int(len(remaining) * val_ratio)) if len(remaining) > 2 else 0
        test_idx.extend(cls_idx[:n_test])
        val_idx.extend(remaining[:n_val])
        train_idx.extend(remaining[n_val:])

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_accuracy(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        y = batch[1]
        p = model.predict(x).argmax(1).cpu()
        y_t = y if isinstance(y, torch.Tensor) else torch.tensor(y)
        correct += (p == y_t).sum().item()
        total   += len(y_t)
    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def build_knn_bank(model, loader):
    """
    Build two kNN feature banks from training data:
      banks[0]: block1 avg-pooled (Near-OOD — firmware micro-noise, pre-GRL)
      banks[1]: bottleneck        (Far-OOD — physical anomaly, post-GRL)
    """
    model.eval()
    feats_l1, feats_bn, labels = [], [], []
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        y = batch[1]
        z_l1, z_bn = model.extract_ood_features(x)
        feats_l1.append(z_l1.cpu())
        feats_bn.append(z_bn.cpu())
        labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    bank_l1     = torch.cat(feats_l1, dim=0)   # (N, 32)
    bank_bn     = torch.cat(feats_bn, dim=0)   # (N, 32)
    bank_labels = torch.cat(labels,   dim=0)   # (N,)
    print(f"  kNN banks: {len(bank_l1):,} vectors  L1(D={bank_l1.size(1)}) + bottleneck(D={bank_bn.size(1)})  k={KNN_K}")
    return [bank_l1, bank_bn], bank_labels


def _knn_score(z_list, banks):
    """
    오염된 L2(Bottleneck) 거리는 완전히 배제하고, 
    오직 GRL의 영향을 받지 않은 L1(z_list[0]) 특징 공간의 거리만 사용하여 OOD 점수를 계산합니다.
    """
    z_l1 = z_list[0]
    bank_l1 = banks[0]
    
    dists = torch.cdist(z_l1.cpu().float(), bank_l1.float())
    knn_d, _ = dists.topk(KNN_K, dim=1, largest=False)
    
    return knn_d[:, -1]

@torch.no_grad()
def calibrate_threshold(model, loader, banks):
    """95th-percentile of two-level kNN distances on SITL val set."""
    model.eval()
    dists = []
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        d = _knn_score(model.extract_ood_features(x), banks)
        dists.extend(d.tolist())
    dists = np.array(dists)
    thr   = float(np.percentile(dists, OOD_PCTILE))
    print(f"  kNN-OOD  mean={dists.mean():.3f}  std={dists.std():.3f}"
          f"  {OOD_PCTILE}th-pct={thr:.3f}")
    return thr


@torch.no_grad()
def compute_rejection_rate(model, loader, banks, threshold):
    """Fraction of windows with two-level kNN distance > threshold (SITL false-rejection)."""
    model.eval()
    rejected = total = 0
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        d = _knn_score(model.extract_ood_features(x), banks)
        rejected += (d > threshold).sum().item()
        total    += len(x)
    return rejected / max(total, 1)


@contextmanager
def _adabn_classification(model):
    """
    Per-flight AdaBN for the classification path (featurizer + bottleneck only).

    Sets those BatchNorm1d layers to train mode so a forward pass normalizes with
    the CURRENT batch (= this flight's windows) statistics instead of the stored
    SITL running stats. Original running_mean/var/num_batches are saved and
    restored on exit, so the OOD path (which calls extract_ood_features in eval
    mode) keeps its untouched SITL BN reference.
    """
    bn = [m for mod in (model.featurizer, model.bottleneck)
          for m in mod.modules() if isinstance(m, nn.BatchNorm1d)]
    saved = [(m.training,
              None if m.running_mean is None else m.running_mean.clone(),
              None if m.running_var  is None else m.running_var.clone(),
              None if m.num_batches_tracked is None else m.num_batches_tracked.clone())
             for m in bn]
    try:
        for m in bn:
            m.train()                # use batch statistics for this forward
        yield
    finally:
        for m, (tr, rm, rv, nb) in zip(bn, saved):
            m.train(tr)
            if rm is not None: m.running_mean.copy_(rm)
            if rv is not None: m.running_var.copy_(rv)
            if nb is not None: m.num_batches_tracked.copy_(nb)


def _classification_bn(model):
    return [m for mod in (model.featurizer, model.bottleneck)
            for m in mod.modules() if isinstance(m, nn.BatchNorm1d)]


@torch.no_grad()
def compute_adabn_stats(model, pool_x, bs=256):
    """
    Offline AdaBN: estimate classification-path BN stats from a MIXED-class pool of
    real windows (cumulative mean/var over the pool). Returns a list of (mean, var)
    aligned with _classification_bn(model); the model is left unmodified.
    """
    bn = _classification_bn(model)
    saved = [(m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone(), m.momentum, m.training) for m in bn]
    for m in bn:
        m.reset_running_stats(); m.momentum = None; m.train()   # momentum=None → cumulative avg
    for i in range(0, len(pool_x), bs):
        model.bottleneck(model.featurizer(pool_x[i:i + bs].to(DEVICE)))
    adapted = [(m.running_mean.clone(), m.running_var.clone()) for m in bn]
    for m, (rm, rv, nb, mom, tr) in zip(bn, saved):
        m.running_mean.copy_(rm); m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nb); m.momentum = mom; m.train(tr)
    return adapted


@contextmanager
def _apply_adabn_stats(model, adapted):
    """Temporarily swap in pre-computed adapted BN stats (eval mode) for the
    classification path; restore the SITL stats on exit (OOD path unaffected)."""
    bn = _classification_bn(model)
    saved = [(m.running_mean.clone(), m.running_var.clone(), m.training) for m in bn]
    try:
        for m, (am, av) in zip(bn, adapted):
            m.running_mean.copy_(am); m.running_var.copy_(av); m.eval()
        yield
    finally:
        for m, (rm, rv, tr) in zip(bn, saved):
            m.running_mean.copy_(rm); m.running_var.copy_(rv); m.train(tr)


@contextmanager
def _adabn_var_only(model, x):
    """
    Per-flight VARIANCE-only AdaBN: adapt BN running_var to THIS flight's window
    statistics but keep the SITL running_mean. Since a flight is single-class, the
    mean carries the class signal — keeping SITL mean avoids the per-flight collapse
    while still rescaling to the real domain. (Two-pass approximation: the per-layer
    variance is captured under the flight's own mean, then applied with the SITL mean.)
    """
    bn = _classification_bn(model)
    saved = [(m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone(), m.momentum, m.training) for m in bn]
    # pass 1: capture this flight's per-layer variance (cumulative over the batch)
    for m in bn:
        m.reset_running_stats(); m.momentum = None; m.train()
    with torch.no_grad():
        model.bottleneck(model.featurizer(x))
    flight_var = [m.running_var.clone() for m in bn]
    try:
        # pass 2: SITL mean + this-flight variance, eval mode
        for m, (rm, rv, nb, mom, tr), fv in zip(bn, saved, flight_var):
            m.running_mean.copy_(rm)        # keep SITL mean (preserves class signal)
            m.running_var.copy_(fv)         # adapt variance to this flight
            m.num_batches_tracked.copy_(nb); m.momentum = mom; m.eval()
        yield
    finally:
        for m, (rm, rv, nb, mom, tr) in zip(bn, saved):
            m.running_mean.copy_(rm); m.running_var.copy_(rv)
            m.num_batches_tracked.copy_(nb); m.momentum = mom; m.train(tr)


def evaluate_realflight(model, csv_files, banks, threshold):
    """
    Count-based Multiple Instance Learning (Weidmann 2003; Foulds & Frank 2010).
    Each flight is a bag; each window is an instance.

    Dual-path BN (ADABN_MODE):
      * OOD path        — always original SITL BN (eval) → extract_ood_features → kNN
      * Classification  — SITL BN ("off") / offline pooled AdaBN / per-flight AdaBN

    For each real-flight CSV:
      1. window → bottleneck z   (OOD path, SITL BN)
      2. kNN OOD score = distance to k-th nearest training neighbor
      3. Instance is VALID (in-distribution) if score <= threshold
      4. MIL quorum: assign a class only if n_valid >= MIL_MIN_VALID
         (and n_valid/n_total >= MIL_MIN_FRAC); else → Unknown.
      5. Among valid instances only, decide PX4 vs ArduPilot by majority.
    """
    model.eval()

    # ── pass 1: load every flight's windows; accumulate a mixed-class real pool ──
    flights = []   # (fname, X)
    for csv_path in sorted(csv_files):
        fname = Path(csv_path).name
        segments = process_rosbag_flight_data(str(csv_path))
        if not segments:
            print(f"  [SKIP] {fname}  (no valid segments)")
            continue
        all_wins = []
        for seg_info in segments:
            feat = seg_info['data'][4]
            if feat is None or len(feat) < MIN_WIN:
                continue
            all_wins.extend(_slide_windows(feat))
        if not all_wins:
            print(f"  [SKIP] {fname}  (no windows)")
            continue
        flights.append((fname, torch.stack(all_wins).to(DEVICE)))

    # ── offline AdaBN: estimate classification BN stats once from the mixed pool ─
    adapted = None
    with torch.no_grad():
        if ADABN_MODE == "offline":
            pool = torch.cat([X for _, X in flights], dim=0)
            if len(pool) >= ADABN_MIN_WIN:
                adapted = compute_adabn_stats(model, pool)

    # ── pass 2: per flight — OOD gate (SITL BN) + MIL + classification ───────────
    results = []
    with torch.no_grad():
        for fname, X in flights:
            # OOD path: original SITL BN (eval)
            z_list = model.extract_ood_features(X)      # [z_l1, z_bn] SITL-BN
            d      = _knn_score(z_list, banks)
            mask   = (d <= threshold)                   # valid (in-distribution) instances
            n_rej  = int((~mask).sum())
            n_acc  = int(mask.sum())
            d_mean = float(d.mean())
            n_tot  = len(X)

            quorum_ok = (n_acc >= MIL_MIN_VALID) and (n_acc / n_tot >= MIL_MIN_FRAC)
            if not quorum_ok:
                verdict, p_px4 = "Unknown", float("nan")
                n_px4 = n_ardu = 0
            else:
                # classification path BN per ADABN_MODE
                if ADABN_MODE == "offline" and adapted is not None:
                    with _apply_adabn_stats(model, adapted):
                        logits = model.classifier(model.bottleneck(model.featurizer(X)))
                elif ADABN_MODE == "per_flight" and n_tot >= ADABN_MIN_WIN:
                    with _adabn_classification(model):
                        logits = model.classifier(model.bottleneck(model.featurizer(X)))
                elif ADABN_MODE == "per_flight_var" and n_tot >= ADABN_MIN_WIN:
                    with _adabn_var_only(model, X):
                        logits = model.classifier(model.bottleneck(model.featurizer(X)))
                else:   # "off" or fallback
                    logits = model.classifier(z_list[1])
                probs_in = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
                p_px4    = float(probs_in.mean())
                n_px4    = int((probs_in > 0.5).sum())
                n_ardu   = n_acc - n_px4
                verdict  = "PX4" if n_px4 >= n_ardu else "ArduPilot"

            print(f"  {fname:<50s}  kNN={d_mean:.3f}  valid={n_acc}/{n_tot}"
                  f"({100*n_acc/n_tot:.0f}%)  → {verdict}")
            results.append({"file": fname, "prediction": verdict,
                "px4_prob":    round(p_px4, 3) if not np.isnan(p_px4) else None,
                "knn_dist":    round(d_mean, 4),
                "n_windows":   n_tot,
                "n_accepted":  n_acc,
                "n_rejected":  n_rej,
                "n_px4":       n_px4,
                "n_ardu":      n_ardu,
                "reject_rate": round(n_rej / n_tot, 4)})
    model.train()
    return results


@torch.no_grad()
def evaluate_cached_realflight(model, cache_paths, banks, threshold):
    """Evaluate and aggregate cached real-flight turn segments by run folder."""
    grouped = {}
    for cache_path in cache_paths:
        cached = np.load(cache_path)
        if not {"X_seq", "y", "runs"}.issubset(cached.files):
            print(f"  [SKIP] {cache_path.name} (X_seq/y/runs missing)")
            continue

        for sequence, project_label, run_name in zip(
            cached["X_seq"], cached["y"], cached["runs"]
        ):
            if int(project_label) not in (0, 1):
                continue
            windows = _windows_from_sequence(sequence)
            if not windows:
                continue
            key = str(run_name)
            entry = grouped.setdefault(
                key,
                {"windows": [], "label": 1 - int(project_label)},
            )
            if entry["label"] != 1 - int(project_label):
                raise ValueError(f"Conflicting labels found for real-flight run {key}")
            entry["windows"].extend(windows)

    model.eval()
    results = []
    for run_name, entry in sorted(grouped.items()):
        x = torch.stack(entry["windows"]).to(DEVICE)
        z_list = model.extract_ood_features(x)
        distances = _knn_score(z_list, banks)
        accepted = distances <= threshold
        n_total = len(x)
        n_accepted = int(accepted.sum())
        n_rejected = n_total - n_accepted
        quorum_ok = (
            n_accepted >= MIL_MIN_VALID
            and n_accepted / n_total >= MIL_MIN_FRAC
        )

        if quorum_ok:
            logits = model.classifier(z_list[1])
            predictions = logits[accepted].argmax(dim=1)
            predicted_label = int(torch.mode(predictions).values)
            prediction = "PX4" if predicted_label == 1 else "ArduPilot"
            n_px4 = int((predictions == 1).sum())
            n_ardu = n_accepted - n_px4
        else:
            predicted_label = None
            prediction = "Unknown"
            n_px4 = n_ardu = 0

        true_label = int(entry["label"])
        ground_truth = "PX4" if true_label == 1 else "ArduPilot"
        results.append(
            {
                "file": run_name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": predicted_label == true_label,
                "knn_dist": round(float(distances.mean()), 4),
                "n_windows": n_total,
                "n_accepted": n_accepted,
                "n_rejected": n_rejected,
                "n_px4": n_px4,
                "n_ardu": n_ardu,
                "reject_rate": round(n_rejected / n_total, 4),
            }
        )
        print(
            f"  {run_name:<40} true={ground_truth:<11} "
            f"pred={prediction:<11} valid={n_accepted}/{n_total}"
        )

    model.train()
    return results


def save_evaluation_report(
    output_path,
    y_true,
    y_pred,
    best_val_acc,
    best_test_acc,
    ood_threshold,
    sitl_false_reject,
    real_results,
):
    """Save SITL and real-flight evaluation results in a readable text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sitl_report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["ArduPilot", "PX4"],
        zero_division=0,
    ).rstrip()
    sitl_accuracy = (
        float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
        if y_true
        else 0.0
    )

    labeled_results = [
        result
        for result in real_results
        if result.get("ground_truth") in ("PX4", "ArduPilot")
    ]
    real_correct = sum(bool(result.get("correct")) for result in labeled_results)
    real_accuracy = (
        real_correct / len(labeled_results) if labeled_results else 0.0
    )

    lines = [
        "CNN + DIVERSIFY Evaluation Results",
        "=" * 115,
        "",
        "SITL Evaluation",
        "-" * 115,
        f"Best validation accuracy : {best_val_acc * 100:.2f}%",
        f"Best-round test accuracy : {best_test_acc * 100:.2f}%",
        f"Final test accuracy      : {sitl_accuracy * 100:.2f}% "
        f"({sum(a == b for a, b in zip(y_true, y_pred))}/{len(y_true)})",
        f"OOD threshold            : {ood_threshold:.6f}",
        f"SITL false rejection     : {sitl_false_reject * 100:.2f}%",
        "",
        "SITL Classification Report",
        "-" * 115,
        sitl_report,
        "",
        "Real Flight Evaluation",
        "-" * 115,
        f"Accuracy: {real_accuracy * 100:.2f}% "
        f"({real_correct}/{len(labeled_results)})",
        f"Unknown predictions: "
        f"{sum(r.get('prediction') == 'Unknown' for r in real_results)}",
        "",
        f"{'No.':<4} | {'Run Folder':<40} | {'True Label':<12} | "
        f"{'Prediction':<12} | {'Status':<7} | {'Valid Windows':<15} | "
        f"{'kNN Dist.':>9}",
        "-" * 115,
    ]

    for index, result in enumerate(real_results, start=1):
        status = "Match" if result.get("correct") else "Fail"
        if result.get("prediction") == "Unknown":
            status = "Unknown"
        valid_windows = (
            f"{result.get('n_accepted', 0)}/{result.get('n_windows', 0)}"
        )
        lines.append(
            f"{index:<4} | {result.get('file', 'Unknown'):<40} | "
            f"{result.get('ground_truth', 'Unknown'):<12} | "
            f"{result.get('prediction', 'Unknown'):<12} | "
            f"{status:<7} | {valid_windows:<15} | "
            f"{result.get('knn_dist', float('nan')):>9.4f}"
        )

    lines.append("-" * 115)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Text report: {output_path}")


@torch.no_grad()
def save_diversify_figures(
    model,
    test_loader,
    y_true,
    y_pred,
    real_results,
    save_dir,
    permutation_repeats=10,
):
    """
    Save a confusion matrix and channel-wise permutation sensitivity plot.

    Sensitivity is the decrease in SITL test accuracy after replacing one
    complete feature channel with the same channel from randomly selected
    samples. Repeating the permutation provides a mean and standard deviation.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    label_to_index = {"ArduPilot": 0, "PX4": 1, "Unknown": 2}
    real_true = [
        label_to_index[result["ground_truth"]]
        for result in real_results
        if result.get("ground_truth") in ("ArduPilot", "PX4")
    ]
    real_pred = [
        label_to_index.get(result.get("prediction"), 2)
        for result in real_results
        if result.get("ground_truth") in ("ArduPilot", "PX4")
    ]
    cm = confusion_matrix(real_true, real_pred, labels=[0, 1, 2])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["ArduPilot", "PX4", "Unknown"],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(
        "CNN + DIVERSIFY Real-Flight Confusion Matrix",
        fontweight="bold",
    )
    fig.tight_layout()
    confusion_path = save_dir / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    all_x, all_y = [], []
    for batch in test_loader:
        all_x.append(batch[0].cpu().float())
        all_y.append(batch[1].cpu().long())
    x_test = torch.cat(all_x)
    labels = torch.cat(all_y).numpy()
    baseline_accuracy = float(
        np.mean(np.asarray(y_true, dtype=np.int64) == np.asarray(y_pred))
    )

    def predict_in_batches(inputs):
        predictions = []
        model.eval()
        for start in range(0, len(inputs), BATCH_SIZE):
            logits = model.predict(
                inputs[start:start + BATCH_SIZE].to(DEVICE)
            )
            predictions.append(logits.argmax(dim=1).cpu())
        return torch.cat(predictions).numpy()

    rng = torch.Generator().manual_seed(SEED)
    sensitivity = np.zeros((x_test.shape[1], permutation_repeats), dtype=float)
    for feature_idx in range(x_test.shape[1]):
        for repeat_idx in range(permutation_repeats):
            permuted = x_test.clone()
            order = torch.randperm(len(x_test), generator=rng)
            permuted[:, feature_idx, :] = x_test[order, feature_idx, :]
            permuted_predictions = predict_in_batches(permuted)
            permuted_accuracy = float(np.mean(permuted_predictions == labels))
            sensitivity[feature_idx, repeat_idx] = (
                baseline_accuracy - permuted_accuracy
            )

    feature_names = list(config.TARGET_FEATURES)
    if len(feature_names) != x_test.shape[1]:
        feature_names = [
            f"Feature {index + 1}" for index in range(x_test.shape[1])
        ]
    mean_drop = sensitivity.mean(axis=1) * 100
    std_drop = sensitivity.std(axis=1) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        feature_names,
        mean_drop,
        yerr=std_drop,
        capsize=5,
        color=["tab:green", "tab:orange", "tab:blue"][:len(feature_names)],
        alpha=0.85,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Test accuracy decrease (percentage points)")
    ax.set_title(
        "CNN + DIVERSIFY Feature Sensitivity\n"
        f"Permutation importance ({permutation_repeats} repeats)",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in mean_drop], padding=3)
    fig.tight_layout()
    sensitivity_path = save_dir / "feature_sensitivity.png"
    fig.savefig(sensitivity_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Confusion matrix: {confusion_path}")
    print(f"  Feature sensitivity: {sensitivity_path}")


@torch.no_grad()
def save_knn_distribution_figures(
    model,
    train_loader,
    test_loader,
    real_cache_paths,
    banks,
    threshold,
    save_dir,
):
    """
    Project L1 OOD features to 2D while retaining their original-space kNN score.

    PCA is fitted only on SITL training features. The displayed color and
    in/OOD status always use the 32-dimensional kNN distance from _knn_score;
    distance is never recomputed in the PCA plane.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    def collect_loader(loader, source):
        features, distances, labels = [], [], []
        for batch in loader:
            x = batch[0].to(DEVICE).float()
            z_l1 = model.extract_ood_features(x)[0]
            features.append(z_l1.cpu().numpy())
            distances.append(
                _knn_score([z_l1], banks).cpu().numpy()
            )
            labels.append(batch[1].cpu().numpy())
        if not features:
            return None
        return {
            "source": source,
            "features": np.concatenate(features),
            "distances": np.concatenate(distances),
            "labels": np.concatenate(labels),
        }

    datasets = [
        collect_loader(train_loader, "SITL Train"),
        collect_loader(test_loader, "SITL Test"),
    ]
    for cache_path in real_cache_paths:
        real_dataset = load_cached_windows(cache_path)
        if len(real_dataset) == 0:
            continue
        datasets.append(
            collect_loader(
                _make_loader(real_dataset, shuffle=False),
                f"Real: {cache_path.stem.replace('_features', '')}",
            )
        )
    datasets = [dataset for dataset in datasets if dataset is not None]
    if not datasets:
        print("  [Warning] No latent features available for kNN plots.")
        return

    train_data = datasets[0]
    pca = PCA(n_components=2)
    train_2d = pca.fit_transform(train_data["features"])

    # Fix PCA sign ambiguity so PX4 (label 1) lies in the positive direction
    # from ArduPilot (label 0), making plots from separate runs comparable.
    train_labels = train_data["labels"]
    class_delta = (
        train_2d[train_labels == 1].mean(axis=0)
        - train_2d[train_labels == 0].mean(axis=0)
    )
    axis_signs = np.where(class_delta < 0, -1.0, 1.0)
    train_data["coordinates"] = train_2d * axis_signs
    for dataset in datasets[1:]:
        dataset["coordinates"] = (
            pca.transform(dataset["features"]) * axis_signs
        )

    all_distances = np.concatenate(
        [dataset["distances"] for dataset in datasets]
    )
    color_max = max(
        float(np.percentile(all_distances, 99)),
        float(threshold),
        1e-8,
    )
    source_markers = ["o", "^", "*", "P", "D", "s"]

    # Plot 1: continuous original-space kNN distance on the PCA plane.
    fig, ax = plt.subplots(figsize=(11, 7))
    scatter_for_colorbar = None
    for index, dataset in enumerate(datasets):
        coordinates = dataset["coordinates"]
        marker = source_markers[index % len(source_markers)]
        scatter_for_colorbar = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=dataset["distances"],
            cmap="turbo",
            vmin=0,
            vmax=color_max,
            marker=marker,
            s=90 if dataset["source"].startswith("Real") else 28,
            alpha=0.8 if dataset["source"].startswith("Real") else 0.45,
            edgecolors="black" if dataset["source"].startswith("Real") else "none",
            linewidths=0.4,
            label=dataset["source"],
        )
    colorbar = fig.colorbar(scatter_for_colorbar, ax=ax)
    colorbar.set_label("kNN distance in 32D L1 latent space")
    colorbar.ax.axhline(
        min(threshold, color_max),
        color="white",
        linewidth=2,
    )
    ax.set_xlabel(
        f"Latent PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    )
    ax.set_ylabel(
        f"Latent PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    )
    ax.set_title(
        "CNN + DIVERSIFY kNN Distance Distribution\n"
        f"Original 32D distance; OOD threshold={threshold:.3f}",
        fontweight="bold",
    )
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5))
    fig.tight_layout()
    distance_path = save_dir / "knn_distance_distribution.png"
    fig.savefig(distance_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: class colors with threshold-based In/OOD markers.
    class_colors = {0: "tab:orange", 1: "tab:green"}
    class_names = {0: "ArduPilot", 1: "PX4"}
    fig, ax = plt.subplots(figsize=(11, 7))
    for source_index, dataset in enumerate(datasets):
        coordinates = dataset["coordinates"]
        accepted = dataset["distances"] <= threshold
        marker = source_markers[source_index % len(source_markers)]
        for class_label in (0, 1):
            in_mask = accepted & (dataset["labels"] == class_label)
            if np.any(in_mask):
                ax.scatter(
                    coordinates[in_mask, 0],
                    coordinates[in_mask, 1],
                    c=class_colors[class_label],
                    marker=marker,
                    s=90 if dataset["source"].startswith("Real") else 28,
                    alpha=0.75,
                    edgecolors="black" if dataset["source"].startswith("Real") else "none",
                    linewidths=0.4,
                    label=(
                        f"{dataset['source']} {class_names[class_label]} (In)"
                    ),
                )
        rejected = ~accepted
        if np.any(rejected):
            ax.scatter(
                coordinates[rejected, 0],
                coordinates[rejected, 1],
                c="red",
                marker="X",
                s=100 if dataset["source"].startswith("Real") else 45,
                edgecolors="black",
                linewidths=0.5,
                label=f"{dataset['source']} (OOD)",
            )

    ax.set_xlabel(
        f"Latent PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    )
    ax.set_ylabel(
        f"Latent PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    )
    ax.set_title(
        "CNN + DIVERSIFY kNN In-Distribution / OOD Status\n"
        f"Threshold={threshold:.3f}",
        fontweight="bold",
    )
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    status_path = save_dir / "knn_ood_status.png"
    fig.savefig(status_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  kNN distance distribution: {distance_path}")
    print(f"  kNN OOD status: {status_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global MAX_EPOCH, LOCAL_EPOCH, BATCH_SIZE, LR, N_FEAT

    parser = argparse.ArgumentParser(
        description="Train the CNN+DIVERSIFY model on project feature caches."
    )
    parser.add_argument("--sitl-folder", default=config.SITL_FOLDER)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCH)
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Train locally without uploading metrics or artifacts.",
    )
    args = parser.parse_args()

    MAX_EPOCH = args.epochs
    LOCAL_EPOCH = args.local_epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr

    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    sitl_cache = config.CACHE_DIR / f"{args.sitl_folder}_features.npz"
    if not sitl_cache.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {sitl_cache}\n"
            "Run `python3 tools/preprocessing/build_features.py` first."
        )
    with np.load(sitl_cache) as cached:
        if "X_seq" not in cached:
            raise KeyError(f"{sitl_cache} does not contain X_seq")
        N_FEAT = int(cached["X_seq"].shape[2])
        cached_labels = cached["y"].copy()
        cached_runs = cached["runs"].copy() if "runs" in cached else None

    _cfg = {
        "feat_hz": FEAT_HZ, "win_sec": WIN_SEC, "win_len": WIN_LEN,
        "hop_len": HOP_LEN, "n_feat": N_FEAT, "num_classes": NUM_CLASSES,
        "latent_domain_n": LATENT_DOMAIN_N, "bottleneck_dim": BOTTLENECK_DIM,
        "dis_hidden": DIS_HIDDEN, "cnn_ch": CNN_CH, "attn_hidden": ATTN_HIDDEN,
        "alpha": ALPHA, "alpha1": ALPHA1, "lam": LAM,
        "local_epoch": LOCAL_EPOCH, "max_epoch": MAX_EPOCH,
        "lr": LR, "lr_decay1": LR_DECAY1, "lr_decay2": LR_DECAY2,
        "weight_decay": WEIGHT_DECAY, "beta1": BETA1,
        "batch_size": BATCH_SIZE, "seed": SEED, "min_win": MIN_WIN,
        "ood_pctile": OOD_PCTILE, "knn_k": KNN_K,
        "mil_min_valid": MIL_MIN_VALID, "mil_min_frac": MIL_MIN_FRAC,
        "adabn_mode": ADABN_MODE,
        "git_sha": GIT_SHA,
    }
    if wandb.run is None:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"cnn_diversify_{ts}",
            job_type="train",
            config=_cfg,
            mode="disabled" if args.no_wandb else None,
        )
    else:
        # called from sweep agent — run already initialised, just sync config
        run = wandb.run
        wandb.config.update(_cfg, allow_val_change=True)
    run.log_code(str(Path(__file__).parent))

    print(f"\n{'='*70}")
    print(f"  CNN + DIVERSIFY  cached sequences  ({ts})")
    print(f"  WIN_LEN={WIN_LEN}  N_FEAT={N_FEAT}  BOTTLENECK={BOTTLENECK_DIM}")
    print(f"  LATENT_K={LATENT_DOMAIN_N}  epochs={MAX_EPOCH}×{LOCAL_EPOCH}  lr={LR}")
    print(f"  alpha={ALPHA}  alpha1={ALPHA1}  lam={LAM}  lr_decay1={LR_DECAY1}")
    print(f"  class: 0=ArduPilot  1=PX4")
    print(f"  device: {DEVICE}")
    print(f"{'='*70}\n")

    # Split turn segments before window generation to prevent leakage between
    # train, validation and test sets.
    train_idx, val_idx, test_idx = split_cached_segments(
        cached_labels, groups=cached_runs
    )
    split_level = "run" if cached_runs is not None else "turn segment"
    print(
        f"Segments ({split_level} split) — train:{len(train_idx)}  "
        f"val:{len(val_idx)}  test:{len(test_idx)}"
    )
    print("Loading cached train windows...")
    train_ds = load_cached_windows(sitl_cache, train_idx)
    print("Loading cached validation windows...")
    val_ds = load_cached_windows(sitl_cache, val_idx)
    print("Loading cached test windows...")
    test_ds = load_cached_windows(sitl_cache, test_idx)

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError(
            "At least one dataset split produced zero windows. Check X_seq, "
            "WIN_LEN/MIN_WIN, and the class counts in the feature cache."
        )

    n_px4  = int((train_ds.labels == 1).sum())
    n_ardu = int((train_ds.labels == 0).sum())
    print(f"Windows — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")
    print(f"  train  PX4:{n_px4} ({100*n_px4/max(len(train_ds),1):.1f}%)"
          f"  Ardu:{n_ardu} ({100*n_ardu/max(len(train_ds),1):.1f}%)")

    train_ld    = _make_loader(train_ds, shuffle=True)
    train_ns_ld = _make_loader(train_ds, shuffle=False)
    val_ld      = _make_loader(val_ds,   shuffle=False)
    test_ld     = _make_loader(test_ds,  shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = DiversifyFlight().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    wandb.config.update({"model_params": n_params,
                         "n_train_windows": len(train_ds),
                         "n_val_windows":   len(val_ds),
                         "n_test_windows":  len(test_ds)})
    model.train()
    opta, optd, opt = _make_optimizers(model)

    best_val_acc  = 0.0
    best_test_acc = 0.0
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"cnn_diversify_{ts}.pt"

    # ── training loop ─────────────────────────────────────────────────────────
    for rnd in range(MAX_EPOCH):
        print(f"\n──── Round {rnd+1:02d}/{MAX_EPOCH} ─────────────────────────────")

        # Phase 1 — auxiliary branch
        for _ in range(LOCAL_EPOCH):
            for b in train_ld: model.update_a(b, opta)

        # Phase 2 — d-branch
        d_tot = d_dis = d_cls = nb = 0
        for _ in range(LOCAL_EPOCH):
            for b in train_ld:
                s = model.update_d(b, optd)
                d_tot += s['total']; d_dis += s['dis']; d_cls += s['cls']; nb += 1
        print(f"  update_d  total={d_tot/nb:.4f}  dis={d_dis/nb:.4f}  cls={d_cls/nb:.4f}")

        # Pseudo-domain reassignment
        model.set_dlabel(train_ns_ld)

        # Phase 3 — main branch
        u_tot = u_cls = u_dis = nb = 0
        for _ in range(LOCAL_EPOCH):
            for b in train_ld:
                s = model.update(b, opt)
                u_tot += s['total']; u_cls += s['cls']; u_dis += s['dis']; nb += 1
        print(f"  update    total={u_tot/nb:.4f}  cls={u_cls/nb:.4f}  dis={u_dis/nb:.4f}")

        tr_acc  = eval_accuracy(model, train_ns_ld)
        val_acc = eval_accuracy(model, val_ld)
        te_acc  = eval_accuracy(model, test_ld)
        print(f"  Acc  train={tr_acc*100:.1f}%  val={val_acc*100:.1f}%"
              f"  SITL-test={te_acc*100:.1f}%")

        wandb.log({
            "round": rnd + 1,
            "loss/update_d_total": d_tot / nb,
            "loss/update_d_dis":   d_dis / nb,
            "loss/update_d_cls":   d_cls / nb,
            "loss/update_total":   u_tot / nb,
            "loss/update_cls":     u_cls / nb,
            "loss/update_dis":     u_dis / nb,
            "acc/train": tr_acc,
            "acc/val":   val_acc,
            "acc/test":  te_acc,
        }, step=rnd + 1)

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = te_acc
            torch.save(model.state_dict(), model_path)
            print(f"  ★ saved  (val={val_acc*100:.1f}%)")
            wandb.run.summary["best/val_acc"]  = best_val_acc
            wandb.run.summary["best/test_acc"] = best_test_acc
            wandb.run.summary["best/round"]    = rnd + 1

        # if (rnd + 1) % CKPT_INTERVAL == 0:
        #     ckpt = model_path.replace(".pt", f"_rnd{rnd+1:03d}.pt")
        #     torch.save(model.state_dict(), ckpt)
        #     print(f"  [ckpt] saved {ckpt}")

    print(f"\n  Best val={best_val_acc*100:.1f}%  → SITL test={best_test_acc*100:.1f}%")

    # ── reload best checkpoint ────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # ── SITL test report ──────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  SITL Test Set")
    y_true, y_pred = [], []
    with torch.no_grad():
        for b in test_ld:
            x = b[0].to(DEVICE).float()
            p = model.predict(x).argmax(1).cpu().tolist()
            y_t = b[1].tolist() if isinstance(b[1], torch.Tensor) else list(b[1])
            y_true.extend(y_t); y_pred.extend(p)
    print(classification_report(y_true, y_pred, target_names=["ArduPilot", "PX4"]))
    # ── kNN OOD calibration ───────────────────────────────────────────────────
    print(f"\n{'='*70}\n  kNN OOD Calibration  (k={KNN_K})")
    bank_feats, _ = build_knn_bank(model, train_ns_ld)
    threshold     = calibrate_threshold(model, val_ld, bank_feats)
    sitl_false_reject = compute_rejection_rate(model, test_ld, bank_feats, threshold)
    print(f"  SITL false-rejection: {sitl_false_reject*100:.1f}%")
    wandb.run.summary["ood/threshold"]         = threshold
    wandb.run.summary["ood/sitl_false_reject"] = sitl_false_reject

    # ── real flight evaluation from the same project caches ──────────────────
    real_cache_paths = [
        config.CACHE_DIR / f"{folder}_features.npz"
        for folder in config.REAL_FLIGHT_FOLDERS
    ]
    real_cache_paths = [path for path in real_cache_paths if path.exists()]
    print(
        f"\n{'='*70}\n  Real Flight Evaluation "
        f"({len(real_cache_paths)} caches)\n{'='*70}\n"
    )
    results = evaluate_cached_realflight(
        model, real_cache_paths, bank_feats, threshold
    )
    save_diversify_figures(
        model,
        test_ld,
        y_true,
        y_pred,
        real_results=results,
        save_dir=PROJECT_ROOT / "results" / "diversify_figs",
    )
    save_knn_distribution_figures(
        model,
        train_ns_ld,
        test_ld,
        real_cache_paths,
        bank_feats,
        threshold,
        save_dir=PROJECT_ROOT / "results" / "diversify_figs",
    )

    ardu    = sum(1 for r in results if r["prediction"] == "ArduPilot")
    px4     = sum(1 for r in results if r["prediction"] == "PX4")
    unknown = sum(1 for r in results if r["prediction"] == "Unknown")

    labeled = [r for r in results if r["ground_truth"] in ("PX4", "ArduPilot")]
    correct = sum(1 for r in labeled if r["correct"])
    accuracy = correct / len(labeled) if labeled else 0.0
    print(f"\n  Total:{len(results)}  ArduPilot:{ardu}  PX4:{px4}  Unknown:{unknown}"
          f"  Accuracy:{correct}/{len(labeled)} ({accuracy*100:.1f}%)")

    statistics_dir = PROJECT_ROOT / "results" / "diversify_statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    out = statistics_dir / f"cnn_diversify_realflight_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    text_report_path = statistics_dir / "real_flight_classification.txt"
    save_evaluation_report(
        text_report_path,
        y_true=y_true,
        y_pred=y_pred,
        best_val_acc=best_val_acc,
        best_test_acc=best_test_acc,
        ood_threshold=threshold,
        sitl_false_reject=sitl_false_reject,
        real_results=results,
    )
    print(f"  Results: {out}\n  Model:   {model_path}")

    # log to history (so the sweep's Bayes optimizer reliably reads the metric)
    wandb.log({
        "realflight/total":     len(results),
        "realflight/ardupilot": ardu,
        "realflight/px4":       px4,
        "realflight/unknown":   unknown,
        "realflight/accuracy":  accuracy,
        "realflight/correct":   correct,
        "realflight/labeled":   len(labeled),
    })
    wandb.run.summary.update({
        "realflight/accuracy":  accuracy,
        "realflight/correct":   correct,
        "realflight/labeled":   len(labeled),
    })
    realflight_table = wandb.Table(
        columns=["file", "ground_truth", "prediction", "correct", "px4_prob", "knn_dist",
                 "n_windows", "n_accepted", "n_rejected", "n_px4",
                 "n_ardu", "reject_rate"],
        data=[[r["file"], r["ground_truth"],
               r["prediction"], r["correct"],
               r.get("px4_prob"), r["knn_dist"],
               r["n_windows"], r["n_accepted"], r["n_rejected"],
               r["n_px4"], r["n_ardu"], r["reject_rate"]] for r in results],
    )
    wandb.log({"realflight/results": realflight_table})

    artifact = wandb.Artifact(
        name="cnn_diversify",
        type="model",
        metadata={
            "timestamp":         ts,
            "best_val_acc":      best_val_acc,
            "best_test_acc":     best_test_acc,
            "ood_threshold":     threshold,
            "sitl_false_reject": sitl_false_reject,
        },
    )
    artifact.add_file(str(model_path))
    artifact.add_file(str(out))
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
