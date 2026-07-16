"""
Domain split visualization: (a) Initial vs (b) Our method (DIVERSIFY) vs (c) Predicted Classes

(a) Cached X_seq feature statistics (mean+std, 2*N_FEAT-dim) → t-SNE
    Color = source domain (SITL-PX4 / SITL-Ardu / Real-PX4 / Real-Ardu)
    Shows the original Sim2Real domain gap.

(b) Trained DIVERSIFY bottleneck features (32-dim) → t-SNE
    Color = pseudo-domain (K=5, discovered by DIVERSIFY)
    Marker = class (PX4 ● / ArduPilot ▲),  Real flight = ★
    Shows class-aware cross-domain clustering.

(c) Trained DIVERSIFY bottleneck features (32-dim) → t-SNE
    Color = Predicted class (Blue=ArduPilot, Red=PX4)
    Marker = true class (PX4 ● / ArduPilot ▲), Real flight = ★
    Shows the actual classification decision boundaries.
"""

import sys
import random
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import train_diversify as td

SEED        = 42
MAX_SITL    = 1500        # cached windows per SITL class
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "diversify_figs"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── choose checkpoint ─────────────────────────────────────────────────────────
def _select_checkpoint():
    """CLI arg, else interactive picker over checkpoints in models/."""
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    pts = sorted(MODEL_DIR.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(
            f"No .pt checkpoint found in {MODEL_DIR}. Pass a model path explicitly."
        )

    print(f"\nAvailable checkpoints ({len(pts)}):")
    for i, p in enumerate(pts):
        print(f"  [{i}] {p.relative_to(PROJECT_ROOT)}")
    default_idx = len(pts) - 1
    while True:
        raw = input(f"Select checkpoint [0-{len(pts)-1}, default={default_idx}]: ").strip()
        if raw == "":
            return pts[default_idx]
        try:
            idx = int(raw)
            if 0 <= idx < len(pts):
                return pts[idx]
        except ValueError:
            pass
        print("Invalid selection. Try again.")


MODEL_PATH = _select_checkpoint()
_ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUTPUT_DIR / f"domain_split_{MODEL_PATH.stem}_{_ts}.png"
print(f"Model:  {MODEL_PATH}")
print(f"Output: {OUT_PATH}")

# ── 1. Load the same cached turn segments used by train_diversify.py ──────────
print("\nLoading model metadata and cached turn segments …")
sd = torch.load(MODEL_PATH, map_location=td.DEVICE)
model_n_feat = int(sd["featurizer.block1.0.weight"].shape[1])
td.N_FEAT = model_n_feat
td.LATENT_DOMAIN_N = int(sd["dclassifier.fc.weight"].shape[0])

sitl_cache = td.config.CACHE_DIR / f"{td.config.SITL_FOLDER}_features.npz"
real_caches = [
    td.config.CACHE_DIR / f"{folder}_features.npz"
    for folder in td.config.REAL_FLIGHT_FOLDERS
]
cache_paths = [sitl_cache, *real_caches]
missing = [path for path in cache_paths if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Required feature cache(s) not found:\n  "
        + "\n  ".join(str(path) for path in missing)
        + "\nRun `python3 tools/preprocessing/build_features.py` first."
    )

for cache_path in cache_paths:
    with np.load(cache_path) as cached:
        if "X_seq" not in cached:
            raise KeyError(f"{cache_path} does not contain X_seq")
        cache_n_feat = int(cached["X_seq"].shape[2])
    if cache_n_feat != model_n_feat:
        raise ValueError(
            f"{cache_path.name} has {cache_n_feat} features, but "
            f"{MODEL_PATH.name} expects {model_n_feat}."
        )


def _sample_class(dataset, label, max_n=None):
    """Return cached windows for one class using DIVERSIFY's label convention."""
    indices = np.flatnonzero(dataset.labels == label)
    if max_n is not None and len(indices) > max_n:
        indices = np.random.choice(indices, max_n, replace=False)
    return [dataset.x[i] for i in indices], [label] * len(indices)


sitl_ds = td.load_cached_windows(sitl_cache)
px4_x, px4_y = _sample_class(sitl_ds, label=1, max_n=MAX_SITL)
ardu_x, ardu_y = _sample_class(sitl_ds, label=0, max_n=MAX_SITL)
print(f"  SITL cache: {sitl_cache.name}")
print(f"    PX4: {len(px4_x)}  Ardu: {len(ardu_x)}")

real_x, real_y = [], []
for cache_path in real_caches:
    dataset = td.load_cached_windows(cache_path)
    real_x.extend(dataset.x)
    real_y.extend(dataset.labels.tolist())
    print(f"  Real cache: {cache_path.name} — windows: {len(dataset)}")

print(
    f"    Real PX4: {sum(y == 1 for y in real_y)}  "
    f"Real Ardu: {sum(y == 0 for y in real_y)}"
)

# ── 3. Source-domain labels (4 groups) ───────────────────────────────────────
# 0=SITL-PX4  1=SITL-Ardu  2=Real-PX4  3=Real-Ardu
all_x = px4_x + ardu_x + real_x
all_class = np.array(px4_y + ardu_y + real_y, dtype=np.int64)
all_src = np.array(
    [0]*len(px4_x) + [1]*len(ardu_x) +
    [2 if y==1 else 3 for y in real_y],
    dtype=np.int64
)
is_real = np.array([False]*len(px4_x) + [False]*len(ardu_x) + [True]*len(real_x))

N = len(all_x)
print(f"Total windows: {N}  (SITL:{len(px4_x)+len(ardu_x)}  Real:{len(real_x)})")
if N < 2:
    raise RuntimeError(
        f"Only {N} usable cached window(s) were created. "
        "Check X_seq, WIN_LEN, MIN_WIN, and cache labels."
    )

tsne_perplexity = min(30, N - 1)

# ── 4. (a) Cached features: mean+std over time → 2*N_FEAT dimensions ──────────
print("\nComputing raw feat statistics (for panel a) …")
raw_feats = np.zeros((N, td.N_FEAT * 2), dtype=np.float32)
for i, w in enumerate(all_x):
    arr = w.numpy()                  # (N_FEAT, WIN_LEN)
    raw_feats[i, :td.N_FEAT]  = arr.mean(axis=1)
    raw_feats[i, td.N_FEAT:]  = arr.std(axis=1)

print("Running t-SNE on raw features …")
tsne_a = TSNE(n_components=2, perplexity=tsne_perplexity, max_iter=1000,
              random_state=SEED, n_jobs=-1)
emb_a  = tsne_a.fit_transform(raw_feats)
print("  done.")

# ── 5. Load trained model ─────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_PATH}")
print(
    f"  N_FEAT={td.N_FEAT}, LATENT_DOMAIN_N={td.LATENT_DOMAIN_N} "
    "(from checkpoint)"
)
model = td.DiversifyFlight().to(td.DEVICE)
model.load_state_dict(sd)
model.eval()

# ── 6. (b) Bottleneck features + pseudo-domain labels & Predictions ──────────
print("Extracting bottleneck features & predictions …")
BSIZE = 256
all_tens = torch.stack(all_x)       # (N, N_FEAT, WIN_LEN)
btn_feats = np.zeros((N, td.BOTTLENECK_DIM), dtype=np.float32)
preds = np.zeros(N, dtype=np.int64) # 모델의 최종 예측 클래스를 담을 배열 추가

with torch.no_grad():
    for s in range(0, N, BSIZE):
        xb = all_tens[s:s+BSIZE].to(td.DEVICE)
        z = model.bottleneck(model.featurizer(xb))
        btn_feats[s:s+len(xb)] = z.cpu().numpy()
        
        # [추가됨] 분류기(Classifier) 통과하여 예측 클래스 확률 계산
        logits = model.classifier(z)
        preds[s:s+len(xb)] = logits.argmax(1).cpu().numpy()

# pseudo-domain via d-branch
print("Running cosine k-means for pseudo-domain labels …")
btn_tens = torch.tensor(btn_feats)
# d-branch features
d_feats = np.zeros((N, td.BOTTLENECK_DIM), dtype=np.float32)
with torch.no_grad():
    for s in range(0, N, BSIZE):
        xb = all_tens[s:s+BSIZE].to(td.DEVICE)
        d_feats[s:s+len(xb)] = model.dbottleneck(model.featurizer(xb)).cpu().numpy()

d_out = np.zeros((N, td.LATENT_DOMAIN_N), dtype=np.float32)
with torch.no_grad():
    for s in range(0, N, BSIZE):
        xb = all_tens[s:s+BSIZE].to(td.DEVICE)
        z  = model.dbottleneck(model.featurizer(xb))
        d_out[s:s+len(xb)] = model.dclassifier(z).cpu().numpy()

# cosine k-means (same as set_dlabel)
from scipy.spatial.distance import cdist
all_fea = np.hstack([d_feats, np.ones((N, 1))])
norms   = np.linalg.norm(all_fea, axis=1, keepdims=True)
all_fea = all_fea / (norms + 1e-8)
aff     = torch.softmax(torch.tensor(d_out), dim=1).numpy()
K       = aff.shape[1]
initc   = aff.T @ all_fea / (1e-8 + aff.sum(0)[:, None])
pred    = cdist(all_fea, initc, "cosine").argmin(1)
aff2    = np.eye(K)[pred]
initc   = aff2.T @ all_fea / (1e-8 + aff2.sum(0)[:, None])
pseudo_dom = cdist(all_fea, initc, "cosine").argmin(1)
print(f"  Pseudo-domain dist: {dict(Counter(pseudo_dom.tolist()))}")

print("\nRunning t-SNE on bottleneck features …")
tsne_b = TSNE(n_components=2, perplexity=tsne_perplexity, max_iter=1000,
              random_state=SEED, n_jobs=-1)
emb_b  = tsne_b.fit_transform(btn_feats)
print("  done.")

# ── 7. Plot ───────────────────────────────────────────────────────────────────
# [수정됨] 1x2 -> 1x3 배열로 확장하고 가로 길이를 늘림
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.patch.set_facecolor("white")

SRC_COLORS = ["#2166ac", "#f4a582", "#1a9641", "#d73027"]
SRC_LABELS = ["SITL PX4", "SITL ArduPilot", "Real PX4", "Real ArduPilot"]
DOM_COLORS = plt.cm.tab10(np.linspace(0, 0.5, K))

S_SMALL  = 8    # SITL point size
S_REAL   = 120  # Real flight star size
ALPHA_BG = 0.35

CLASS_MARKERS = {1: "o", 0: "^"}   # PX4=circle, Ardu=triangle
cls_handles = [
    plt.scatter([], [], c="gray", s=30, marker="o", label="True PX4 ●"),
    plt.scatter([], [], c="gray", s=30, marker="^", label="True ArduPilot ▲"),
    plt.scatter([], [], c="gray", s=80, marker="*",
                edgecolors="black", linewidths=0.5, label="Real flight ★"),
]

# ─── Panel (a): Initial domain split ───
ax = axes[0]
ax.set_facecolor("#f8f8f8")
for grp in range(4):
    mask = (all_src == grp) & ~is_real
    ax.scatter(emb_a[mask, 0], emb_a[mask, 1],
               c=SRC_COLORS[grp], s=S_SMALL, alpha=ALPHA_BG,
               linewidths=0, rasterized=True)

for grp, (src_grp, label_idx) in enumerate([(2, 1), (3, 0)]):
    mask = (all_src == src_grp) & is_real
    if mask.sum() == 0:
        continue
    ax.scatter(emb_a[mask, 0], emb_a[mask, 1],
               c=SRC_COLORS[src_grp], s=S_REAL, marker="*",
               edgecolors="black", linewidths=0.5, zorder=5)

patches = [mpatches.Patch(color=SRC_COLORS[i], label=SRC_LABELS[i]) for i in range(4)]
star_h  = plt.scatter([], [], c="gray", s=80, marker="*", edgecolors="black", linewidths=0.5, label="Real flight ★")
ax.legend(handles=patches + [star_h], fontsize=8, loc="upper right", framealpha=0.9)
ax.set_title("(a) Initial domain split", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# ─── Panel (b): Our domain split (Pseudo-domains) ───
ax = axes[1]
ax.set_facecolor("#f8f8f8")

for d in range(K):
    for cls in [0, 1]:
        mask = (pseudo_dom == d) & (all_class == cls) & ~is_real
        if mask.sum() == 0:
            continue
        ax.scatter(emb_b[mask, 0], emb_b[mask, 1],
                   c=[DOM_COLORS[d]], s=S_SMALL, marker=CLASS_MARKERS[cls],
                   alpha=ALPHA_BG, linewidths=0, rasterized=True)

for i in np.where(is_real)[0]:
    d   = pseudo_dom[i]
    ax.scatter(emb_b[i, 0], emb_b[i, 1],
               c=[DOM_COLORS[d]], s=S_REAL, marker="*",
               edgecolors="black", linewidths=0.8, zorder=5)

dom_patches = [mpatches.Patch(color=DOM_COLORS[d], label=f"Pseudo-domain {d}") for d in range(K)]
ax.legend(handles=dom_patches + cls_handles, fontsize=8, loc="upper right", framealpha=0.9)
ax.set_title("(b) Our domain split (DIVERSIFY)", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# ─── Panel (c): True Classes ───
ax = axes[2]
ax.set_facecolor("#f8f8f8")

# 실제 정답(true class)용 색상 (Blue: ArduPilot, Red: PX4)
CLASS_COLORS = {0: "#2166ac", 1: "#d73027"}

for cls in [0, 1]:
    # 실제 정답(cls)으로 색상 + 마커 모양 결정
    mask = (all_class == cls) & ~is_real
    if mask.sum() == 0:
        continue
    ax.scatter(emb_b[mask, 0], emb_b[mask, 1],
               c=[CLASS_COLORS[cls]], s=S_SMALL, marker=CLASS_MARKERS[cls],
               alpha=ALPHA_BG, linewidths=0, rasterized=True)

for i in np.where(is_real)[0]:
    t_cls = all_class[i] # 실제 정답(true label)으로 색상 결정
    ax.scatter(emb_b[i, 0], emb_b[i, 1],
               c=[CLASS_COLORS[t_cls]], s=S_REAL, marker="*",
               edgecolors="black", linewidths=0.8, zorder=5)

class_patches = [
    mpatches.Patch(color=CLASS_COLORS[0], label="True: ArduPilot (Blue)"),
    mpatches.Patch(color=CLASS_COLORS[1], label="True: PX4 (Red)")
]
ax.legend(handles=class_patches + cls_handles, fontsize=8, loc="upper right", framealpha=0.9)
ax.set_title("(c) True Classes", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

plt.suptitle("DIVERSIFY Latent Space Analysis", fontsize=16, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_PATH}")
