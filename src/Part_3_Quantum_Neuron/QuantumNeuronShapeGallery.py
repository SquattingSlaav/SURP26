import numpy as np
import matplotlib.pyplot as plt
import os

from QuantumNeuronMCMC import (
    phases20, p_ref, update_neuron_params, evaluate_array, score, SHOTS,
)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

N_SHAPES = 8
HI_SHOTS = 16384
DEDUP_COSINE_THRESHOLD = 0.999


def cosine_sim(a, b):
    a, b = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def dedupe(arrays, params, scores, chain_ids, trials):
    kept_idx = []
    kept_arrays = []
    for i, arr in enumerate(arrays):
        is_dup = any(cosine_sim(arr, kept) > DEDUP_COSINE_THRESHOLD for kept in kept_arrays)
        if not is_dup:
            kept_idx.append(i)
            kept_arrays.append(arr)
    kept_idx = np.array(kept_idx)
    return arrays[kept_idx], params[kept_idx], scores[kept_idx], chain_ids[kept_idx], trials[kept_idx]


def build_templates(n=20):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    templates = {
        'checkerboard': p_ref.astype(float) * 2 - 1,
        'diag_main': (i - j).astype(float),
        'diag_anti': (i + j).astype(float),
        'row_monotone': i.astype(float) * np.ones((n, n)),
        'col_monotone': j.astype(float) * np.ones((n, n)),
        'radial': np.sqrt((i - (n - 1) / 2) ** 2 + (j - (n - 1) / 2) ** 2),
    }
    center = (n - 1) / 2
    corner_centers = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    corner_bumps = [np.exp(-((i - ci) ** 2 + (j - cj) ** 2) / (2 * 5 ** 2))
                     for ci, cj in corner_centers]
    templates['corner'] = np.max(corner_bumps, axis=0)
    return templates


TEMPLATES = build_templates()
FEATURE_NAMES = list(TEMPLATES.keys()) + ['symmetric', 'antisymm180']


def shape_features(array):
    feats = []
    for name, tmpl in TEMPLATES.items():
        feats.append(np.corrcoef(array.ravel(), tmpl.ravel())[0, 1])
    feats.append(-np.linalg.norm(array - array.T))
    feats.append(-np.linalg.norm(array - np.flipud(np.fliplr(array))))
    return np.array(feats)


def compute_all_features(arrays):
    return np.array([shape_features(a) for a in arrays])


def zscore(features):
    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (features - mu) / sigma


def farthest_point_selection(features_z, scores, n_shapes=N_SHAPES):
    n = len(features_z)
    n_shapes = min(n_shapes, n)
    seed_idx = int(np.argmin(scores))  # best-scoring (lowest mismatch) model always included
    selected = [seed_idx]
    used_labels = {label_shape(features_z[seed_idx])}
    min_dist = np.linalg.norm(features_z - features_z[seed_idx], axis=1)

    while len(selected) < n_shapes:
        min_dist[selected] = -np.inf
        # walk candidates farthest-first, take the first one whose dominant
        # label isn't already used -- keeps the presentation's 8 panels
        # labeled as 8 distinct categories, not e.g. two "checkerboard-like"
        order = np.argsort(-min_dist)
        next_idx = None
        for idx in order:
            if idx in selected:
                continue
            if label_shape(features_z[idx]) not in used_labels:
                next_idx = int(idx)
                break
        if next_idx is None:  # fewer unique labels than n_shapes: fall back
            for idx in order:
                if idx not in selected:
                    next_idx = int(idx)
                    break
        selected.append(next_idx)
        used_labels.add(label_shape(features_z[next_idx]))
        dist_to_new = np.linalg.norm(features_z - features_z[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist_to_new)
        min_dist[selected] = -np.inf
    return selected


def label_shape(features_z_row):
    idx = int(np.argmax(np.abs(features_z_row)))
    if abs(features_z_row[idx]) < 1.0:
        return "mixed/other"
    name = FEATURE_NAMES[idx]
    labels = {
        'checkerboard': 'checkerboard-like', 'diag_main': 'diagonal-separator-like',
        'diag_anti': 'anti-diagonal-separator-like', 'row_monotone': 'row-monotone',
        'col_monotone': 'column-monotone', 'radial': 'radial', 'corner': 'corner-dominant',
        'symmetric': 'symmetric', 'antisymm180': '180-antisymmetric',
    }
    return labels.get(name, name)


def build_gallery(snapshots_path="results/mh_snapshots.npz"):
    d = np.load(snapshots_path)
    arrays, params, scores, chain_ids, trials = (
        d["snapshot_arrays"], d["snapshot_params"], d["snapshot_scores"],
        d["snapshot_chain_id"], d["snapshot_trial"],
    )
    print(f"Loaded {len(scores)} raw snapshots")

    arrays, params, scores, chain_ids, trials = dedupe(arrays, params, scores, chain_ids, trials)
    print(f"{len(scores)} unique snapshots after dedup (cosine sim > "
          f"{DEDUP_COSINE_THRESHOLD} dropped)")

    features = compute_all_features(arrays)
    features_z = zscore(features)
    selected = farthest_point_selection(features_z, scores, N_SHAPES)

    gallery_arrays = arrays[selected]
    gallery_params = params[selected]
    gallery_scores = scores[selected]
    gallery_features = features_z[selected]
    gallery_labels = np.array([label_shape(features_z[i]) for i in selected])

    np.savez("results/shape_gallery.npz",
             gallery_arrays=gallery_arrays, gallery_params=gallery_params,
             gallery_scores=gallery_scores, gallery_features=gallery_features,
             gallery_labels=gallery_labels)
    print(f"Saved {len(selected)} shapes -> results/shape_gallery.npz")
    for lbl, sc in zip(gallery_labels, gallery_scores):
        print(f"  {lbl:28s} score={sc}")

    return gallery_arrays, gallery_scores, gallery_labels


def plot_gallery(gallery_arrays, gallery_scores, gallery_labels, title_suffix,
                  fname):
    n = len(gallery_arrays)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for k in range(n):
        ax = axes[k]
        im = ax.pcolormesh(phases20, phases20, gallery_arrays[k], shading='auto', cmap='RdBu')
        ax.set_title(f"{gallery_labels[k]}\n({title_suffix}={gallery_scores[k]}/400)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

    for k in range(n, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved -> {fname}")


def shot_measure_gallery(gallery_params, shots=SHOTS, hi_shots=HI_SHOTS,
                          n_shot_repeats=5, n_hi_repeats=3):
    # real sim.run(..., shots=N) measurements for each selected model -- the
    # exact-statevector arrays used to pick/rank the 8 shapes have no
    # measurement noise and are not what a real run would show
    n = len(gallery_params)
    shot_arrays = np.zeros((n, n_shot_repeats, 20, 20))
    shot_scores = np.zeros((n, n_shot_repeats), dtype=int)
    hi_arrays = np.zeros((n, n_hi_repeats, 20, 20))
    hi_scores = np.zeros((n, n_hi_repeats), dtype=int)

    for k, p in enumerate(gallery_params):
        params = [int(v) for v in p]
        update_neuron_params(params)
        for r in range(n_shot_repeats):
            arr, _, _ = evaluate_array(shots=shots)
            shot_arrays[k, r] = arr
            shot_scores[k, r] = score(arr)
        for r in range(n_hi_repeats):
            arr, _, _ = evaluate_array(shots=hi_shots)
            hi_arrays[k, r] = arr
            hi_scores[k, r] = score(arr)
        print(f"  model {k}: {shots}-shot scores={list(shot_scores[k])}  "
              f"{hi_shots}-shot scores={list(hi_scores[k])}", flush=True)

    return shot_arrays, shot_scores, hi_arrays, hi_scores


if __name__ == "__main__":
    gallery_arrays, gallery_scores, gallery_labels = build_gallery()
    plot_gallery(gallery_arrays, gallery_scores, gallery_labels,
                 title_suffix="exact score",
                 fname="figures/shape_gallery_exact.png")
    print("(figures/shape_gallery_exact.png is the idealized exact-statevector "
          "reference -- no measurement noise, NOT a real run.)")

    d = np.load("results/shape_gallery.npz")
    print(f"\nRe-measuring all {len(d['gallery_params'])} selected models with real "
          f"sim.run(shots=...) calls ({SHOTS} shots x5 repeats, {HI_SHOTS} shots x3 repeats)...")
    shot_arrays, shot_scores, hi_arrays, hi_scores = shot_measure_gallery(d["gallery_params"])

    hi_mean_arrays = hi_arrays.mean(axis=1)
    hi_mean_scores = np.round(hi_scores.mean(axis=1)).astype(int)

    np.savez("results/shape_gallery_shotbased.npz",
             gallery_params=d["gallery_params"], gallery_labels=d["gallery_labels"],
             exact_scores=d["gallery_scores"],
             shot_arrays=shot_arrays, shot_scores=shot_scores,
             hi_arrays=hi_arrays, hi_scores=hi_scores,
             hi_mean_arrays=hi_mean_arrays, hi_mean_scores=hi_mean_scores)

    plot_gallery(hi_mean_arrays, hi_mean_scores, d["gallery_labels"],
                 title_suffix=f"mean of 3x{HI_SHOTS}-shot score",
                 fname="figures/shape_gallery_shotbased.png")
    print("figures/shape_gallery_shotbased.png is the REAL measured-shot gallery "
          "(mean of 3 independent 16384-shot runs per model).")
    print("Done.")
