import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def fmt_epoch(e):
    return "not reached" if e is None or e < 0 else str(int(e))


def build_table():
    classical = np.load("results/xor_classical_best.npz")
    quantum = np.load("results/xor_quantum_best_extended.npz")

    rows = []
    rows.append({
        "model": "Classical NN (best grid-search)",
        "param_count": int(classical["param_count"]),
        "epoch_90": int(classical["epoch_90"]),
        "epoch_95": int(classical["epoch_95"]),
        "final_acc": float(classical["final_region_acc"]),
        "final_mse": float(classical["final_region_mse"]),
        "resolution": "exact (every epoch)",
        "status": "done",
    })
    rows.append({
        "model": "Shing Chi's classical-quantum network",
        "param_count": None, "epoch_90": None, "epoch_95": None,
        "final_acc": None, "final_mse": None, "resolution": None,
        "status": "TBD - awaiting Shing Chi's results",
    })
    rows.append({
        "model": "Pure quantum network (our best, 13-param entangled)",
        "param_count": int(quantum["param_count"]),
        "epoch_90": int(quantum["epoch_90"]),
        "epoch_95": int(quantum["epoch_95"]),
        "final_acc": float(quantum["final_region_acc"]),
        "final_mse": float(quantum["final_region_mse"]),
        "resolution": f"±{int(quantum['eval_every'])}-epoch (thinned logging)",
        "status": "done",
    })

    np.savez("results/model_comparison_table.npz", rows=np.array(rows, dtype=object))
    return rows


def print_table(rows):
    print(f"{'Model':<45} {'Params':>7} {'Ep->90%':>10} {'Ep->95%':>10} "
          f"{'FinalAcc':>9} {'FinalMSE':>9}  Notes")
    print("-" * 115)
    for r in rows:
        if r["status"].startswith("TBD"):
            print(f"{r['model']:<45} {'TBD':>7} {'TBD':>10} {'TBD':>10} "
                  f"{'TBD':>9} {'TBD':>9}  {r['status']}")
        else:
            print(f"{r['model']:<45} {r['param_count']:>7} "
                  f"{fmt_epoch(r['epoch_90']):>10} {fmt_epoch(r['epoch_95']):>10} "
                  f"{r['final_acc']*100:>8.1f}% {r['final_mse']:>9.4f}  {r['resolution']}")


def plot_table(rows, fname="figures/model_comparison_table.png"):
    fig, (ax_table, ax_bar) = plt.subplots(1, 2, figsize=(18, 5),
                                            gridspec_kw={'width_ratios': [1.5, 1]})

    col_labels = ["Model", "Params", "Epochs->90%", "Epochs->95%", "Final Acc", "Final MSE"]
    cell_text = []
    for r in rows:
        if r["status"].startswith("TBD"):
            cell_text.append([r["model"], "TBD", "TBD", "TBD", "TBD", "TBD"])
        else:
            cell_text.append([
                r["model"], str(r["param_count"]),
                fmt_epoch(r["epoch_90"]), fmt_epoch(r["epoch_95"]),
                f"{r['final_acc']*100:.1f}%", f"{r['final_mse']:.4f}",
            ])

    ax_table.axis('off')
    tbl = ax_table.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center',
                          colWidths=[0.38, 0.12, 0.14, 0.14, 0.12, 0.12])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2)
    ax_table.set_title("Model comparison: classical NN vs. classical-quantum vs. pure quantum")

    names = [r["model"].split(" (")[0] for r in rows]
    pending = [r["status"].startswith("TBD") for r in rows]
    # pending row gets a full-height hatched placeholder bar (not a 0-height bar)
    # so it visibly reads as "not yet known", not as a measured 0% accuracy
    accs = [100 if p else r["final_acc"] * 100 for r, p in zip(rows, pending)]
    colors = ['lightgrey' if p else 'steelblue' for p in pending]
    bars = ax_bar.bar(names, accs, color=colors)
    for bar, p in zip(bars, pending):
        if p:
            bar.set_hatch('///')
            bar.set_edgecolor('grey')
            bar.set_alpha(0.5)
    ax_bar.axhline(90, color='grey', linestyle='--', linewidth=1)
    ax_bar.axhline(95, color='grey', linestyle=':', linewidth=1)
    ax_bar.set_ylabel('Final region accuracy (%)')
    ax_bar.set_ylim(0, 105)
    ax_bar.set_xticks(range(len(names)))
    ax_bar.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax_bar.set_title('Final accuracy (hatched = pending)')

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved -> {fname}")


if __name__ == "__main__":
    rows = build_table()
    print_table(rows)
    plot_table(rows)
    print("Done.")
