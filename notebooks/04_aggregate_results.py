"""
Notebook 04 — Aggregate Results
================================
Collects individual benchmark CSVs from all interns, merges them into a
single master comparison table, generates publication-ready visualisations,
produces a LaTeX-ready table string, and runs a basic statistical-
significance check.

Paste this script cell-by-cell into a Google Colab notebook, or run it as a
Python script after all benchmark notebooks (03) have finished.

Steps covered:
  1.  Mount Google Drive + install extra dependencies
  2.  Scan Drive for all ``results_*.csv`` files
  3.  Merge into a master DataFrame
  4.  Compute Efficiency Score  η = mAP@0.5 / (FLOPs / 10⁹)  [alternative]
  5.  Generate publication-ready visualisations
      5a. Energy vs Accuracy Pareto curve
      5b. Parameter comparison bar chart
      5c. FPS comparison bar chart
      5d. mAP across datasets (grouped bar)
      5e. Efficiency Score ranking
      5f. Radar chart — top-5 models
  6.  Save all figures as high-res PNGs
  7.  Print and save LaTeX-ready table
  8.  Statistical significance test (paired t-test placeholder)
"""

# ────────────────────────────────────────────────────────────────────────────
# Cell 1: Mount Drive + install dependencies
# ────────────────────────────────────────────────────────────────────────────

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "pandas>=2.0.0", "matplotlib>=3.7.0", "seaborn>=0.12.0",
     "scipy>=1.10.0", "tabulate>=0.9.0"],
    check=False,
)
print("✅ Dependencies installed.")

import os

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    DRIVE_ROOT = "/content/drive/MyDrive/agrokd_project"
    print("✅ Google Drive mounted.")
except ImportError:
    DRIVE_ROOT = "/tmp/agrokd_project"
    print("ℹ️  Not in Colab — using local /tmp directory.")

RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")
FIGS_DIR    = os.path.join(RESULTS_DIR, "figures", "aggregated")
os.makedirs(FIGS_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Cell 2: Scan Drive for all result CSVs
# ────────────────────────────────────────────────────────────────────────────

import glob as _glob

csv_files = _glob.glob(os.path.join(RESULTS_DIR, "results_*.csv"))
csv_files += _glob.glob(os.path.join(RESULTS_DIR, "benchmark_results.csv"))
csv_files = sorted(set(csv_files))

print(f"📂 Found {len(csv_files)} result file(s):")
for f in csv_files:
    print(f"   {f}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 3: Merge into master DataFrame
# ────────────────────────────────────────────────────────────────────────────

import math

import pandas as pd

if csv_files:
    frames = []
    for f in csv_files:
        try:
            df_tmp = pd.read_csv(f)
            frames.append(df_tmp)
        except Exception as e:
            print(f"  ⚠️  Could not read {f}: {e}")

    if frames:
        master_df = pd.concat(frames, ignore_index=True)
        # Drop duplicate rows (same Model + Dataset)
        master_df.drop_duplicates(subset=["Model", "Dataset"], keep="last", inplace=True)
        master_df.reset_index(drop=True, inplace=True)
    else:
        master_df = pd.DataFrame()
else:
    master_df = pd.DataFrame()

if master_df.empty:
    print("⚠️  No result CSVs found — generating a synthetic dataset for demonstration.")
    import numpy as np
    rng = np.random.default_rng(42)
    models_list = [
        "yolov8n", "yolov8s", "yolov8m", "yolov9t", "yolov9s",
        "yolov10n", "yolov10s", "yolov11n", "rt-detr-l", "yolo-world",
    ]
    datasets_list = ["MH-Weed16", "CottonWeed", "RiceWeed"]
    rows = []
    for m in models_list:
        for d in datasets_list:
            rows.append({
                "Model": m,
                "Dataset": d,
                "Params(M)": float(rng.uniform(1, 70)),
                "GFLOPs":    float(rng.uniform(2, 200)),
                "mAP50":     float(rng.uniform(0.55, 0.87)),
                "mAP50-95":  float(rng.uniform(0.35, 0.65)),
                "Precision": float(rng.uniform(0.60, 0.90)),
                "Recall":    float(rng.uniform(0.55, 0.88)),
                "F1":        float(rng.uniform(0.58, 0.86)),
                "FPS":       float(rng.uniform(20, 300)),
                "Energy(J/frame)": float(rng.uniform(0.001, 0.012)),
                "Efficiency_Score": float(rng.uniform(50, 400)),
            })
    master_df = pd.DataFrame(rows)
else:
    print(f"✅ Master table: {len(master_df)} rows, {len(master_df.columns)} columns")

# Ensure numeric types where needed
NUM_COLS = ["Params(M)", "GFLOPs", "mAP50", "mAP50-95", "FPS", "Energy(J/frame)", "Efficiency_Score"]
for col in NUM_COLS:
    if col in master_df.columns:
        master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

print(master_df.to_string(index=False))


# ────────────────────────────────────────────────────────────────────────────
# Cell 4: Compute / refresh Efficiency Score
# ────────────────────────────────────────────────────────────────────────────

# η = mAP@0.5 / energy_per_frame_J  (from measurement)
# Alternative: η = mAP@0.5 / (GFLOPs / 1e9)  when energy is unavailable

if "Energy(J/frame)" in master_df.columns and master_df["Energy(J/frame)"].notna().any():
    master_df["Efficiency_Score"] = master_df["mAP50"] / master_df["Energy(J/frame)"].replace(0, float("nan"))
else:
    master_df["Efficiency_Score"] = master_df["mAP50"] / (master_df["GFLOPs"].replace(0, float("nan")) / 1e9)

print("✅ Efficiency scores recomputed.")


# ────────────────────────────────────────────────────────────────────────────
# Cell 5: Generate publication-ready visualisations
# ────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

models_order = master_df["Model"].unique().tolist()


def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(FIGS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved: {name}")
    return path


# ── 5a. Pareto curve ─────────────────────────────────────────────────────────

def _pareto_front(xs, ys):
    n = len(xs)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i != j and xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


# Average across datasets per model
_ENERGY_COL = "Energy(J/frame)"
avg_df = master_df.groupby("Model")[[_ENERGY_COL, "mAP50", "GFLOPs"]].mean().reset_index()
x_col = _ENERGY_COL if avg_df[_ENERGY_COL].notna().any() else "GFLOPs"
x_label = "Energy per frame (J)" if x_col == _ENERGY_COL else "GFLOPs"

fig, ax = plt.subplots(figsize=(8, 5))
xs = avg_df[x_col].tolist()
ys = avg_df["mAP50"].tolist()
labels = avg_df["Model"].tolist()

ax.scatter(xs, ys, s=60, color="steelblue", zorder=3, label="Models")
for xi, yi, lbl in zip(xs, ys, labels):
    ax.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)

valid = [i for i in range(len(xs)) if not math.isnan(xs[i]) and not math.isnan(ys[i])]
if len(valid) >= 2:
    vx = [xs[i] for i in valid]
    vy = [ys[i] for i in valid]
    pi = _pareto_front(vx, vy)
    pairs = sorted(zip([vx[i] for i in pi], [vy[i] for i in pi]))
    ax.plot([p[0] for p in pairs], [p[1] for p in pairs], "r--", linewidth=1.5, label="Pareto front")
    ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], color="red", zorder=4, s=80)

ax.set_xlabel(x_label)
ax.set_ylabel("mAP@0.5 (avg across datasets)")
ax.set_title("Energy vs Accuracy — Pareto Curve")
ax.legend()
plt.tight_layout()
_save(fig, "pareto_energy_vs_map.png")

# ── 5b. Parameter comparison ─────────────────────────────────────────────────

param_df = master_df.groupby("Model")["Params(M)"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(param_df)))  # type: ignore[attr-defined]
bars = ax.bar(param_df["Model"], param_df["Params(M)"], color=colors)
ax.set_xlabel("Model")
ax.set_ylabel("Parameters (M)")
ax.set_title("Model Parameter Comparison")
ax.tick_params(axis="x", rotation=45)
for bar, val in zip(bars, param_df["Params(M)"]):
    if not math.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
_save(fig, "params_comparison.png")

# ── 5c. FPS comparison ───────────────────────────────────────────────────────

fps_df = master_df.groupby("Model")["FPS"].mean().reset_index().sort_values("FPS", ascending=False)
fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(fps_df)))  # type: ignore[attr-defined]
bars = ax.bar(fps_df["Model"], fps_df["FPS"], color=colors)
ax.set_xlabel("Model")
ax.set_ylabel("FPS")
ax.set_title("Inference Speed Comparison")
ax.tick_params(axis="x", rotation=45)
for bar, val in zip(bars, fps_df["FPS"]):
    if not math.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
_save(fig, "fps_comparison.png")

# ── 5d. mAP across datasets ──────────────────────────────────────────────────

ds_list = sorted(master_df["Dataset"].unique().tolist())
pivot = master_df.pivot_table(index="Model", columns="Dataset", values="mAP50", aggfunc="mean")
pivot = pivot.reindex(index=models_order)

fig, ax = plt.subplots(figsize=(13, 5))
n_ds = len(ds_list)
x = np.arange(len(pivot))
width = 0.8 / n_ds
ds_colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
for di, ds_name in enumerate(ds_list):
    vals = pivot.get(ds_name, pd.Series([float("nan")] * len(pivot))).fillna(0).tolist()
    offset = (di - n_ds / 2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=ds_name, color=ds_colors[di % len(ds_colors)], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(pivot.index.tolist(), rotation=45, ha="right")
ax.set_xlabel("Model")
ax.set_ylabel("mAP@0.5")
ax.set_title("mAP@0.5 Across Datasets")
ax.legend()
plt.tight_layout()
_save(fig, "map_across_datasets.png")

# ── 5e. Efficiency Score ranking ─────────────────────────────────────────────

eff_df = master_df.groupby("Model")["Efficiency_Score"].mean().reset_index()
eff_df = eff_df.sort_values("Efficiency_Score", ascending=True)

fig, ax = plt.subplots(figsize=(9, 4))
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(eff_df)))  # type: ignore[attr-defined]
ax.barh(eff_df["Model"], eff_df["Efficiency_Score"].fillna(0), color=colors)
ax.set_xlabel("Efficiency Score (mAP50 / J·frame⁻¹)")
ax.set_title("Model Efficiency Score Ranking")
plt.tight_layout()
_save(fig, "efficiency_ranking.png")

# ── 5f. Radar chart — top-5 models ───────────────────────────────────────────

radar_metrics = ["mAP50", "mAP50-95", "FPS", "Efficiency_Score"]
radar_labels  = ["mAP50", "mAP50-95", "FPS", "Efficiency"]

top5_models = (
    master_df.groupby("Model")["mAP50"].mean()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

top5_df = master_df[master_df["Model"].isin(top5_models)].groupby("Model")[radar_metrics].mean()

# Normalise 0–1 per metric
top5_norm = top5_df.copy()
for col in radar_metrics:
    mn = top5_df[col].min()
    mx = top5_df[col].max()
    if mx > mn:
        top5_norm[col] = (top5_df[col] - mn) / (mx - mn)
    else:
        top5_norm[col] = 0.5

n_r = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, n_r, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
r_colors = plt.cm.tab10(np.linspace(0, 1, 5))  # type: ignore[attr-defined]
for (model_nm, row), color in zip(top5_norm.iterrows(), r_colors):
    values = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, values, "o-", linewidth=1.5, label=model_nm, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("Top-5 Models — Multi-metric Radar", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
_save(fig, "radar_top5.png")

print(f"\n✅ All 6 figures saved to: {FIGS_DIR}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 6: Save master CSV
# ────────────────────────────────────────────────────────────────────────────

master_csv = os.path.join(RESULTS_DIR, "master_results.csv")
master_df.to_csv(master_csv, index=False)
print(f"💾 Master CSV saved to: {master_csv}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 7: LaTeX-ready table
# ────────────────────────────────────────────────────────────────────────────

latex_cols = ["Model", "Dataset", "Params(M)", "GFLOPs", "mAP50", "mAP50-95", "FPS", "Efficiency_Score"]
latex_df = master_df[[c for c in latex_cols if c in master_df.columns]].copy()

# Format floats
for col in ["Params(M)", "GFLOPs", "mAP50", "mAP50-95", "FPS", "Efficiency_Score"]:
    if col in latex_df.columns:
        latex_df[col] = latex_df[col].apply(
            lambda v: f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else str(v)
        )

latex_str = latex_df.to_latex(index=False, escape=True)
latex_path = os.path.join(RESULTS_DIR, "results_table.tex")
with open(latex_path, "w") as fh:
    fh.write(latex_str)

print("📄 LaTeX table preview (first 10 rows):")
print(master_df[latex_cols[:6]].head(10).to_string(index=False))
print(f"\n📄 Full LaTeX table saved to: {latex_path}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 8: Statistical significance test (paired t-test)
# ────────────────────────────────────────────────────────────────────────────

try:
    from scipy import stats  # type: ignore

    if "mAP50" in master_df.columns and len(top5_models) >= 2:
        m1, m2 = top5_models[0], top5_models[1]
        vals1 = master_df[master_df["Model"] == m1]["mAP50"].dropna().tolist()
        vals2 = master_df[master_df["Model"] == m2]["mAP50"].dropna().tolist()
        min_len = min(len(vals1), len(vals2))
        if min_len >= 2:
            t_stat, p_val = stats.ttest_rel(vals1[:min_len], vals2[:min_len])
            print(f"\n📊 Paired t-test: {m1} vs {m2}")
            print(f"   t-statistic: {t_stat:.4f}")
            print(f"   p-value    : {p_val:.4f}")
            print(f"   {'✅ Significant (p < 0.05)' if p_val < 0.05 else 'ℹ️  Not significant (p ≥ 0.05)'}")
        else:
            print("ℹ️  Not enough data points for paired t-test (need ≥ 2 matched pairs).")
    else:
        print("ℹ️  Skipping t-test — insufficient data.")
except ImportError:
    print("ℹ️  scipy not available — skipping statistical test.")

print("\n🎉 Notebook 04 (Aggregate Results) complete.")
