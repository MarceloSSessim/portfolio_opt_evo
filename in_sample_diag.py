# diagnostics_nsga2.py
# ---------------------------------------------------------------
# Analyse hypervolume & crowding-distance logs for ALL windows
# produced by batch_run().  Requires matplotlib + pandas ≥1.5.
#
# Folder structure expected:
# results_NSGA2/
# ├── 2022_01_03/
# │   ├── iter_001/hypervolumes_log.csv
# │   ├── iter_001/crowding_distances_log.csv
# │   ├── ...
# ├── 2022_01_17/
# │   └── ...
# ---------------------------------------------------------------
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ----------------------------------------------------------------
# Helper ----------------------------------------------------------
# ----------------------------------------------------------------
def list_iteration_logs(window_dir: Path) -> List[Tuple[int, Path, Path]]:
    """Return [(iter_id, path_hv, path_cd), ...] for one window."""
    out = []
    for iter_dir in window_dir.glob("iter_*"):
        m = re.match(r"iter_(\d+)", iter_dir.name)
        if not m:
            continue
        hv_path = iter_dir / "hypervolumes_log.csv"
        cd_path = iter_dir / "crowding_distances_log.csv"
        if hv_path.exists() and cd_path.exists():
            out.append((int(m.group(1)), hv_path, cd_path))
    return sorted(out, key=lambda x: x[0])

def load_logs(hv_path: Path, cd_path: Path, iter_id: int,
              window_tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load logs, adding iteration and window identifiers."""
    df_hv = pd.read_csv(hv_path)
    df_cd = pd.read_csv(cd_path)
    df_hv["iteration"] = iter_id
    df_cd["iteration"] = iter_id
    df_hv["window"] = window_tag
    df_cd["window"] = window_tag
    return df_hv, df_cd

# ----------------------------------------------------------------
# Main diagnostics ------------------------------------------------
# ----------------------------------------------------------------
def run_diagnostics(root_results: str, plateau_eps: float = 1e-4, plateau_patience: int = 10,
                    output_root: str = "./in_sample_diag"):
    root = Path(root_results)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    windows = sorted([d for d in root.iterdir() if d.is_dir()])

    if not windows:
        raise RuntimeError("Nenhum subdiretório de janela encontrado.")

    all_hv, all_cd = [], []

    # ---------- loop over windows ----------
    for win_dir in windows:
        window_tag = win_dir.name
        iters = list_iteration_logs(win_dir)
        if not iters:
            print(f"[{window_tag}] ‼️  Nenhum log encontrado; pulando.")
            continue

        # -- load & stack the CSVs --
        df_hv_w, df_cd_w = [], []
        for iter_id, hv_path, cd_path in iters:
            hv, cd = load_logs(hv_path, cd_path, iter_id, window_tag)
            df_hv_w.append(hv)
            df_cd_w.append(cd)
        df_hv_w = pd.concat(df_hv_w, ignore_index=True)
        df_cd_w = pd.concat(df_cd_w, ignore_index=True)

        # -- plot per-window curves --
        plot_window_curves(df_hv_w, window_tag,
                           value_col="hypervolume",
                           ylabel="Hypervolume")
        plot_window_curves(df_cd_w, window_tag,
                           value_col="crowding_distance",
                           ylabel="Crowding distance")

        # -- detect plateau generation for each iteration --
        plateau_info = detect_plateau(df_hv_w, "hypervolume",
                                      eps=plateau_eps,
                                      patience=plateau_patience)
        plateau_path = win_dir / "plateau_generations.csv"
        pd.DataFrame(plateau_info).to_csv(plateau_path, index=False)
        print(f"[{window_tag}] ✔️  Plateau generations → {plateau_path.name}")

        # accumulate for global plots
        all_hv.append(df_hv_w)
        all_cd.append(df_cd_w)

    # ---------- aggregate across windows ----------
    df_hv_all = pd.concat(all_hv, ignore_index=True)
    df_cd_all = pd.concat(all_cd, ignore_index=True)

    plot_global_envelope(df_hv_all, "hypervolume",
                         ylabel="Hypervolume",
                         fname=output_root  / "global_hypervolume_envelope.png")
    plot_global_envelope(df_cd_all, "crowding_distance",
                         ylabel="Crowding distance",
                         fname=output_root  / "global_crowding_envelope.png")

    print("\n🏁  Diagnostics concluídos. Plots salvos no diretório de resultados.")

# ----------------------------------------------------------------
# Plot helpers ----------------------------------------------------
# ----------------------------------------------------------------
def plot_window_curves(df: pd.DataFrame, window_tag: str,
                       value_col: str, ylabel: str):
    plt.figure(figsize=(7, 4))
    for iter_id, g in df.groupby("iteration"):
        plt.plot(g["generation"], g[value_col],
                 alpha=0.35, linewidth=0.8, label=f"iter {iter_id:03d}")
    plt.title(f"{ylabel} – window {window_tag}")
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = df["window"].iloc[0]  # just to access folder name
    fig_dir = output_root / "plots_per_window" / out_path
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{value_col}_{window_tag}.png", dpi=300)
    plt.close()

def plot_global_envelope(df: pd.DataFrame, value_col: str,
                         ylabel: str, fname: Path):
    """Plot mean ±1 sd envelope across windows & iterations."""
    # group by global generation index (assume same #gens per run)
    grp = df.groupby("generation")[value_col]
    mean = grp.mean()
    sd   = grp.std()

    plt.figure(figsize=(7, 4))
    plt.plot(mean.index, mean.values, linewidth=1.8, label="mean")
    plt.fill_between(mean.index,
                     mean - sd,
                     mean + sd,
                     alpha=0.25, label="±1 sd")
    plt.title(f"{ylabel} – mean ±1 sd across all windows")
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"🌐  Global envelope plot salvo → {fname.name}")

# ----------------------------------------------------------------
# Plateau detection ----------------------------------------------
# ----------------------------------------------------------------
def detect_plateau(df: pd.DataFrame, value_col: str,
                   eps: float, patience: int) -> List[Dict]:
    """Return list of dicts with iteration, plateau_generation (or NaN)."""
    out = []
    for (iter_id), g in df.groupby("iteration"):
        g = g.sort_values("generation")
        diffs = g[value_col].diff().abs()
        rolling_ok = diffs < eps
        plateau_gen = np.nan
        count = 0
        # sliding window to detect 'patience' consecutive small diffs
        for gen, ok in zip(g["generation"], rolling_ok):
            count = count + 1 if ok else 0
            if count >= patience:
                plateau_gen = gen - patience + 1
                break
        out.append({"iteration": iter_id,
                    "plateau_generation": plateau_gen})
    return out

# ----------------------------------------------------------------
# Entry point -----------------------------------------------------
# ----------------------------------------------------------------
if __name__ == "__main__":
    run_diagnostics(
        root_results="/mnt/c/Users/msses/Desktop/ETF/results_NSGA2",
        plateau_eps=1e-4,
        plateau_patience=10,
        output_root="/mnt/c/Users/msses/Desktop/ETF/in_sample_diag"
    )
