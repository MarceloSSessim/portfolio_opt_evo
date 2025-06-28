import os, time, random, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# ------------------------------------------------------------------
# utilitário: adiciona colunas de tempo
# ------------------------------------------------------------------
def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["year"]  = df["week_start"].dt.isocalendar().year
    df["week"]  = df["week_start"].dt.isocalendar().week
    df["month"] = df["week_start"].dt.month
    return df

# ------------------------------------------------------------------
# subprocesso: roda 1 iteração
# ------------------------------------------------------------------

def reseed():
    # Usa PID + contador de tempo para gerar algo único
    seed = int(time.time() * 1e6) ^ (os.getpid() << 16)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed & 0xFFFFFFFF)

def run_single_iteration(k, df_etf, df_rf, window_dir):
    """Executa uma iteração do NSGA-II em subprocesso."""
    import traceback
    reseed() 
    from nsga2_v1 import run_nsga2_once   # import local é mais seguro

    iter_dir = window_dir / f"iter_{k:03d}"
    try:
        iter_dir.mkdir(parents=True, exist_ok=True)          # ★ Essencial
        print(f"    → Iteração {k:03d}  (PID {os.getpid()})")
        run_nsga2_once(df_etf=df_etf, df_rf=df_rf, iteration_dir=iter_dir)
        return k, None                                       # sucesso
    except Exception as e:
        # Salva traceback para poder inspecionar depois
        err_file = iter_dir / "error.txt"
        err_file.write_text(traceback.format_exc())
        return k, e                                          # erro

# ------------------------------------------------------------------
# versão paralela da função principal
# ------------------------------------------------------------------
def batch_run(folder_etf, folder_rf, output_path,
              n_files_window=12, step_files=2, n_iter=100):

    folder_etf  = Path(folder_etf)
    folder_rf   = Path(folder_rf)
    output_path = Path(output_path)

    all_etf_files = sorted(folder_etf.glob("weekly_log_returns_*.csv"))
    last_start    = len(all_etf_files) - n_files_window

    for start in range(0, last_start + 1, step_files):
        etf_files = all_etf_files[start: start + n_files_window]
        date_tag  = etf_files[0].stem.replace("weekly_log_returns_", "")
        window_dir = output_path / date_tag
        window_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Janela {date_tag}  ({n_files_window} arquivos)")

        # ----- lê arquivos apenas uma vez (pode trocar para lazy load) -----
        df_etf = add_time_columns(
            pd.concat(map(pd.read_csv, etf_files), ignore_index=True)
        )
        rf_files = [folder_rf / ("rf_" + f.name) for f in etf_files]
        df_rf = add_time_columns(
            pd.concat(map(pd.read_csv, rf_files), ignore_index=True)
            .rename(columns={"date": "week_start"}, errors="ignore")
        )

        # ----- executa em paralelo -----
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            futures = [
                executor.submit(run_single_iteration, k, df_etf, df_rf, window_dir)
                for k in range(1, n_iter + 1)
            ]
            for fut in as_completed(futures):
                k, err = fut.result()
                if err is not None:
                    print(f"⚠️  Iteração {k:03d} falhou: {err}")

# ---------------------------------------------------------
# 3) Chamada
# ---------------------------------------------------------
if __name__ == "__main__":
    batch_run(
        folder_etf  = "/mnt/c/Users/msses/Desktop/ETF/weekly_log_returns",
        folder_rf   = "/mnt/c/Users/msses/Desktop/ETF/DTB1",
        output_path = "/mnt/c/Users/msses/Desktop/ETF/results_NSGA2",
        n_files_window = 12,   # 48 semanas
        step_files     = 2,    # 8 semanas
        n_iter         = 100
    )