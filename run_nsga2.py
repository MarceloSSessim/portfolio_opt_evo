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
    seed = int(time.time() * 1e6) ^ (os.getpid() << 16)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed & 0xFFFFFFFF)

def run_single_iteration(k, df_etf, df_rf, window_dir):
    """Executa uma iteração do NSGA-II em subprocesso."""
    import traceback
    reseed()

    from nsga2_v1 import run_nsga2_once  # import local é mais seguro

    iter_dir = window_dir / f"iter_{k:03d}"
    start_it = time.perf_counter()      # ★ tempo (inicio da iteração)
    try:
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"    → Iteração {k:03d}  (PID {os.getpid()})")
        run_nsga2_once(df_etf=df_etf, df_rf=df_rf, iteration_dir=iter_dir)
        elapsed = time.perf_counter() - start_it                   # ★ tempo
        return k, elapsed, None                                    # sucesso
    except Exception as e:
        elapsed = time.perf_counter() - start_it                   # ★ tempo
        err_file = iter_dir / "error.txt"
        err_file.write_text(traceback.format_exc())
        return k, elapsed, e                                       # erro

# ------------------------------------------------------------------
# versão paralela da função principal
# ------------------------------------------------------------------
def proxima_faixa_iters(window_dir: Path, n_novos: int):
    """
    Examina window_dir/iter_* e devolve um range começando do próximo índice.

    Ex.: se já existem iter_001 … iter_050 e n_novos=10  ➜  range(51, 61)
    """
    existentes = [
        int(p.name.split("_")[1])
        for p in window_dir.glob("iter_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()
    ]
    ultimo = max(existentes, default=0)
    return range(ultimo + 1, ultimo + n_novos + 1)

def batch_run(folder_etf, folder_rf, output_path,
              n_files_window=36, step_files=2, n_iter=100):

    folder_etf  = Path(folder_etf)
    folder_rf   = Path(folder_rf)
    output_path = Path(output_path)

    all_etf_files = sorted(folder_etf.glob("weekly_log_returns_*.csv"))
    last_start    = len(all_etf_files) - n_files_window

    total_files = len(all_etf_files)
    limite = total_files - (n_files_window + 2) + 1  # critério de parada

    if limite <= 0:
        print(f"Não há janelas suficientes para iniciar (precisaria de pelo menos {n_files_window + 2} arquivos).")
        return

    for start in range(0, limite, step_files):
        etf_files = all_etf_files[start: start + n_files_window]
        date_tag  = etf_files[0].stem.replace("weekly_log_returns_", "")
        window_dir = output_path / date_tag
        window_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Janela {date_tag}  ({n_files_window} arquivos)")
        block_start = time.perf_counter()                        #tempo (inicio bloco)

        # ----- lê arquivos somente uma vez -----
        df_etf = add_time_columns(
            pd.concat(map(pd.read_csv, etf_files), ignore_index=True)
        )
        rf_files = [folder_rf / ("rf_" + f.name) for f in etf_files]
        df_rf = add_time_columns(
            pd.concat(map(pd.read_csv, rf_files), ignore_index=True)
              .rename(columns={"date": "week_start"}, errors="ignore")
        )

        # ----- executa em paralelo -----
        n_cores = os.cpu_count()

        # 1) quais índices ainda faltam?
        novos_idxs = proxima_faixa_iters(window_dir, n_iter)
        if not novos_idxs:
            print("Todos os índices já existem — nada a fazer.")
            continue

        with ProcessPoolExecutor(max_workers=n_cores - 2) as executor:
            futures = [
                executor.submit(run_single_iteration, k, df_etf, df_rf, window_dir)
                for k in novos_idxs
            ]
            for fut in as_completed(futures):
                k, t_it, err = fut.result()
                if err is not None:
                    print(f"⚠️  Iteração {k:03d} falhou em {t_it:.1f}s → {err}")
                else:
                    print(f"✅ Iteração {k:03d} concluída em {t_it:.1f}s")


        block_elapsed = time.perf_counter() - block_start        # tempo
        print(f"⏱️  Janela {date_tag} finalizada em {block_elapsed/60:.1f} min")

# ---------------------------------------------------------
# 3) Chamada
# ---------------------------------------------------------
if __name__ == "__main__":
    batch_run(
        folder_etf  = "/mnt/c/Users/msses/Desktop/ETF/weekly_log_returns",
        folder_rf   = "/mnt/c/Users/msses/Desktop/ETF/DTB1",
        output_path = "/mnt/c/Users/msses/Desktop/ETF/results_NSGA2",
        n_files_window = 36,   # 96 semanas
        step_files     = 2,    # 8 semanas
        n_iter         = 100
    )