# baseline.py  -----------------------------------------------------------
import os, time, random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import cvxpy as cp  

# -----------------------------------------------------------------------
# 1. Helpers
# -----------------------------------------------------------------------
def pivot_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte long→wide: linhas = datas, colunas = tickers (retornos)."""
    return (
        df.pivot(index="week_start", columns="ticker", values="log_return")
          .sort_index()
    )

def equally_weighted(n: int) -> np.ndarray:
    """w_i = 1 / n."""
    return np.repeat(1.0 / n, n)

def markowitz_long_only(mu: np.ndarray, Sigma: np.ndarray,
                        risk_aversion: float = 1.0) -> np.ndarray:
    """
    Resolve:  max  muᵀw − λ · wᵀΣw      (Markowitz “quadratic utility”)
    s.t.      ∑w = 1,  w ≥ 0.
    """
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]
    cp.Problem(objective, constraints).solve(solver=cp.ECOS)
    return np.asarray(w.value).flatten()

def perf_metrics(r: pd.Series, rf: pd.Series = None) -> Dict[str, float]:
    """
    r: retornos da carteira (Series indexada por data)
    rf: retornos RF; se None, assume 0.
    """
    if rf is None:
        rf = 0.0
    else:
        rf = rf.reindex(r.index).fillna(method="ffill").fillna(0.0)

    excess = r - rf
    mean   = excess.mean()
    std    = excess.std(ddof=1)
    downside = excess[excess < 0].std(ddof=1)

    return {
        "cum_ret":   r.sum(),
        "mean":      mean,
        "std":       std,
        "sharpe":    mean / std if std else np.nan,
        "sortino":   mean / downside if downside else np.nan,
    }

# -----------------------------------------------------------------------
# 2. Loop principal por janela
# -----------------------------------------------------------------------
def run_baselines(folder_etf, folder_rf, output_path,
                  n_files_window=36, step_files=2,
                  risk_aversion=1.0, num_tickers=50):
    """
    Para cada janela de treino (n_files_window arquivos),
    - Seleciona ETFs (K-means)              ➜ tickers_selecionados
    - Calcula pesos EW e Markowitz          ➜ w_EW, w_MV
    - Avalia nas 2 semanas seguintes        ➜ métricas
    - Salva resultados em CSV.
    """
    from utils import (                # suas funções
        add_time_columns,
        filtrar_tickers_completos,
        selecionar_tickers_kmeans,
    )

    folder_etf  = Path(folder_etf)
    folder_rf   = Path(folder_rf)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_etf_files = sorted(folder_etf.glob("weekly_log_returns_*.csv"))
    limite = len(all_etf_files) - (n_files_window + 2) + 1
    if limite <= 0:
        raise ValueError("Não há janelas suficientes para iniciar.")

    records = []     # acumula todas as medições

    for start in range(0, limite, step_files):
        train_files = all_etf_files[start: start + n_files_window]
        test_files  = all_etf_files[start + n_files_window:
                                    start + n_files_window + 2]

        tag_train = Path(train_files[0]).stem.replace("weekly_log_returns_", "")
        print(f"\n>>> Baselines | janela-treino começando em {tag_train}")

        # ---- leitura e pré-processamento ----
        df_etf = add_time_columns(
            pd.concat(map(pd.read_csv, train_files + test_files), ignore_index=True)
        )
        df_rf  = add_time_columns(
            pd.concat(map(pd.read_csv,
                          [(folder_rf / f"rf_{f.name}") for f in train_files + test_files]),
                      ignore_index=True)
              .rename(columns={"date": "week_start"}, errors="ignore")
        )

        # ---- seleciona tickers completos na janela-treino ----
        df_train = df_etf[df_etf["week_start"].isin(
            pd.to_datetime([f.stem.replace("weekly_log_returns_", "") for f in train_files])
        )]
        df_train = filtrar_tickers_completos(df_train)
        df_train = df_train.copy()
        df_train["year"]  = df_train["week_start"].dt.year
        df_train["month"] = df_train["week_start"].dt.month
        df_train["week"]  = df_train["week_start"].dt.isocalendar().week

        df_train, tickers_selecionados = selecionar_tickers_kmeans(
            df_train, num_tickers
        )

        # ---- matrizes de retorno ----
        R_train = pivot_returns(df_train)[tickers_selecionados]
        mu      = R_train.mean().values
        Sigma   = R_train.cov().values

        # ---- pesos baseline ----
        w_EW = equally_weighted(len(tickers_selecionados))
        w_MV = markowitz_long_only(mu, Sigma, risk_aversion)

        # ---- retorna da janela-teste ----
        df_test = df_etf[df_etf["week_start"].isin(
            pd.to_datetime([f.stem.replace("weekly_log_returns_", "") for f in test_files])
        ) & df_etf["ticker"].isin(tickers_selecionados)]
        R_test = pivot_returns(df_test)[tickers_selecionados]
        rf_test = pivot_returns(df_rf)[["rf"]].squeeze() if "rf" in df_rf.columns else None

        # ---- métricas ----
        m_EW = perf_metrics(R_test @ w_EW, rf_test)
        m_MV = perf_metrics(R_test @ w_MV, rf_test)

        for label, metrics in [("EW", m_EW), ("Markowitz", m_MV)]:
            rec = {
                "train_start": train_files[0].stem[-10:],
                "train_end":   train_files[-1].stem[-10:],
                "test_start":  test_files[0].stem[-10:],
                "test_end":    test_files[-1].stem[-10:],
                "baseline":    label,
                **metrics,
            }
            records.append(rec)

    # ---- salva agregado ----
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path / "baseline_performance.csv", index=False)
    print(f"\n✔️  Resultados consolidados em {output_path/'baseline_performance.csv'}")


# -----------------------------------------------------------------------
# 3. Execução independente
# -----------------------------------------------------------------------
if __name__ == "__main__":
    run_baselines(
        folder_etf  = "/mnt/c/Users/msses/Desktop/ETF/weekly_log_returns",
        folder_rf   = "/mnt/c/Users/msses/Desktop/ETF/DTB1",
        output_path = "/mnt/c/Users/msses/Desktop/ETF/results_baselines",
        n_files_window = 36,
        step_files     = 2,
        num_tickers    = 50,     # mesmo K-means usado no NSGA-II
        risk_aversion  = 1.0
        )
