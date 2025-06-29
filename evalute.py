# ----------------------------------------------
# 0. Imports
# ----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ----------------------------------------------
# 1. Dados do S&P 500 → semanais
# ----------------------------------------------
def preparar_sp500(path_sp500, week_freq="W-MON"):
    """
    Lê o CSV diário do S&P 500 e devolve DataFrame com log-retornos semanais.

    Parameters
    ----------
    path_sp500 : str or Path
        Caminho para o CSV (colunas: Date, Close/Last ...).
    week_freq : str
        Frequência da semana usada pelo pandas (DEFAULT = 'W-MON' → semana
        começando na segunda-feira, alinhada aos seus arquivos weekly_*).

    Returns
    -------
    pd.DataFrame
        Índice = week_start (datetime64[ns]),
        coluna 'log_return' com o retorno da semana.
    """
    df = (
        pd.read_csv(path_sp500)
          .assign(Date=lambda d: pd.to_datetime(d["Date"]))
          .sort_values("Date")
          .assign(
              Close=lambda d: (
                  d["Close/Last"].astype(str)
                                 .str.replace(r"[$,]", "", regex=True)
                                 .astype(float)
              )
          )
    )
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["log_return"])

    # === agrega para frequência semanal ===
    df["week_start"] = df["Date"].dt.to_period(week_freq).dt.start_time
    df_weekly = (
        df.groupby("week_start")["log_return"]
          .sum()                        # soma log-retornos dentro da semana
          .to_frame()
    )
    return df_weekly

# ----------------------------------------------
# 2. Métricas de desempenho
# ----------------------------------------------
def calcular_metricas(rets, rf, freq=52):
    excess = rets - rf
    mu           = excess.mean()
    sigma        = rets.std()
    downside_std = np.sqrt(((np.minimum(0, excess))**2).mean())
    cvar         = -np.mean(np.sort(rets)[int(0.05 * len(rets))])

    sharpe  = (mu * freq) / (sigma        * np.sqrt(freq)) if sigma        > 0 else np.nan
    sortino = (mu * freq) / (downside_std * np.sqrt(freq)) if downside_std > 0 else np.nan

    return {
        "retorno_medio": mu * freq,
        "sharpe":        sharpe,
        "cvar_5%":       cvar,
        "sortino":       sortino,
        "volatilidade":  sigma * np.sqrt(freq),
    }

# ----------------------------------------------
# 3. Avaliação de portfólios + S&P 500
# ----------------------------------------------
def avaliar_portfolios(path_results, path_etf, path_rf,
                       path_sp500, n_window=12, n_future=2):
    """
    Avalia os portfólios gerados pela sua NSGA-II, compara com S&P 500
    e devolve (i) tabela de métricas, (ii) DataFrame longo de retornos.

    Returns
    -------
    df_avaliacoes      : pd.DataFrame – métricas por iteração
    df_retornos_long   : pd.DataFrame – colunas [week_start, retorno, tipo, janela, iteracao]
    """
    path_results = Path(path_results)
    all_etf_files = sorted(Path(path_etf).glob("weekly_log_returns_*.csv"))
    all_rf_files  = sorted(Path(path_rf ).glob("rf_weekly_log_returns_*.csv"))
    sp500_weekly  = preparar_sp500(path_sp500)     # já indexado por week_start

    avaliacoes, registros_retornos = [], []

    # --- percorre janelas -----------------------------------------------
    for i, etf_file in enumerate(all_etf_files[:-n_window - n_future]):
        tag = etf_file.stem.replace("weekly_log_returns_", "")
        window_dir = path_results / tag
        if not window_dir.exists():
            continue

        # arquivos das janelas FUTURAS
        etfs_futuros = all_etf_files[i + n_window : i + n_window + n_future]
        rfs_futuros  = all_rf_files [i + n_window : i + n_window + n_future]
        if len(etfs_futuros) < n_future:
            continue

        # --- consolida ETFs / RF / S&P 500 dessa janela ------------------
        df_etf_futuro = pd.concat(map(pd.read_csv, etfs_futuros)).set_index("week_start")
        df_rf_futuro  = (
            pd.concat(map(pd.read_csv, rfs_futuros))
              .rename(columns={"date": "week_start"})
              .set_index("week_start")
        )
        rf_vec = df_rf_futuro["log_return"].reindex(df_etf_futuro.index).fillna(0).values

        # S&P 500 para as mesmas semanas
        sp500_vec = sp500_weekly.reindex(df_etf_futuro.index)["log_return"].fillna(0).values

        # ----------------- SP500 métricas (1 vez por tag) ----------------
        # (opcional: guardar num dicionário se quiser evitar duplicação)
        met_sp = calcular_metricas(sp500_vec, rf_vec)
        prefix_sp = {f"{k}_sp500": v for k, v in met_sp.items()}

        # ----------------- PORTFÓLIOS por iteração -----------------------
        for iter_file in sorted(window_dir.glob("iter_*/best_portfolio_by_sharpe.csv")):
            try:
                port = pd.read_csv(iter_file, index_col="ticker")["weight"]
                port = port / port.sum()

                subset = df_etf_futuro[df_etf_futuro.columns.intersection(port.index)]
                rets   = subset.dot(port.loc[subset.columns].values)

                # -- métricas desta iteração ------------------------------
                met = calcular_metricas(rets.values, rf_vec[:len(rets)])
                met.update({
                    "janela":   tag,
                    "iteracao": iter_file.parent.name,
                    "n_periodos_testados": len(rets),
                    **prefix_sp      # cola métricas do S&P 500
                })
                avaliacoes.append(met)

                # -- salva retornos semanais no formato LONG --------------
                registros_retornos.extend(
                    [
                        # retornos da iter / portfolio
                        {
                            "week_start": ws,
                            "retorno":   ret,
                            "tipo":      "portfolio",
                            "janela":    tag,
                            "iteracao":  iter_file.parent.name,
                        }
                        for ws, ret in rets.items()
                    ]
                )


                etfs_escolhidos = port[port > 0].index.tolist()
                for etf in etfs_escolhidos:
                    registros_etfs.append({
                        "janela": tag,
                        "iteracao": iter_file.parent.name,
                        "ticker": etf
                    })

            except Exception as e:
                print(f"⚠️ Erro na {iter_file}: {e}")

        # --- registros do S&P 500 (uma linha por semana) -----------------
        registros_retornos.extend(
            [
                {
                    "week_start": ws,
                    "retorno":   ret,
                    "tipo":      "sp500",
                    "janela":    tag,
                    "iteracao":  "sp500",  # marcador fixo
                }
                for ws, ret in zip(df_etf_futuro.index, sp500_vec)
            ]
        )

    df_avaliacoes    = pd.DataFrame(avaliacoes)
    df_retornos_long = pd.DataFrame(registros_retornos)
    df_etfs_escolhidos = pd.DataFrame(registros_etfs)
    return df_avaliacoes, df_retornos_long, df_etfs_escolhidos

# ----------------------------------------------
# 4. Boxplot de distribuição semanal
# ----------------------------------------------
def plot_boxplot_com_linha_sp500(df_retornos_long, figsize=(18, 6)):
    """
    Plota boxplots dos retornos dos portfólios (por semana)
    com uma linha por cima representando o retorno semanal do S&P 500.

    Parameters
    ----------
    df_retornos_long : DataFrame
        Deve conter as colunas:
        - 'week_start' (datetime)
        - 'retorno'
        - 'tipo': 'portfolio' ou 'sp500'
    """
    # Separa os dados
    df_portfolios = df_retornos_long[df_retornos_long["tipo"] == "portfolio"]
    df_sp500 = df_retornos_long[df_retornos_long["tipo"] == "sp500"]

    # Ordena as semanas para manter consistência no eixo X
    ordem_semanas = sorted(df_retornos_long["week_start"].unique())

    plt.figure(figsize=figsize)

    # === BOXPLOT dos portfólios ===
    sns.boxplot(
        data=df_portfolios,
        x="week_start", y="retorno",
        order=ordem_semanas,
        color="lightblue",
        fliersize=1
    )

    # === LINHA do S&P 500 ===
    sp500_agrupado = df_sp500.groupby("week_start")["retorno"].mean()
    plt.plot(
        ordem_semanas,
        sp500_agrupado.loc[ordem_semanas],
        color="red",
        marker="o",
        linestyle="-",
        label="S&P 500"
    )

    # === Estética ===
    plt.xticks(rotation=90)
    plt.xlabel("Semana")
    plt.ylabel("Retorno log semanal")
    plt.title("Distribuição dos retornos dos portfólios vs. S&P 500")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_etfs_mais_escolhidos(df_etfs_escolhidos, top_n=5, figsize=(14, 6)):
    """
    Plota os TOP ETFs mais escolhidos por janela de avaliação.
    Para cada janela, mostra até 5 barras (ETFs) com a quantidade de vezes escolhidas.

    Parameters
    ----------
    df_etfs_escolhidos : DataFrame
        Deve conter colunas: 'janela', 'ticker'
    """
    # Contagem de quantas vezes cada ETF foi escolhido por janela
    contagem = (
        df_etfs_escolhidos
        .groupby(['janela', 'ticker'])
        .size()
        .reset_index(name='n_escolhas')
    )

    # Para cada janela, pega os top N
    top_por_janela = (
        contagem.sort_values(['janela', 'n_escolhas'], ascending=[True, False])
                .groupby('janela')
                .head(top_n)
    )

    # Ordena janelas no eixo X (se forem datas)
    if np.issubdtype(top_por_janela['janela'].dtype, np.datetime64):
        top_por_janela = top_por_janela.sort_values("janela")

    # Cria o gráfico
    plt.figure(figsize=figsize)
    sns.barplot(
        data=top_por_janela,
        x="janela", y="n_escolhas", hue="ticker",
        dodge=True
    )

    plt.xticks(rotation=45)
    plt.xlabel("Janela (início do período)")
    plt.ylabel("Número de vezes escolhido")
    plt.title(f"Top {top_n} ETFs mais escolhidos por janela")
    plt.legend(title="Ticker", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

df_avaliacoes, df_retornos, df_etfs = avaliar_portfolios(
    path_results="/caminho/para/resultados",
    path_etf="/caminho/para/etfs",
    path_rf="/caminho/para/rf",
    path_sp500="/mnt/c/Users/msses/Desktop/ETF/S&P500/HistoricalData_1750724799560.csv",
    n_window=12,        # como antes
    n_future=2
)

# tabela de comparação
print(df_avaliacoes.head())

# gráfico de distribuição
plot_boxplot_com_linha_sp500(df_retornos)

plot_etfs_mais_escolhidos(df_etfs, top_n=5)

