# ------------------------------------------------------------
# 0. Imports e configuração de diretórios (WSL)
# ------------------------------------------------------------
from pathlib import Path
import re
import pandas as pd
import numpy as np

BASE_DIR   = Path("/mnt/c/Users/msses/Desktop/ETF")
DIR_RESULTS = BASE_DIR / "results_NSGA2"           # .../AAAA_MM_DD/iter_XXX/...
DIR_ETF     = BASE_DIR / "weekly_log_returns"      # arquivos weekly_log_returns_YYYY_MM_DD.csv
DIR_RF      = BASE_DIR / "DTB1"                    # arquivos rf_weekly_log_returns_YYYY_MM_DD.csv
SP500_PATH  = BASE_DIR / "S&P500" / "HistoricalData_1750724799560.csv"

# ------------------------------------------------------------
# 1. Funções utilitárias
# ------------------------------------------------------------
def preparar_sp500(path_sp500: Path, week_freq: str = "W-MON") -> pd.Series:
    """
    Converte o CSV diário do S&P 500 (coluna 'Date', 'Close/Last')
    para log-retorno semanal alinhado à segunda-feira.
    Retorna pd.Series indexada por week_start.
    """
    df = (
        pd.read_csv(path_sp500)
          .assign(Date=lambda d: pd.to_datetime(d["Date"]))
          .sort_values("Date")
    )
    df.set_index("Date", inplace=True)
    # Preço de fechamento na data → último preço da semana “terminando” na data
    weekly_px = df["Close/Last"].resample(week_freq).last().dropna()
    weekly_lr = np.log(weekly_px).diff().dropna()
    weekly_lr.name = "sp500_log_return"
    return weekly_lr

def calcular_metricas(rets: np.ndarray, rf: np.ndarray, freq: int = 52) -> dict:
    """
    Recebe vetor de retornos do portfólio e vetor de risk-free (mesma length, log-retornos).
    Devolve Sharpe e Sortino anualizados + retorno médio e vol. anualizada.
    """
    excess = rets - rf
    mu           = excess.mean()
    sigma        = rets.std()
    downside_std = np.sqrt(((np.minimum(0, excess))**2).mean())

    sharpe  = (mu * freq) / (sigma        * np.sqrt(freq)) if sigma        > 0 else np.nan
    sortino = (mu * freq) / (downside_std * np.sqrt(freq)) if downside_std > 0 else np.nan

    return {"sharpe_ratio": sharpe, "sortino_ratio": sortino}

def date_from_filename(fname: str) -> str:
    """Extrai AAAA_MM_DD do nome do arquivo."""
    m = re.search(r"\d{4}_\d{2}_\d{2}", fname)
    return m.group(0) if m else ""

# ------------------------------------------------------------
# 2. Pré-carrega listas de arquivos ordenadas por data
# ------------------------------------------------------------
all_etf_files = sorted(DIR_ETF.glob("weekly_log_returns_*.csv"),
                       key=lambda p: date_from_filename(p.name))
all_rf_files  = sorted(DIR_RF .glob("rf_weekly_log_returns_*.csv"),
                       key=lambda p: date_from_filename(p.name))

# Mapeia data_tag → posição dentro da lista (útil p/ achar janelas futuras)
idx_etf_by_tag = {date_from_filename(p.name): i for i, p in enumerate(all_etf_files)}

# S&P 500 em série semanal
sp500_weekly = preparar_sp500(SP500_PATH)

# ------------------------------------------------------------
# 3. Percorre todos os portfolios gerados
# ------------------------------------------------------------
records = []
WINDOW_TRAIN  = 12   # 12 arquivos (= 12 semanas de treino)
WINDOW_TEST   = 2    # 2 arquivos (= 8 semanas de avaliação)

for best_csv in DIR_RESULTS.glob("*/*/best_portfolio_by_sharpe.csv"):
    # Paths importantes
    iter_dir   = best_csv.parent                       # .../YYYY_MM_DD/iter_###
    date_tag   = iter_dir.parent.name                  # pasta AAAA_MM_DD
    iter_tag   = iter_dir.name                         # iter_###
    
    # --- Verifica se conseguimos achar posição do date_tag nos arquivos ETF
    if date_tag not in idx_etf_by_tag:
        continue  # pula se não achar correspondência
    
    i0 = idx_etf_by_tag[date_tag]            # índice do 1.º arquivo da janela de treino
    eval_etf_files = all_etf_files[i0 + WINDOW_TRAIN : i0 + WINDOW_TRAIN + WINDOW_TEST]
    eval_rf_files  = all_rf_files [i0 + WINDOW_TRAIN : i0 + WINDOW_TRAIN + WINDOW_TEST]
    if len(eval_etf_files) < WINDOW_TEST or len(eval_rf_files) < WINDOW_TEST:
        continue  # não há 2 arquivos futuros; pula
    
    # --- Lê portfólio (tickers + pesos normalizados)
    df_port = (
        pd.read_csv(best_csv)
          .assign(weight=lambda d: d["weight"] / d["weight"].sum())
          .query("weight > 0")              # garante apenas ativos escolhidos
    )
    
    # --- Concatena retornos dos 2 arquivos de avaliação
    df_eval = pd.concat([pd.read_csv(f) for f in eval_etf_files], ignore_index=True)
    df_rf   = pd.concat([pd.read_csv(f) for f in eval_rf_files],  ignore_index=True)
    
    # Semana inicial da avaliação = primeiro arquivo futuro
    eval_start = pd.to_datetime(date_from_filename(eval_etf_files[0].name))
    
    # --- Calcula vetor de retornos semanais do portfólio (len = 2)
    #     Para cada semana: sum(weight_i * log_return_i)
    port_rets = (
        df_eval
            .merge(df_port[["ticker", "weight"]], on="ticker", how="inner")
            .groupby("week_start", as_index=False)
            .apply(lambda g: (g["log_return"] * g["weight"]).sum())
            .rename(columns={0: "port_ret"})
            .sort_values("week_start")
    )
    
    # --- Vetor risk-free (assume coluna 'log_return' no arquivo de RF)
    rf_rets = (
        df_rf
            .sort_values("week_start")["log_return"]
            .values
    )[:len(port_rets)]  # garante mesmo tamanho
    
    # --- Métricas
    metr = calcular_metricas(port_rets["port_ret"].values, rf_rets)
    
    # --- Retorno acumulado portfólio e S&P 500
    port_sum  = port_rets["port_ret"].sum()
    sp500_sum = sp500_weekly.loc[
        port_rets["week_start"].values
    ].sum()
    
    # --- Salva registro
    records.append({
        "date_eval":   eval_start,          # início da janela de avaliação
        "iter":        iter_tag,            # iter_###
        "retorno":     port_sum,
        **metr,                             # sharpe_ratio, sortino_ratio
        "sp500":       sp500_sum,
    })

# ------------------------------------------------------------
# 4. DataFrame final
# ------------------------------------------------------------
df_avaliacao = pd.DataFrame(records) \
                 .sort_values(["date_eval", "iter"]) \
                 .reset_index(drop=True)

print(df_avaliacao.head())

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
    fig = plt.gcf()  # captura a figura atual
    plt.close(fig)   # evita que o gráfico fique "pendurado" na tela
    return fig       # permite salvar fora da função
    

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
    fig = plt.gcf()
    plt.close(fig)
    return fig

df_avaliacoes, df_retornos, df_etfs = avaliar_portfolios(
    path_results="/mnt/c/Users/msses/Desktop/ETF/results_NSGA2",
    path_etf="/mnt/c/Users/msses/Desktop/ETF/weekly_log_returns",
    path_rf="/mnt/c/Users/msses/Desktop/ETF/DTB1",
    path_sp500="/mnt/c/Users/msses/Desktop/ETF/S&P500/HistoricalData_1750724799560.csv",
    n_window=12,        # como antes
    n_future=2
)

# tabela de comparação
print(df_avaliacoes.head())

# gráfico de distribuição
plot_boxplot_com_linha_sp500(df_retornos)

plot_etfs_mais_escolhidos(df_etfs, top_n=5)

# Criar pasta para salvar resultados da avaliação
output_dir = Path("/mnt/c/Users/msses/Desktop/ETF/avaliacao_final")
output_dir.mkdir(parents=True, exist_ok=True)

# Salva os DataFrames principais
df_avaliacoes.to_csv(output_dir / "avaliacoes.csv", index=False)
df_retornos.to_csv(output_dir / "retornos_long.csv", index=False)
df_etfs.to_csv(output_dir / "etfs_escolhidos.csv", index=False)

# Salva os gráficos em arquivos PNG
# --- Boxplot com linha do S&P 500 ---
fig1 = plot_boxplot_com_linha_sp500(df_retornos)
fig1.savefig(output_dir / "boxplot_sp500.png", dpi=150)

fig2 = plot_etfs_mais_escolhidos(df_etfs, top_n=5)
fig2.savefig(output_dir / "etfs_mais_escolhidos.png", dpi=150)