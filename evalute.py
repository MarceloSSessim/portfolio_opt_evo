import pandas as pd
import numpy as np

# ============================================================================
# 1) PARÂMETROS
# ============================================================================

csv_populacao = "populacao_final_portfolios.csv"   # gerado no seu código
csv_retornos  = "retornos_2_meses.csv"             # <-- substitua pelo seu arquivo
risk_free_daily = 0.0                              # taxa livre diária (log), ajuste se necessário
periodos_ano = 252                                 # se for diário; use 21 para mensal, etc.

# ============================================================================
# 2) LEITURA DOS DADOS
# ============================================================================

# a) pesos da população final  ----------------------------------------------
df_weights = pd.read_csv(csv_populacao)

# b) retornos log dos 2 meses seguintes -------------------------------------
#    - índice: datas
#    - colunas: mesmos tickers presentes em df_weights
df_future = pd.read_csv(csv_retornos, index_col=0, parse_dates=True)

# garantir que apenas os ativos em comum sejam usados
common_cols = df_weights.columns.intersection(df_future.columns)
df_weights = df_weights[common_cols]
df_future  = df_future[common_cols]

# ============================================================================
# 3) FUNÇÕES AUXILIARES
# ============================================================================

def portfolio_returns(weights: pd.Series, rets: pd.DataFrame) -> pd.Series:
    """
    Calcula a série de retornos diários de um portfólio.
    """
    # Se algum peso faltou na série, assume 0
    w = weights.reindex(rets.columns, fill_value=0).values
    return rets.dot(w)              # log-retorno diário do portfólio

def annualize(avg_daily, std_daily, n=periodos_ano):
    """
    Devolve média e desvio padrão anualizados.
    """
    return avg_daily * n, std_daily * np.sqrt(n)

def sharpe_ratio(avg_d, std_d, rf=risk_free_daily, n=periodos_ano):
    mu_a, sigma_a = annualize(avg_d - rf, std_d, n)
    return mu_a / sigma_a if sigma_a != 0 else np.nan

def sortino_ratio(ret_series, rf=risk_free_daily, n=periodos_ano):
    diff = ret_series - rf
    avg_diff = diff.mean()
    downside_std = diff[diff < 0].std()
    if pd.isna(downside_std) or downside_std == 0:
        return np.nan
    return (avg_diff * n) / (downside_std * np.sqrt(n))

# ============================================================================
# 4) AVALIAÇÃO DE TODA A POPULAÇÃO
# ============================================================================

metricas = []
for idx, weights in df_weights.iterrows():
    r_port = portfolio_returns(weights, df_future)

    avg_d   = r_port.mean()
    std_d   = r_port.std()
    log_ret = r_port.sum()                # log-retorno acumulado (≈ ln(1+R_total))

    sharpe  = sharpe_ratio(avg_d, std_d)
    sortino = sortino_ratio(r_port)

    metricas.append({
        "portfolio_id": idx,
        "log_return_total": log_ret,
        "avg_daily": avg_d,
        "std_daily": std_d,
        "sharpe_ann": sharpe,
        "sortino_ann": sortino
    })

df_metrics = pd.DataFrame(metricas)

# ============================================================================
# 5) SELEÇÃO DA MELHOR SOLUÇÃO
#    (aqui usamos Sharpe; troque a linha abaixo para outro critério se quiser)
# ============================================================================

df_metrics_sorted = df_metrics.sort_values("sharpe_ann", ascending=False)
melhor_id = df_metrics_sorted.iloc[0]["portfolio_id"]
melhor_portfolio = df_weights.loc[melhor_id]

print("===== MELHOR PORTFÓLIO (criterio: Sharpe) =====")
print(melhor_portfolio[melhor_portfolio > 0].sort_values(ascending=False))
print("\nMétricas:")
print(df_metrics_sorted.head(1).T)

# Se quiser salvar o ranking completo:
# df_metrics_sorted.to_csv("avaliacao_portfolios.csv", index=False)
