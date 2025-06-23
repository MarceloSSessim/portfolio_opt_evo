import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan
from typing import Tuple, List, Dict
from typing import Optional
from deap import creator
from scipy.stats import dirichlet

def filtrar_tickers_completos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra o DataFrame para manter apenas os tickers com o número máximo de observações.

    Args:
        df: DataFrame contendo a coluna 'ticker'.

    Returns:
        DataFrame contendo apenas os tickers completos.
    """
    contagem_por_ticker = df['ticker'].value_counts()
    n_max = contagem_por_ticker.max()
    tickers_completos = contagem_por_ticker[contagem_por_ticker == n_max].index.tolist()
    df_filtrado = df[df['ticker'].isin(tickers_completos)]
    return df_filtrado

def selecionar_tickers_representativos(
    df_filtrado: pd.DataFrame, num_tickers: int
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aplica HDBSCAN para selecionar os tickers mais representativos com base nos log_returns.

    Args:
        df_filtrado: DataFrame com tickers completos.
        num_tickers: Número de tickers representativos a retornar.

    Returns:
        df_final: DataFrame apenas com os tickers selecionados.
        tickers_selecionados: Lista de tickers mais representativos.
    """
    df_pivot = df_filtrado.pivot_table(
        index='ticker',
        columns=['year', 'month', 'week_start'],
        values='log_return'
    ).fillna(0)

    X = StandardScaler().fit_transform(df_pivot)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    cluster_labels = clusterer.fit_predict(X)

    df_clusters = pd.DataFrame({
        'ticker': df_pivot.index,
        'cluster': cluster_labels
    })

    maior_cluster = df_clusters['cluster'].value_counts().idxmax()
    tickers_maior_cluster = df_clusters[df_clusters['cluster'] == maior_cluster]['ticker'].tolist()

    tickers_selecionados = tickers_maior_cluster[:num_tickers]
    df_final = df_filtrado[df_filtrado['ticker'].isin(tickers_selecionados)]

    return df_final, tickers_selecionados

def avaliar(
    individuo: List[float],
    log_returns: np.ndarray,
    beta: float,
) -> Tuple[float, float]:
    """
    Avalia o desempenho de um indivíduo com base no retorno ponderado e na variância dos retornos.

    Args:
        individuo: Lista contendo pesos e seleção binária dos ativos.
        log_returns: Matriz [T x N] de retornos logarítmicos.
        beta: Fator de desconto temporal.
        cvar_alpha: (obsoleto) Percentil do CVaR — não utilizado aqui.

    Returns:
        Tuple contendo (retorno ponderado, variância dos retornos).
    """
    n_semanas, n_ativos = log_returns.shape
    pesos = np.array(individuo[:n_ativos])
    selecionados = np.array(individuo[n_ativos:])
    pesos = pesos * selecionados

    soma_pesos = pesos.sum()
    if soma_pesos > 0:
        pesos = pesos / soma_pesos

    beta_weights = np.array([beta**(n_semanas - 1 - i) for i in range(n_semanas)])
    portfolio_returns = log_returns @ pesos
    weighted_return = np.sum(portfolio_returns * beta_weights)

    # Variância dos retornos da carteira
    variancia = np.var(portfolio_returns)

    return weighted_return, variancia

def feasible(individuo: List[float], n_ativos: int) -> bool:
    """
    Verifica se o indivíduo é viável: soma dos pesos igual a 1 e pelo menos dois ativos selecionados.

    Args:
        individuo: Lista com pesos e binários.
        n_ativos: Número total de ativos.

    Returns:
        True se o indivíduo for viável, False caso contrário.
    """
    pesos = np.array(individuo[:n_ativos])
    selecionados = np.array(individuo[n_ativos:])
    pesos = pesos * selecionados

    soma_pesos = pesos.sum()
    num_selecionados = selecionados.sum()
    # DEBUG
    print(f"[FEASIBLE] Soma pesos: {soma_pesos:.4f} | Selecionados: {num_selecionados} | Viável? {abs(soma_pesos - 1) <= 0.01 and num_selecionados > 1}")

    return abs(soma_pesos - 1) <= 0.01 and num_selecionados >= 2


def distance(individuo: List[float], n_ativos: int) -> float:
    pesos = np.array(individuo[:n_ativos])
    selecionados = np.array(individuo[n_ativos:])
    pesos = pesos * selecionados
    soma_pesos = pesos.sum()

    if soma_pesos == 0:
        return 1.0  # completamente inviável

    pesos = pesos / soma_pesos

    # Penaliza se algum ativo ultrapassar 15% (0.15), não 95%
    excesso_concentracao = max(0, max(pesos) - 0.15)

    # Penaliza também se menos de 2 ativos forem usados (reforço à factibilidade)
    penalizacao_num_ativos = max(0, 2 - selecionados.sum()) * 0.5

    distancia_soma = abs(1.0 - soma_pesos)

    return distancia_soma + excesso_concentracao + penalizacao_num_ativos


def custom_mutate(
    individuo: List[float],
    n_ativos: int,
    indpb_float: float,
    sigma: float,  # ignorado aqui, mantido por compatibilidade
    indpb_bin: float
) -> Tuple[List[float]]:
    """
    Realiza mutações com:
    - Mutação uniforme nos pesos contínuos (com normalização posterior)
    - Mutação nos bits binários (garantindo >= 2 ativos)
    - Normalização final dos pesos nos ativos selecionados

    Args:
        individuo: Lista com pesos + bits binários.
        n_ativos: Número total de ativos.
        indpb_float: Probabilidade de mutação de cada peso.
        sigma: (Ignorado) Mantido para compatibilidade com DEAP.
        indpb_bin: Probabilidade de mutação de cada bit binário.

    Returns:
        Uma tupla com o indivíduo mutado.
    """
    # Muta bits binários
    for i in range(n_ativos, 2 * n_ativos):
        if np.random.rand() < indpb_bin:
            individuo[i] = 1 - individuo[i]

    # Garante ao menos dois ativos selecionados
    selecionados = np.array(individuo[n_ativos:])
    if selecionados.sum() < 2:
        indices = np.random.choice(range(n_ativos), size=2, replace=False)
        selecionados[:] = 0
        selecionados[indices] = 1
        individuo[n_ativos:] = selecionados.tolist()

    # Mutação uniforme nos pesos (antes da normalização)
    pesos = np.array(individuo[:n_ativos])
    for i in range(n_ativos):
        if np.random.rand() < indpb_float:
            pesos[i] = np.random.uniform(0.0, 1.0)

    # Normaliza os pesos apenas nos ativos selecionados
    pesos = pesos * selecionados
    soma = pesos.sum()

    if soma > 0:
        pesos = pesos / soma
    else:
        # fallback raro: todos pesos mutados zeraram
        ativos_sel = np.where(selecionados == 1)[0]
        novos_pesos = np.random.dirichlet([2.0] * len(ativos_sel))
        pesos = np.zeros(n_ativos)
        pesos[ativos_sel] = novos_pesos

    individuo[:n_ativos] = pesos.tolist()
    return individuo,



def get_adaptive_params(
    gen: int,
    ngen: int,
    cxpb_start: float,
    cxpb_end: float,
    mutpb_start: float,
    mutpb_end: float,
    indpb_float_start: float,
    indpb_float_end: float,
    sigma_start: float,
    sigma_end: float,
    indpb_bin_start: float,
    indpb_bin_end: float
) -> Dict[str, float]:
    """
    Adapta dinamicamente múltiplos hiperparâmetros com base na geração atual.

    Returns:
        Um dicionário com os parâmetros atualizados.
    """
    frac = gen / ngen

    return {
        "cxpb": cxpb_start + frac * (cxpb_end - cxpb_start),
        "mutpb": mutpb_start + frac * (mutpb_end - mutpb_start),
        "indpb_float": indpb_float_start + frac * (indpb_float_end - indpb_float_start),
        "sigma": sigma_start + frac * (sigma_end - sigma_start),
        "indpb_bin": indpb_bin_start + frac * (indpb_bin_end - indpb_bin_start)
    }

def sample_dirichlet_with_bounds_fast(
    n: int,
    max_peso: float,
    min_peso: float,
    alpha_val: float = 2.0,
    batch_size: int = 100,
    max_tries: int = 1000
) -> np.ndarray:
    """
    Gera vetor Dirichlet com limites, de forma mais eficiente.
    """
    alpha = [alpha_val] * n
    for _ in range(max_tries):
        samples = dirichlet.rvs(alpha, size=batch_size)
        mask = (samples <= max_peso).all(axis=1) & (samples >= min_peso).all(axis=1)
        if mask.any():
            return samples[mask][0]
    raise ValueError("Não foi possível gerar vetor dentro dos limites.")

