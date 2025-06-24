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
    Aplica HDBSCAN para selecionar tickers representativos:
    percorre os clusters em rodadas por faixa (perto, longe, medio), com distâncias pré-computadas.
    Clusters maiores têm mais chance de contribuir com tickers.
    """

    # === Prepara matriz ===
    df_pivot = df_filtrado.pivot_table(
        index='ticker',
        columns=['year', 'month', 'week_start'],
        values='log_return'
    ).fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_pivot)

    # === Agrupamento com HDBSCAN ===
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    cluster_labels = clusterer.fit_predict(X)

    df_clusters = pd.DataFrame({
        'ticker': df_pivot.index,
        'cluster': cluster_labels
    })

    clusters_validos = df_clusters[df_clusters['cluster'] != -1].groupby('cluster')
    clusters_ordenados = sorted(
        clusters_validos.groups.items(),
        key=lambda item: len(item[1]),
        reverse=True
    )

    # === Pré-cálculo de distâncias e faixas por cluster ===
    clusters_info = {}
    total_tickers = 0

    for cluster_id, indices_cluster in clusters_ordenados:
        tickers_cluster = df_clusters.loc[indices_cluster, 'ticker'].values
        X_cluster = X[indices_cluster]
        centroide = X_cluster.mean(axis=0)
        distancias = np.linalg.norm(X_cluster - centroide, axis=1)

        ordenacao = np.argsort(distancias)
        n = len(ordenacao)

        idx_perto = ordenacao[:max(1, n // 3)]
        idx_medio = ordenacao[n // 3: 2 * n // 3]
        idx_longe = ordenacao[2 * n // 3:]

        clusters_info[cluster_id] = {
            'tickers': tickers_cluster,
            'faixas': {
                'perto': idx_perto,
                'medio': idx_medio,
                'longe': idx_longe
            }
        }

        total_tickers += n

    # === Criar lista de clusters ponderada pelo tamanho ===
    fator_repeticao = 10  # ajustável
    clusters_repetidos = [
        cluster_id
        for cluster_id, info in clusters_info.items()
        for _ in range(int(np.ceil(len(info['tickers']) / total_tickers * fator_repeticao)))
    ]

    # === Loop principal de seleção ===
    tickers_selecionados = []
    tipo_ordem = ['perto', 'longe', 'medio']
    rodada = 0

    while len(tickers_selecionados) < num_tickers:
        faixa = tipo_ordem[rodada % len(tipo_ordem)]

        for cluster_id in clusters_repetidos:
            if len(tickers_selecionados) >= num_tickers:
                break

            tickers_cluster = clusters_info[cluster_id]['tickers']
            faixas = clusters_info[cluster_id]['faixas']

            candidatos_idx = faixas[faixa]
            candidatos = [tickers_cluster[i] for i in candidatos_idx]

            for ticker in candidatos:
                if ticker not in tickers_selecionados:
                    tickers_selecionados.append(ticker)
                    break  # passa para o próximo cluster

        rodada += 1

    df_final = df_filtrado[df_filtrado['ticker'].isin(tickers_selecionados)]
    return df_final, tickers_selecionados

def avaliar(individuo, log_returns_beta, n_ativos, cvar_alpha):
    pesos = np.array(individuo)

    # Se quiser garantir normalização extra
    if pesos.sum() > 0:
        pesos = pesos

    # Cálculo do retorno e CVaR como antes
    portfolio_returns = log_returns_beta @ pesos
    retorno = portfolio_returns.sum()

    sorted_returns = np.sort(portfolio_returns)
    index_cut = max(1, int(np.floor(cvar_alpha * len(sorted_returns))))
    cvar = -np.mean(sorted_returns[:index_cut])

    return (retorno, cvar)

def feasible(individuo: List[float], n_ativos: int) -> bool:
    """
    Verifica se o indivíduo é viável:
    - pesos não negativos
    - somam aproximadamente 1
    - pelo menos 4 ativos com peso acima de um limiar mínimo
    """
    pesos = np.array(individuo)

    soma_pesos = pesos.sum()
    num_ativos_usados = np.sum(pesos > 0.01)  # ativos relevantes

    return (
        np.all(pesos >= 0.0) and
        abs(soma_pesos - 1.0) <= 0.01 and
        num_ativos_usados >= 4
    )


def distance(individuo: List[float], n_ativos: int) -> float:
    """
    Função de penalidade (distância) para indivíduos inviáveis.
    Penaliza:
    - soma dos pesos diferente de 1
    - concentração excessiva em um único ativo
    - número insuficiente de ativos com peso significativo
    """
    pesos = np.array(individuo)
    soma_pesos = pesos.sum()

    if soma_pesos == 0:
        return 1.0  # completamente inviável

    pesos = pesos

    excesso_concentracao = max(0, max(pesos) - 0.80)  # ou 0.15, se quiser mais diversificação
    distancia_soma = abs(1.0 - soma_pesos)
    num_ativos_usados = np.sum(pesos > 0.01)
    penalizacao_num_ativos = max(0, 4 - num_ativos_usados) * 0.5

    return distancia_soma + excesso_concentracao + penalizacao_num_ativos



def custom_mutate(individuo, indpb_float, sigma = .05, **kwargs):
    for i in range(len(individuo)):
        if np.random.rand() < indpb_float:
            individuo[i] += np.random.normal(0, sigma)
            individuo[i] = max(0.0, individuo[i])  # sem negativos

    # Normalizar após mutação
    soma = sum(individuo)
    if soma > 0:
        for i in range(len(individuo)):
            individuo[i]
    return individuo,


def custom_crossover(ind1, ind2, **kwargs):
    alpha = np.random.rand()
    for i in range(len(ind1)):
        ind1[i], ind2[i] = (
            alpha * ind1[i] + (1 - alpha) * ind2[i],
            alpha * ind2[i] + (1 - alpha) * ind1[i],
        )

    # Normalizar ambos
    for ind in [ind1, ind2]:
        soma = sum(ind)
        if soma > 0:
            for i in range(len(ind)):
                ind[i]

    return ind1, ind2

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

