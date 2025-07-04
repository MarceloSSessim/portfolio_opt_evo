import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan
from typing import Tuple, List, Dict
from typing import Optional
from deap import creator
from scipy.stats import dirichlet
from sklearn.cluster import KMeans

def generate_weights_with_negatives(n, min_peso, max_peso, total=1.0, max_attempts=1000):
    for _ in range(max_attempts):
        # Gera pesos aleatórios uniformemente entre min_peso e max_peso
        w = np.random.uniform(min_peso, max_peso, size=n)
        soma = np.sum(w)
        if soma == 0:
            continue
        w_norm = w / soma * total  # normaliza para que somem `total`

        # Verifica se todos ainda estão dentro dos limites
        if np.all(w_norm >= min_peso) and np.all(w_norm <= max_peso):
            return w_norm
    raise ValueError("Não foi possível gerar pesos viáveis após várias tentativas.")

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

def selecionar_tickers_kmeans(
    df_filtrado: pd.DataFrame, num_tickers: int
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aplica K-Means para selecionar `num_tickers` representativos.
    Cada ticker é representado por sua série de log_retornos ao longo do tempo.
    Seleciona o ticker mais próximo de cada centróide como representante.
    """

    # === Prepara matriz ===
    df_pivot = df_filtrado.pivot_table(
        index='ticker',
        columns=['year', 'month', 'week_start'],
        values='log_return'
    ).fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_pivot)
    tickers = df_pivot.index.to_numpy()

    # === K-Means clustering ===
    kmeans = KMeans(n_clusters=num_tickers, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # === Seleciona o ticker mais próximo de cada centróide ===
    tickers_selecionados = []

    for cluster_idx in range(num_tickers):
        # Índices dos tickers pertencentes ao cluster
        cluster_member_indices = np.where(labels == cluster_idx)[0]
        if len(cluster_member_indices) == 0:
            continue  # segurança extra (evita cluster vazio, embora raro)

        membros = X[cluster_member_indices]
        centroide = centroids[cluster_idx]

        # Calcula a distância ao centróide
        dists = np.linalg.norm(membros - centroide, axis=1)
        idx_mais_proximo = cluster_member_indices[np.argmin(dists)]

        ticker_mais_proximo = tickers[idx_mais_proximo]
        tickers_selecionados.append(ticker_mais_proximo)

    # === Filtra o DataFrame ===
    df_final = df_filtrado[df_filtrado['ticker'].isin(tickers_selecionados)]

    return df_final, tickers_selecionados

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

def avaliar(individuo, log_returns_beta, n_ativos):
    """
    Objetivos: (1) Retorno acumulado (max), (2) Variância (min).
    Aceita pesos negativos.
    """
    pesos         = np.asarray(individuo[:n_ativos])
    selecionados  = np.asarray(individuo[n_ativos:], dtype=bool)
    pesos        *= selecionados            # zera ativos não escolhidos

    if not np.isclose(pesos.sum(), 1.0, atol=1e-2):
        soma = pesos.sum()
        if soma != 0:
            pesos /= soma

    portfolio_ret = log_returns_beta @ pesos
    retorno       = portfolio_ret.mean()
    variancia     = np.var(portfolio_ret, ddof=1)  # usa ddof=1 para amostra

    return retorno, variancia               # (max, min)

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

    return abs(soma_pesos - 1) <= 0.01 and num_selecionados >= 4


def distance(individuo: List[float], n_ativos: int) -> float:
    pesos = np.array(individuo[:n_ativos])
    selecionados = np.array(individuo[n_ativos:])
    pesos = pesos * selecionados
    soma_pesos = pesos.sum()

    if soma_pesos == 0:
        return 1.0  # completamente inviável

    pesos = pesos / soma_pesos

    # Penaliza se algum ativo ultrapassar 15% (0.15), não 95%
    excesso_concentracao = max(0, max(pesos) - 0.80)

    # Penaliza também se menos de 2 ativos forem usados (reforço à factibilidade)
    penalizacao_num_ativos = max(0, 4 - selecionados.sum()) * 0.5

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
    Mutação customizada:
    - Mutação uniforme nos pesos contínuos (preserva pesos de ativos não selecionados)
    - Mutação nos bits binários (garante >= 2 ativos selecionados)
    - Normalização apenas dos pesos dos ativos selecionados
    """

    # Mutação dos bits binários
    for i in range(n_ativos, 2 * n_ativos):
        if np.random.rand() < indpb_bin:
            individuo[i] = 1 - individuo[i]

    # Garante pelo menos dois ativos selecionados
    selecionados = np.array(individuo[n_ativos:])
    if selecionados.sum() < 2:
        indices = np.random.choice(range(n_ativos), size=2, replace=False)
        selecionados[:] = 0
        selecionados[indices] = 1
        individuo[n_ativos:] = selecionados.tolist()

    # Mutação uniforme nos pesos
    pesos = np.array(individuo[:n_ativos])
    for i in range(n_ativos):
        if np.random.rand() < indpb_float:
            pesos[i] = np.random.uniform(0.0, 1.0)

    # Normaliza apenas os ativos selecionados
    ativos_sel = selecionados.astype(bool)
    soma = pesos[ativos_sel].sum()

    if soma > 0:
        pesos[ativos_sel] /= soma
    else:
        novos_pesos = np.random.dirichlet([2.0] * ativos_sel.sum())
        pesos[ativos_sel] = novos_pesos

    individuo[:n_ativos] = pesos.tolist()
    return individuo,


def cruzar_pesos(w1: np.ndarray, w2: np.ndarray, bits: np.ndarray, alpha: float) -> np.ndarray:
    """
    Executa o crossover aritmético entre dois vetores de pesos e normaliza
    os pesos dos ativos selecionados com base nos bits binários fornecidos.

    Args:
        w1: Vetor de pesos do primeiro pai (shape: [n_ativos]).
        w2: Vetor de pesos do segundo pai (shape: [n_ativos]).
        bits: Vetor binário (0 ou 1) indicando os ativos selecionados no filho (shape: [n_ativos]).
        alpha: Parâmetro de interpolação (entre 0 e 1) usado no crossover aritmético.

    Returns:
        Vetor de pesos resultante (np.ndarray), com soma igual a 1 apenas nos ativos selecionados.
        Os demais pesos são preservados.
    """
    pesos = alpha * w1 + (1 - alpha) * w2

    # Garante ao menos dois ativos selecionados
    if bits.sum() < 2:
        idx = np.random.choice(range(len(bits)), size=2, replace=False)
        bits[:] = 0
        bits[idx] = 1

    # Normaliza apenas os ativos selecionados
    selecionados = bits.astype(bool)
    soma = pesos[selecionados].sum()

    if soma > 0:
        pesos[selecionados] /= soma
    else:
        ativos_sel = np.where(selecionados)[0]
        novos_pesos = np.random.dirichlet([2.0] * len(ativos_sel))
        pesos[ativos_sel] = novos_pesos

    return pesos


def custom_crossover(ind1: List[float], ind2: List[float], n_ativos: int) -> Tuple[List[float], List[float]]:
    """
    Realiza crossover customizado entre dois indivíduos para problemas de otimização de portfólio.

    - Crossover uniforme nos bits binários de seleção de ativos.
    - Crossover aritmético nos pesos dos ativos (com alpha sorteado).
    - Normalização apenas sobre os pesos dos ativos selecionados.
    - Garante que cada filho selecione pelo menos dois ativos.

    Args:
        ind1: Primeiro indivíduo (lista com pesos e bits binários, comprimento 2 * n_ativos).
        ind2: Segundo indivíduo (mesmo formato de ind1).
        n_ativos: Número total de ativos (metade do tamanho do indivíduo).

    Returns:
        Uma tupla contendo dois indivíduos-filhos (List[float]), no mesmo formato dos pais.
    """
    alpha = np.random.rand()

    p1_w, p1_b = np.array(ind1[:n_ativos]), np.array(ind1[n_ativos:])
    p2_w, p2_b = np.array(ind2[:n_ativos]), np.array(ind2[n_ativos:])

    mask = np.random.rand(n_ativos) < 0.5
    f1_b = np.where(mask, p1_b, p2_b)
    f2_b = np.where(mask, p2_b, p1_b)

    f1_w = cruzar_pesos(p1_w, p2_w, f1_b.copy(), alpha)
    f2_w = cruzar_pesos(p2_w, p1_w, f2_b.copy(), alpha)

    filho1 = f1_w.tolist() + f1_b.tolist()
    filho2 = f2_w.tolist() + f2_b.tolist()

    return filho1, filho2


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
    alpha_val: float = 1.0,
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

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["year"]  = df["week_start"].dt.isocalendar().year
    df["week"]  = df["week_start"].dt.isocalendar().week
    df["month"] = df["week_start"].dt.month
    return df