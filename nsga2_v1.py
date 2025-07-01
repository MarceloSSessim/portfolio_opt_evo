import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math
from pymoo.indicators.hv import HV
from deap import base, creator, tools
from deap.tools.emo import assignCrowdingDist, sortNondominated
from scipy.stats import dirichlet
from deap.tools import DeltaPenalty
from typing import Tuple
from utils import (
    filtrar_tickers_completos,
    avaliar,
    feasible,
    distance,
    custom_mutate,
    get_adaptive_params,
    sample_dirichlet_with_bounds_fast,
    custom_crossover,
    selecionar_tickers_kmeans,
    generate_weights_with_negatives
)
from functools import partial

def run_nsga2_once(
    df_etf: pd.DataFrame,        # dados já concatenados da janela
    df_rf:  pd.DataFrame,        # dados RF correspondentes
    iteration_dir: Path        # onde salvar tudo desta execução
):
    """
    Roda 1 iteração do NSGA-II usando os DataFrames já carregados.
    Salva todos os outputs dentro de iteration_dir.
    """
    iteration_dir.mkdir(parents=True, exist_ok=True)

    num_tickers = 15
    BETA = 1
    CVaR_ALPHA = 0.05
    POP_SIZE = 600
    N_GEN = 125
    # Parâmetros de mutação e crossover adaptativos
    INDPB_FLOAT_START = 0.8
    INDPB_FLOAT_END = 0.4

    SIGMA_START = 0.3
    SIGMA_END = 0.05

    INDPB_BIN_START = 0.7
    INDPB_BIN_END = 0.3

    CXPB_INDPB = 0.3

    CX_START = 0.8
    CX_END = 0.7

    MUT_START = 0.1
    MUT_END = 0.4
    
    df_filtrado = filtrar_tickers_completos(df_etf)
    df_filtrado = df_filtrado.copy()  # ← evita qualquer "view"
    df_filtrado['year'] = pd.to_datetime(df_filtrado['week_start']).dt.year
    df_filtrado['month'] = pd.to_datetime(df_filtrado['week_start']).dt.month
    df_filtrado['week'] = pd.to_datetime(df_filtrado['week_start']).dt.isocalendar().week
    df_final, tickers_selecionados = selecionar_tickers_kmeans(df_filtrado, num_tickers)

    # Matriz de retornos e dimensões
    tickers = df_final['ticker'].unique()
    df_pivot = df_final.pivot(index='week_start', columns='ticker', values='log_return').fillna(0)


    # df_pivot: linhas = semanas (em ordem), colunas = tickers
    log_returns = df_pivot.values
    n_semanas = log_returns.shape[0]

    # Vetor de pesos beta: mais peso para semanas recentes
    beta_weights = np.array([BETA**(n_semanas - 1 - i) for i in range(n_semanas)]).reshape(-1, 1)

    # Aplica ponderação beta nos retornos
    log_returns_beta = log_returns * beta_weights
    n_semanas, n_ativos = log_returns_beta.shape

    # ========== DEAP SETUP ========== #
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # max retorno, min CVaR
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    # Adiciona o atributo necessário para crowding distance
    creator.Individual.crowding_dist = 0.0

    def generate_valid_individual(
        n_ativos: int,
        max_peso: float = 0.95,
        min_peso: float = 0.01
    ) -> creator.Individual:
        """
        Gera um indivíduo válido para um problema de otimização de portfólio com DEAP,
        combinando pesos contínuos e uma seleção binária de ativos.

        Os pesos são gerados para os ativos selecionados e normalizados para somar 1,
        respeitando os limites mínimos e máximos definidos.

        Args:
            n_ativos (int): Número total de ativos disponíveis.
            max_peso (float): Peso máximo permitido por ativo.
            min_peso (float): Peso mínimo permitido por ativo.

        Returns:
            creator.Individual: Indivíduo com `2 * n_ativos` genes (pesos + binários),
                                que satisfaz as restrições de viabilidade.
        """
        while True:
            # Sorteia quantos ativos serão selecionados (mínimo 2)
            n_selecionados = np.random.randint(4, n_ativos + 1)

            # Evita combinações impossíveis
            if n_selecionados * min_peso > 1.0 or n_selecionados * max_peso < 1.0:
                continue

            # Sorteia os ativos selecionados
            selecionados = np.zeros(n_ativos, dtype=int)
            indices = np.random.choice(n_ativos, size=n_selecionados, replace=False)
            selecionados[indices] = 1

            # Gera pesos válidos dentro dos limites
            try:
                pesos_selecionados = sample_dirichlet_with_bounds_fast(
                    n=n_selecionados,
                    max_peso=max_peso,
                    min_peso=min_peso,
                    alpha_val=1.0,
                    batch_size=100
                )
            except ValueError:
                continue  # tenta novamente

            # Monta vetor de pesos completo
            pesos = np.zeros(n_ativos)
            pesos[indices] = pesos_selecionados

            # Constrói indivíduo
            individuo = creator.Individual(pesos.tolist() + selecionados.tolist())

            if feasible(individuo, n_ativos):
                return individuo

    toolbox = base.Toolbox()

    # Genes contínuos e binários
    toolbox.register("attr_float", lambda: np.random.uniform(0, 1))
    toolbox.register("attr_bin", lambda: np.random.randint(0, 2))

    toolbox.register("individual", lambda: generate_valid_individual(n_ativos))

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Funções adaptadas com partial
    avaliar_fn = partial(avaliar, log_returns_beta=log_returns_beta, n_ativos=n_ativos)
    feasible_fn = partial(feasible, n_ativos=n_ativos)
    distance_fn = partial(distance, n_ativos=n_ativos)
    mutate_fn = partial(custom_mutate, n_ativos=n_ativos)
    crossover_fn = partial(custom_crossover, n_ativos=n_ativos)

    toolbox.register("evaluate", DeltaPenalty(feasible_fn, (-1e4, 1e4), distance_fn)(avaliar_fn))
    toolbox.register("mate", crossover_fn)
    toolbox.register("mutate", mutate_fn)
    toolbox.register("select", tools.selNSGA2)


    def _to_min(ind):
        ret, var = ind.fitness.values
        return -ret, var                    # max→min

    def calcular_hipervolume(pop, eps=1e-4):
        # só a 1ª frente não-dominada
        front = tools.sortNondominated(pop, len(pop),
                                    first_front_only=True)[0]

        if not front:
            return 0.0

        objs = np.array([_to_min(ind) for ind in front])

        # ponto de referência = (pior             + ε)
        ref = objs.max(axis=0) + eps

        return HV(ref_point=ref)(objs)      # pymoo


    # ========== EXECUÇÃO ========== #

    def individuo_para_hash(individuo: creator.Individual, precision: int = 6) -> Tuple:
        """Converte um indivíduo em uma tupla com arredondamento, para comparação rápida."""
        return tuple(round(gene, precision) for gene in individuo)

    pop = []
    hashes = set()
    while len(pop) < POP_SIZE:
        ind = toolbox.individual()
        h = individuo_para_hash(ind)
        if h not in hashes:
            pop.append(ind)
            hashes.add(h)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hypervolumes_log = []
    crowding_distances_log = []
    for gen in range(1, N_GEN + 1):
        params = get_adaptive_params(
            gen=gen,
            ngen=N_GEN,
            cxpb_start=CX_START, cxpb_end=CX_END,
            mutpb_start=MUT_START, mutpb_end=MUT_END,
            indpb_float_start=INDPB_FLOAT_START, indpb_float_end=INDPB_FLOAT_END,
            sigma_start=SIGMA_START, sigma_end=SIGMA_END,
            indpb_bin_start=INDPB_BIN_END, indpb_bin_end=INDPB_BIN_START
        )

        cxpb = params["cxpb"]
        mutpb = params["mutpb"]

        hashes = set(individuo_para_hash(ind) for ind in pop)
        offspring_unicos = []
        pais = toolbox.select(pop, len(pop))
        i = 0
        while len(offspring_unicos) < len(pop):
            p1 = toolbox.clone(pais[i % len(pais)])
            p2 = toolbox.clone(pais[(i + 1) % len(pais)])
            i += 2

            filhos_validos = []

            while len(filhos_validos) < 2:
                f1, f2 = toolbox.clone(p1), toolbox.clone(p2)

                if np.random.rand() < cxpb:
                    toolbox.mate(f1, f2)
                    del f1.fitness.values
                    del f2.fitness.values

                for f in [f1, f2]:
                    tentativa = 0
                    while True:
                        f_try = toolbox.clone(f)
                        if np.random.rand() < mutpb:
                            toolbox.mutate(
                                f_try,
                                indpb_float=params["indpb_float"],
                                sigma=params["sigma"],
                                indpb_bin=params["indpb_bin"]
                            )
                            del f_try.fitness.values

                        h = individuo_para_hash(f_try)
                        if h not in hashes:
                            hashes.add(h)
                            filhos_validos.append(f_try)
                            break

                        tentativa += 1
                        if tentativa > 20:
                            break  # evita loop infinito

                    if len(filhos_validos) >= 2 or len(offspring_unicos) + len(filhos_validos) >= len(pop):
                        break

            # <<< NOVO >>> Adiciona filhos únicos ao offspring
            for f in filhos_validos:
                if len(offspring_unicos) < len(pop):
                    offspring_unicos.append(f)

        num_novos = int(0.20 * POP_SIZE)
        novos_inds = []
        while len(novos_inds) < num_novos:
            ind = toolbox.individual()
            h = individuo_para_hash(ind)
            if h not in hashes:
                novos_inds.append(ind)
                hashes.add(h)
        # Coletar fitness da população atual
        retornos_geracao = [ind.fitness.values[0] for ind in pop]
        vars_geracao = [ind.fitness.values[1] for ind in pop]
        print(f"Geração {gen:3d} | Retorno Máx: {max(retornos_geracao):.6f} | Variância Mín: {min(vars_geracao):.6f}")


        todos = pop + offspring_unicos + novos_inds
        invalid_ind = [ind for ind in todos if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

                # Seleciona nova população com NSGA-II
        pop[:] = tools.selNSGA2(todos, POP_SIZE)

        # Ordena e separa por frentes de Pareto
        fronts = tools.sortNondominated(pop, len(pop), first_front_only=False)

        # Função auxiliar para calcular crowding distance manualmente
        def crowding_distance(front):
            """
            Atribui crowding_dist para cada indivíduo da frente.
            Retorna a lista de distâncias.
            """
            n = len(front)
            if n == 0:
                return []

            distances = [0.0] * n
            num_obj = len(front[0].fitness.values)

            for m in range(num_obj):
                # Ordena índices pelo objetivo m
                idx_sorted = sorted(range(n), key=lambda i: front[i].fitness.values[m])
                f_min = front[idx_sorted[0]].fitness.values[m]
                f_max = front[idx_sorted[-1]].fitness.values[m]

                distances[idx_sorted[0]] = float('inf')
                distances[idx_sorted[-1]] = float('inf')

                if f_max == f_min:
                    continue  # evita divisão por zero

                for k in range(1, n - 1):
                    prev_val = front[idx_sorted[k - 1]].fitness.values[m]
                    next_val = front[idx_sorted[k + 1]].fitness.values[m]
                    distances[idx_sorted[k]] += (next_val - prev_val) / (f_max - f_min)

            # Atribui ao atributo .crowding_dist de cada indivíduo
            for ind, d in zip(front, distances):
                ind.crowding_dist = d

            return distances

        # Aplica o cálculo de crowding distance em cada frente
        for fr in fronts:
            crowding_distance(fr)

        # Calcula média das distâncias finitas
        dists_finitas = [
            ind.crowding_dist for ind in pop
            if hasattr(ind, "crowding_dist") and math.isfinite(ind.crowding_dist)
        ]
        cd_val = float(np.mean(dists_finitas)) if dists_finitas else 0.0

        # Calcula hipervolume da população atual
        hv_val = calcular_hipervolume(pop)

        # Salva os logs da geração
        hypervolumes_log.append((gen, hv_val))
        crowding_distances_log.append((gen, round(cd_val, 8)))


        


    # Pareto colorido (frentes)
    fronts = tools.sortNondominated(pop, len(pop), first_front_only=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(fronts)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, front in enumerate(fronts):
        xs = [ind.fitness.values[1] for ind in front]
        ys = [ind.fitness.values[0] for ind in front]
        ax.scatter(xs, ys, color=colors[i], alpha=0.6, edgecolor="k", s=30,
                label=f"F{i+1}")
    ax.set(xlabel="CVaR", ylabel="Retorno", title="Pareto – frentes coloridas")
    ax.legend(title="Frentes", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(); fig.tight_layout()
    fig.savefig(iteration_dir / "pareto_colorido.png", dpi=150)
    plt.close(fig)

    ## 4.4 Portfólios de toda a população
    portfolios = []
    for ind in pop:
        w = np.array(ind[:n_ativos]) * np.array(ind[n_ativos:])
        if w.sum(): w /= w.sum()
        portfolios.append({t: p for t, p in zip(df_pivot.columns, w)})

    pd.DataFrame(portfolios).fillna(0.0).to_csv(
        iteration_dir / "populacao_final_portfolios.csv", index=False
    )
    df_hv = pd.DataFrame(hypervolumes_log, columns=["generation", "hypervolume"])
    df_cd = pd.DataFrame(crowding_distances_log, columns=["generation", "crowding_distance"])

    df_hv.to_csv(iteration_dir / "log_hypervolume.csv", index=False)
    df_cd.to_csv(iteration_dir / "log_crowding_distance.csv", index=False)
    # ------------------------------------------------------------------
    # 0) Utilidades de cálculo
    # ------------------------------------------------------------------
    def get_portfolio_returns(ind, df_ret):
        """
        Constrói pesos a partir do indivíduo [pesos_cont | máscara_bin]
        e devolve o vetor de retornos semanais (shape = T,).
        Retorna None se a soma bruta de pesos == 0 (indivíduo inviável).
        """
        n_assets = df_ret.shape[1]
        w_raw = np.asarray(ind[:n_assets]) * np.asarray(ind[n_assets:])
        if w_raw.sum() == 0:
            return None
        w = w_raw / w_raw.sum()
        return df_ret @ w

    # ------- Dominância de 1ª ordem --------
    def dominates_fsd(ret_a, ret_b):
        """
        True se A domina B em 1ª ordem:
        CDF_A(x) <= CDF_B(x)  ∀ x   ↔   valores ordenados de A >= de B
        """
        a_sorted = np.sort(ret_a)
        b_sorted = np.sort(ret_b)
        return np.all(a_sorted >= b_sorted)

    # ------- Dominância de 2ª ordem --------
    def dominates_ssd(ret_a, ret_b):
        """
        True se A domina B em 2ª ordem:
        ∫_{-∞}^x F_A(t) dt  <=  ∫_{-∞}^x F_B(t) dt   ∀ x.
        Implementação:
        • Constrói grid de valores único.
        • Calcula CDF empírica de A e B nesse grid.
        • Soma cumulativa (discretização) ≈ integral.
        """
        # Grid comum
        grid = np.sort(np.unique(np.concatenate([ret_a, ret_b])))
        cdf_a = np.searchsorted(np.sort(ret_a), grid, side='right') / ret_a.size
        cdf_b = np.searchsorted(np.sort(ret_b), grid, side='right') / ret_b.size
        # Integral acumulada (Δx constante = 1 -> basta cumsum)
        area_a = np.cumsum(cdf_a)
        area_b = np.cumsum(cdf_b)
        return np.all(area_a <= area_b)

    # ------------------------------------------------------------------
    # 1) Calcula retornos de todos indivíduos da 1ª frente
    # ------------------------------------------------------------------
    first_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    ret_matrix = df_pivot.values                # T x N
    port_returns = {}                           # id(ind) -> vetor retornos
    valid_inds   = []
    for ind in first_front:
        r = get_portfolio_returns(ind, ret_matrix)
        if r is not None:
            port_returns[id(ind)] = r
            valid_inds.append(ind)

    # ------------------------------------------------------------------
    # 2) FILTRO FSD (remove dominados em 1ª ordem)
    # ------------------------------------------------------------------
    survivors_fsd = []
    for ind_a in valid_inds:
        if not any(
            dominates_fsd(port_returns[id(ind_b)], port_returns[id(ind_a)])
            for ind_b in valid_inds if ind_b is not ind_a
        ):
            survivors_fsd.append(ind_a)

    # ------------------------------------------------------------------
    # 3) FILTRO SSD (se >1 sobreviveram)
    # ------------------------------------------------------------------
    if len(survivors_fsd) == 1:
        final_candidates = survivors_fsd
    else:
        final_candidates = []
        for ind_a in survivors_fsd:
            if not any(
                dominates_ssd(port_returns[id(ind_b)], port_returns[id(ind_a)])
                for ind_b in survivors_fsd if ind_b is not ind_a
            ):
                final_candidates.append(ind_a)

    # ------------------------------------------------------------------
    # 4) Desempate (maior retorno médio)
    # ------------------------------------------------------------------
    best_ind = max(
        final_candidates,
        key=lambda ind: port_returns[id(ind)].mean()
    )

    # ------------------------------------------------------------------
    # 5) Salva portfólio vencedor
    # ------------------------------------------------------------------
    n_assets = df_pivot.shape[1]
    w_final = np.array(best_ind[:n_assets]) * np.array(best_ind[n_assets:])
    w_final /= w_final.sum()

    best_port = (
        pd.Series(w_final, index=df_pivot.columns)
        .sort_values(ascending=False)
    )

    best_port.to_csv(
        iteration_dir / "best_portfolio_by_SSD.csv",
        header=["weight"], index_label="ticker"
    )

    print("Portfólio escolhido por dominância estocástica (FSD → SSD) salvo em:",
        iteration_dir / "best_portfolio_by_SSD.csv")


    # retorno final
    return best_port


