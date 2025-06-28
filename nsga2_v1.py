import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from deap import base, creator, tools
from scipy.stats import dirichlet
from deap.tools import DeltaPenalty
from typing import Tuple
from utils import (
    selecionar_tickers_representativos,
    filtrar_tickers_completos,
    avaliar,
    feasible,
    distance,
    custom_mutate,
    get_adaptive_params,
    sample_dirichlet_with_bounds_fast,
    custom_crossover,
    selecionar_tickers_kmeans
)
from functools import partial

# ========== PARÂMETROS ========== #

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

    num_tickers = 30
    BETA = .98
    CVaR_ALPHA = 0.05
    POP_SIZE = 700
    N_GEN = 200
    # Parâmetros de mutação e crossover adaptativos
    INDPB_FLOAT_START = 0.8
    INDPB_FLOAT_END = 0.6

    SIGMA_START = 0.3
    SIGMA_END = 0.05

    INDPB_BIN_START = 0.9
    INDPB_BIN_END = 0.6

    CXPB_INDPB = 0.3

    CX_START = 0.4
    CX_END = 0.2

    MUT_START = 0.5
    MUT_END = 0.8
    
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
    avaliar_fn = partial(avaliar, log_returns_beta=log_returns_beta, n_ativos=n_ativos, cvar_alpha = CVaR_ALPHA)
    feasible_fn = partial(feasible, n_ativos=n_ativos)
    distance_fn = partial(distance, n_ativos=n_ativos)
    mutate_fn = partial(custom_mutate, n_ativos=n_ativos)
    crossover_fn = partial(custom_crossover, n_ativos=n_ativos)

    toolbox.register("evaluate", DeltaPenalty(feasible_fn, (-1e4, 1e4), distance_fn)(avaliar_fn))
    toolbox.register("mate", crossover_fn)
    toolbox.register("mutate", mutate_fn)
    toolbox.register("select", tools.selNSGA2)

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
    #LOG 
    historico_retornos = []
    historico_vars = []

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

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

        historico_retornos.append(retornos_geracao)
        historico_vars.append(vars_geracao)
        print(f"Geração {gen:3d} | Retorno Máx: {max(retornos_geracao):.6f} | Variância Mín: {min(vars_geracao):.6f}")

        todos = pop + offspring_unicos + novos_inds
        invalid_ind = [ind for ind in todos if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        pop[:] = tools.selNSGA2(todos, POP_SIZE)

    ## 4.2 histórico: df_ret / df_var
    df_ret = pd.DataFrame(historico_retornos)
    df_var = pd.DataFrame(historico_vars)

    # Plot retornos
    fig, ax = plt.subplots()
    df_ret.mean(axis=1).plot(ax=ax, label="Média Retorno")
    df_ret.min(axis=1).plot(ax=ax, label="Melhor Retorno", ls="--")
    ax.set(title="Evolução dos Retornos", xlabel="Geração", ylabel="Retorno")
    ax.grid(); ax.legend()
    fig.savefig(iteration_dir / "retorno_plot.png", dpi=150)
    plt.close(fig)

    # Plot CVaR
    fig, ax = plt.subplots()
    df_var.mean(axis=1).plot(ax=ax, label="Média CVaR")
    df_var.min(axis=1).plot(ax=ax, label="Melhor CVaR", ls="--")
    ax.set(title="Evolução da CVaR", xlabel="Geração", ylabel="CVaR")
    ax.grid(); ax.legend()
    fig.savefig(iteration_dir / "cvar_plot.png", dpi=150)
    plt.close(fig)

    ## 4.3 Pareto final
    vars_ = [ind.fitness.values[1] for ind in pop]
    rets_ = [ind.fitness.values[0] for ind in pop]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(vars_, rets_, alpha=0.6, edgecolor="k")
    ax.set(xlabel="CVaR", ylabel="Retorno", title="Pareto – População Final")
    ax.grid(); fig.tight_layout()
    fig.savefig(iteration_dir / "pareto_final.png", dpi=150)
    plt.close(fig)

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

    ## 5. Sharpe na 1ª frente
    rf_vec = (df_rf.set_index("week_start")["log_return"]
            .reindex(df_pivot.index)
            .fillna(0.0)
            .values)

    first_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    freq_year = 52
    best_sharpe = -np.inf
    best_ind = None

    for ind in first_front:
        w_raw = np.array(ind[:n_ativos]) * np.array(ind[n_ativos:])
        if w_raw.sum() == 0:
            continue
        w = w_raw / w_raw.sum()
        r_port = df_pivot.values.dot(w)
        excess = r_port - rf_vec
        mu, sigma = excess.mean(), r_port.std()
        sharpe = (mu * freq_year) / (sigma * np.sqrt(freq_year)) if sigma > 0 else np.nan
        if sharpe > best_sharpe:
            best_sharpe, best_ind = sharpe, ind

    # salva portfólio campeão
    weights = (np.array(best_ind[:n_ativos]) *
            np.array(best_ind[n_ativos:]))
    weights = weights / weights.sum()
    best_port = (pd.Series(weights, index=df_pivot.columns)
                .loc[lambda s: s > 0]
                .sort_values(ascending=False))

    # Salva CSV com índice (ticker)
    best_port.to_csv(
        iteration_dir / "best_portfolio_by_sharpe.csv",
        header=["weight"],
        index_label="ticker"
    )

    # Salva resumo em texto
    resumo = (
        f"Melhor Sharpe anualizado: {best_sharpe:.4f}\n"
        "Pesos (normalizados):\n" +
        "\n".join(f"{tk}: {w:.4f}" for tk, w in best_port.items())
    )
    (iteration_dir / "summary.txt").write_text(resumo, encoding="utf-8")

    # retorno final
    return best_sharpe, best_port


