import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools
from scipy.stats import dirichlet
from deap.tools import DeltaPenalty
from utils import (
    selecionar_tickers_representativos,
    filtrar_tickers_completos,
    avaliar,
    feasible,
    distance,
    custom_mutate,
    get_adaptive_params,
    sample_dirichlet_with_bounds_fast
)
from functools import partial

# ========== PARÂMETROS ========== #
year = 2023
folder_path = "/mnt/c/Users/msses/Desktop/ETF/weekly_log_returns"
num_tickers = 30
BETA = .98
CVaR_ALPHA = 0.05
POP_SIZE = 500
N_GEN = 100
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

MUT_START = 0.8
MUT_END = 0.6

# ========== CARREGAMENTO DOS DADOS ========== #
arquivos_do_ano = [
    f for f in os.listdir(folder_path)
    if f.startswith(f"weekly_log_returns_{year}_") and f.endswith(".csv")
]

df_concatenado = pd.concat(
    [pd.read_csv(os.path.join(folder_path, arquivo)) for arquivo in sorted(arquivos_do_ano)],
    ignore_index=True
)

df_filtrado = filtrar_tickers_completos(df_concatenado)
df_final, tickers_selecionados = selecionar_tickers_representativos(df_filtrado, num_tickers)

# Matriz de retornos e dimensões
tickers = df_final['ticker'].unique()
df_pivot = df_final.pivot(index='week_start', columns='ticker', values='log_return').fillna(0)
log_returns = df_pivot[tickers].values
n_semanas, n_ativos = log_returns.shape

# ========== DEAP SETUP ========== #
creator.create("FitnessMulti", base.Fitness, weights=(2.0, -1.0))  # max retorno, min CVaR
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
avaliar_fn = partial(avaliar, log_returns=log_returns, beta=BETA)
feasible_fn = partial(feasible, n_ativos=n_ativos)
distance_fn = partial(distance, n_ativos=n_ativos)
mutate_fn = partial(custom_mutate, n_ativos=n_ativos)

toolbox.register("evaluate", DeltaPenalty(feasible_fn, (-1e4, 1e4), distance_fn)(avaliar_fn))
toolbox.register("mate", partial(tools.cxUniform, indpb=CXPB_INDPB))
toolbox.register("mutate", mutate_fn)
toolbox.register("select", tools.selNSGA2)

# ========== EXECUÇÃO ========== #
pop = toolbox.population(n=POP_SIZE)
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

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for i in range(1, len(offspring), 2):
        if np.random.rand() < cxpb:
            toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values
            del offspring[i].fitness.values

    for mutant in offspring:
        if np.random.rand() < mutpb:
            toolbox.mutate(
                mutant,
                indpb_float=params["indpb_float"],
                sigma=params["sigma"],
                indpb_bin=params["indpb_bin"]
            )
            del mutant.fitness.values


    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Injetar diversidade
    num_novos = int(0.60 * POP_SIZE)
    novos_inds = [toolbox.individual() for _ in range(num_novos)]


    fitnesses = toolbox.map(toolbox.evaluate, novos_inds)
    for ind, fit in zip(novos_inds, fitnesses):
        ind.fitness.values = fit


    # Coletar fitness da população atual
    retornos_geracao = [ind.fitness.values[0] for ind in pop]
    vars_geracao = [ind.fitness.values[1] for ind in pop]

    historico_retornos.append(retornos_geracao)
    historico_vars.append(vars_geracao)
    print(f"Geração {gen:3d} | Retorno Máx: {max(retornos_geracao):.6f} | Variância Mín: {min(vars_geracao):.6f}")

    pop[:] = tools.selNSGA2(pop + offspring + novos_inds, POP_SIZE)

# ========== RESULTADOS FINAIS ========== #
melhores_inds = tools.selBest(pop, k=5)

print("\n==================== MELHORES INDIVÍDUOS ====================\n")
for idx, ind in enumerate(melhores_inds, start=1):
    pesos = np.array(ind[:n_ativos])
    selecionados = np.array(ind[n_ativos:])
    pesos = pesos * selecionados
    soma_pesos = pesos.sum()
    if soma_pesos > 0:
        pesos = pesos / soma_pesos

    portfolio = {
        ticker: peso for ticker, peso, sel in zip(tickers, pesos, selecionados) if sel == 1
    }

    retorno, cvar = avaliar_fn(ind)

    print(f"Indivíduo {idx}")
    print(f"  Retorno ponderado (beta): {retorno:.6f}")
    print(f"  CVaR (alpha={CVaR_ALPHA:.2f}): {cvar:.6f}")
    print("  Portfólio (ativos e pesos normalizados):")
    for ticker, peso in portfolio.items():
        print(f"    {ticker}: {peso:.4f}")
    print("------------------------------------------------------")


# Converter para DataFrame
df_ret = pd.DataFrame(historico_retornos)
df_var = pd.DataFrame(historico_vars)

# Plotar evolução da média e dispersão dos objetivos
plt.figure()
df_ret.mean(axis=1).plot(label='Média Retorno')
df_ret.min(axis=1).plot(label='Melhor Retorno', linestyle='--')
plt.title('Evolução dos Retornos')
plt.xlabel('Geração')
plt.ylabel('Retorno')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("retorno_plot.png")

plt.figure()
df_var.mean(axis=1).plot(label='Média VAR')
df_var.min(axis=1).plot(label='Melhor VAR', linestyle='--')
plt.title('Evolução da VAR')
plt.xlabel('Geração')
plt.ylabel('VAR')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig("cvar_plot.png")

vars = [ind.fitness.values[1] for ind in pop]
returns = [ind.fitness.values[0] for ind in pop]

plt.figure(figsize=(8, 6))
plt.scatter(vars, returns, alpha=0.6, edgecolor='k')
plt.xlabel("VAR")
plt.ylabel("Retorno")
plt.title("Fronteira de Pareto - População Final")
plt.grid(True)
plt.tight_layout()
plt.savefig("pareto_final.png")
plt.show()

portfolios = []
for ind in pop:
    pesos = np.array(ind[:n_ativos])
    selecionados = np.array(ind[n_ativos:])
    pesos = pesos * selecionados
    if pesos.sum() > 0:
        pesos = pesos / pesos.sum()

    portfolio = {ticker: peso for ticker, peso, sel in zip(tickers, pesos, selecionados) if sel == 1}
    portfolios.append(portfolio)

df_portfolios = pd.DataFrame(portfolios).fillna(0)
df_portfolios.to_csv("populacao_final_portfolios.csv", index=False)