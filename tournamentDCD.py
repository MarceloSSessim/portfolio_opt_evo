import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools
from deap.tools import DeltaPenalty
import random
from utils import (
    selecionar_tickers_representativos,
    filtrar_tickers_completos,
    avaliar,
    feasible,
    distance,
    custom_mutate,
    get_adaptive_params
)
from functools import partial

# ========== PARÂMETROS ========== #
year = 2020
folder_path = "/mnt/c/Users/msses/Desktop/ETF/weekly_log_returns"
num_tickers = 30
BETA = 1
CVaR_ALPHA = 0.05
POP_SIZE = 700
N_GEN = 200
# Parâmetros de mutação e crossover adaptativos
INDPB_FLOAT_START = 0.8
INDPB_FLOAT_END = 0.6

SIGMA_START = 0.3
SIGMA_END = 0.08

INDPB_BIN_START = 0.7
INDPB_BIN_END = 0.6

CXPB_INDPB = 0.3

CX_START = 0.4
CX_END = 0.2

MUT_START = 0.8
MUT_END = 0.5

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
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # max retorno, min CVaR
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Genes contínuos e binários
toolbox.register("attr_float", lambda: np.random.uniform(0, 1))
toolbox.register("attr_bin", lambda: np.random.randint(0, 2))

def generate_individual():
    pesos = [np.random.uniform(0, 1) for _ in range(n_ativos)]
    binarios = [np.random.randint(0, 2) for _ in range(n_ativos)]
    return creator.Individual(pesos + binarios)

toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Funções adaptadas com partial
avaliar_fn = partial(avaliar, log_returns=log_returns, beta=BETA)
feasible_fn = partial(feasible, n_ativos=n_ativos)
distance_fn = partial(distance, n_ativos=n_ativos)
mutate_fn = partial(custom_mutate, n_ativos=n_ativos)

toolbox.register("evaluate", DeltaPenalty(feasible_fn, (-5, 5), distance_fn)(avaliar_fn))
toolbox.register("mate", partial(tools.cxUniform, indpb=CXPB_INDPB))
toolbox.register("mutate", mutate_fn)
toolbox.register("select", tools.selTournamentDCD)

# ========== EXECUÇÃO ========== #
pop = toolbox.population(n=POP_SIZE)
historico_retornos = []
historico_vars = []

# Avaliação inicial
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

    # ========== PROPORÇÕES ==========
    n_total = POP_SIZE
    n_tournament = int(0.1 * n_total)
    n_random = int(0.4 * n_total)
    n_offspring = n_total - n_tournament - n_random  # restante (50%)

    # ===== 1. Seleção via torneio =====
    pais_torneio = toolbox.select(pop, n_tournament)
    pais_torneio = [toolbox.clone(ind) for ind in pais_torneio]

    # ===== 2. Gerar offspring via crossover e mutação =====
    offspring = []
    while len(offspring) < n_offspring:
        p1, p2 = random.sample(pais_torneio, 2)
        c1, c2 = toolbox.clone(p1), toolbox.clone(p2)

        if np.random.rand() < cxpb:
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values

        if np.random.rand() < mutpb:
            c1, = toolbox.mutate(
                c1,
                indpb_float=params["indpb_float"],
                sigma=params["sigma"],
                indpb_bin=params["indpb_bin"]
            )
            del c1.fitness.values

        if np.random.rand() < mutpb:
            c2, = toolbox.mutate(
                c2,
                indpb_float=params["indpb_float"],
                sigma=params["sigma"],
                indpb_bin=params["indpb_bin"]
            )
            del c2.fitness.values

        offspring.extend([c1, c2])

    offspring = offspring[:n_offspring]


    # ===== 3. Gerar novos indivíduos aleatórios =====
    novos_inds = [toolbox.individual() for _ in range(n_random)]
    for ind in novos_inds:
        del ind.fitness.values

    # ===== 4. Avaliar indivíduos não avaliados =====
    todos_filhos = pais_torneio + offspring + novos_inds
    invalid_ind = [ind for ind in todos_filhos if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # ===== 5. Log da geração =====
    retornos_geracao = [ind.fitness.values[0] for ind in pop]
    cvars_geracao = [ind.fitness.values[1] for ind in pop]
    historico_retornos.append(retornos_geracao)
    historico_vars.append(cvars_geracao)

    # ===== 6. Selecionar nova geração com elitismo =====
    pop[:] = tools.selNSGA2(pop + todos_filhos, POP_SIZE)

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

    retorno, var = avaliar_fn(ind)

    print(f"Indivíduo {idx}")
    print(f"  Retorno ponderado (beta): {retorno:.6f}")
    print(f"  VAR: {var:.6f}")
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
df_var.mean(axis=1).plot(label='Média VaR')
df_var.min(axis=1).plot(label='Melhor VaR', linestyle='--')
plt.title('Evolução dos VaR')
plt.xlabel('Geração')
plt.ylabel('CVaR')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig("cvar_plot.png")
