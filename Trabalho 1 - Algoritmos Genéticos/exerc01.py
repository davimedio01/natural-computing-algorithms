"""
Trabalho 1 - Algoritmos Genéticos
Data de Entrega: 20/04/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np               # Matrizes e Funções Matemáticas

#####################################################
#              Funções de Avaliação                 #
#####################################################

# Função do Exercício 01: Aptidão bitstring [1 1 1 1 0 1 1 0 1 1 1 1]


def f1(individual_bitstring: np.ndarray):
    """Função de Aptidão representado 
    pela bitstring [1 1 1 1 0 1 1 0 1 1 1 1].

    Args:
        individual_bitstring (np.ndarray[int, ...]): vetor de bitstring do indivíduo
        
    Returns:
        dist (int): valor da aptidão baseado na distância de Hamming
    """

    # Criação da bitstring de verificação do exercício 01
    padrão_bitstring = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1])

    # Calculo da Distância de Hamming para encontrar o valor total da aptidão para bit 0
    hamming_dist = np.sum(individual_bitstring != padrão_bitstring)

    return (individual_bitstring.shape[0] - hamming_dist)


#####################################################
#          Algoritmo Genético: Bitstring            #
#####################################################

########################################
#              0 - Geração             #
########################################

# Geração Aleatória da População
def generate_population(
    bitstring_size=12,
    population_size=10,
):
    """Função responsável por gerar uma nova população,
    em bitstring (vetor de bits inteiros de 0/1).

    Args:       
        bitstring_size (int): tamanho do bitstring (padrão: 12)
        population_size (int): tamanho da população (padrão: 10)
        
    Returns:
        population (np.ndarray[[int, ...], ...]): população inicial gerada
    """
    population = np.random.randint(2, size=(population_size, bitstring_size))
    return population


########################################
#              1 - Seleção             #
########################################

# Seleção por Roleta
def __roulette_selection(
    cur_population: np.ndarray,
    fitness: np.ndarray,
):
    """Seleção de indivíduo pelo método da Roleta.

    Args:
        cur_population (np.ndarray[[int, ...], ...]): vetor contendo a população atual para seleção
        fitness (np.ndarray[int, ...]): vetor contendo todos os valores de aptidão da população
        
    Returns:
        new_population (np.ndarray[[int, ...], ...]): vetor contendo a nova população selecionada
        selected_fitness (np.ndarray[int, ...]): vetor contendo todos os valores de aptidão da população selecionada
    """

    # Determinar a porção da roleta para cada indivíduo no intervalo [start; end]
    proportions = fitness / (np.sum(fitness))

    # Posição dos indivíduos selecionados aleatoriamente com base nas porções
    idx_selected = np.random.choice(
        cur_population.shape[0], size=cur_population.shape[0], p=proportions)

    return cur_population[idx_selected], fitness[idx_selected]


# Seleção por Torneio
def __tournament_selection(
    cur_population: np.ndarray,
    fitness: np.ndarray,
    size=3,
):
    """Seleção de indivíduo pelo método do Torneio.

    Args:
        cur_population (np.ndarray[[int, ...], ...]): vetor contendo a população atual para seleção
        fitness (np.ndarray[int, ...]): vetor contendo todos os valores de aptidão da população
        size (int): número de indivíduos selecionados aleatóriamente (padrão: 3)
        
    Returns:
        new_population (np.ndarray[[int, ...], ...]): vetor contendo a nova população selecionada
        selected_fitness (np.ndarray[int, ...]): vetor contendo todos os valores de aptidão da população selecionada
    """

    # Criação do vetor para nova população, com base no tamanho da população atual
    new_population = np.empty_like(cur_population)
    selected_fitness = np.empty_like(fitness)

    # Percorrendo o vetor da população atual (número de linhas)
    for i in range(cur_population.shape[0]):
        # Escolha aleatória dos indivíduos candidatos
        idx_candidates = np.random.choice(
            cur_population.shape[0], size, replace=False)

        # Escolha do vencedor com base no MAIOR valor obtido
        idx_winner = np.argmax(fitness[idx_candidates])

        # Salvando o vencedor na nova população
        new_population[i] = cur_population[idx_candidates[idx_winner]]
        selected_fitness[i] = fitness[idx_candidates[idx_winner]]

    return new_population, selected_fitness


########################################
#           2 - Reprodução             #
########################################

# Crossover
def __crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate=0.8,
):
    """Aplicação de crossover entre dois indivíduos.

    Args:
        parent1 (np.ndarray[int, ...]): vetor representando o primeiro indivíduo
        parent2 (np.ndarray[int, ...]): vetor representando o segundo indivíduo
        crossover_rate (float): float que representa a taxa acontecimento de crossover (padrão: 0.8)
        
    Returns:
        child1 (np.ndarray[int, ...]): vetor representando o primeiro filho
        child2 (np.ndarray[int, ...]): vetor representando o segundo filho
    """

    # Para ocorrer o crossover, um número aleatório deve ser menor ou igual a taxa
    if np.random.rand() <= crossover_rate:
        # Sorteia um ponto de corte
        crossover_point = np.random.randint(0, parent1.shape[0])

        # Realização do crossover
        child1 = np.concatenate(
            (parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate(
            (parent2[:crossover_point], parent1[crossover_point:]))
    else:
        # Não ocorrência de crossover, apenas mantém os pais
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)

    return child1, child2


# Mutação
def __mutation(
    individual: np.ndarray,
    mutation_rate=0.2,
):
    """Aplicação de mutação em um indivíduo.

    Args:
        individual (np.ndarray[int, ...]): vetor representando o indivíduo a sofrer mutação
        mutation_rate (float): float que representa a taxa de mutação (padrão: 0.2)
        
    Returns:
        mutant (np.ndarray[int, ...]): vetor representando o indivíduo com mutação
    """
    # Cria o vetor inicial do mutante como cópia do indivíduo
    mutant = np.copy(individual)

    # Percorrendo cada posição do indivíduo
    for i in range(np.size(mutant)):
        # Gera um número aletório e verifica com a taxa de mutação
        if np.random.rand() <= mutation_rate:
            mutant[i] = 1 - mutant[i]

    return mutant


# Reprodução
def __reproduction(
    selected_population: np.ndarray,
    selected_fitness: np.ndarray,
    elitism=False,
    elite_size: int = 3,
    crossover_rate=0.8,
    mutation_rate=0.2,
):
    """Reprodução de uma determinada população, em bitstring, 
    considerando crossover e mutação.

    Args:
        selected_population (np.ndarray[[int, ...], ...]): vetor com a população selecionada
        selected_fitness (np.ndarray[float, ...]): vetor contendo todos os valores de aptidão da população
        elitism (bool): considerar ou não o elitismo (padrão: False)
        elite_size (int, opcional se 'elitism=False'): quantidade de indivíduos para elitismo (padrão: 3)
        crossover_rate (float): taxa de crossover (padrão: 0.8)
        mutation_rate (float): taxa de mutação (padrão: 0.2)
        
    Returns:
        new_population (np.ndarray[[int, ...], ...]): vetor com a nova população
    """

    # Seleção de todos os pais para reprodução
    parents = selected_population

    # Elitismo
    if elitism:
        # Seleção dos indivíduos de elite (primeiros 'elite_size' da população previamente ordenada...)
        elite = selected_population[np.argsort(
            selected_fitness)][::-1][:elite_size]

        # Seleção dos indivíduos sem a elite, para geração de filhos
        parents = selected_population[np.argsort(
            selected_fitness)][::-1][elite_size:]

    # Criação de novos indivíduos com crossover e mutação
    children = []
    for i in range(0, parents.shape[0]-1, 2):
        # Percorre a população em dois a dois, selecionando pares contínuos
        parent1, parent2 = parents[i], parents[i + 1]

        # Fase de crossover
        child1, child2 = __crossover(parent1, parent2, crossover_rate)

        # Fase de mutação
        child1 = __mutation(child1, mutation_rate)
        child2 = __mutation(child2, mutation_rate)

        # Adiciona os filhos na nova população
        children.append(child1)
        children.append(child2)

    if elitism:
        # Caso o número da população seja ímpar, adiciona o último indivíduo
        if ((parents.shape[0] - elite_size) % 2 == 1):
            children.append(parents[-1])
    else:
        # Caso o número da população seja ímpar, adiciona o último indivíduo
        if (parents.shape[0] % 2 == 1):
            children.append(parents[-1])

    # Adicionando a elite e os filhos gerados
    new_population = np.concatenate(
        (elite, children)) if elitism else np.array(children)

    return new_population


########################################
#         3 - Função Principal         #
########################################

# Algoritmo Genético
def genetic_algorithm(
    initial_population: np.ndarray,
    fitness_func: callable,
    selection_func: callable = __roulette_selection,
    max_gen=10000,
    max_patience=100,
    size_tournament: int = 3,
    elitism=False,
    elite_size: int = 3,
    crossover_rate=0.8,
    mutation_rate=0.2,
):
    """Aplicação do Algoritmo Genético, a partir
    de uma população de bitstring, para min/max de uma 
    função multivariável.

    Seleção por Torneio ou por Roleta.
    -> Definida pelo parâmetro 'selection_func'.
       -> Torneio: __tournament_selection(cur_population: np.ndarray, fitness: np.ndarray, size=3).
       O 'size' é atribuído pelo parâmetro 'size_tournament'.
       -> Roleta (padrão): __roulette_selection(cur_population: np.ndarray, fitness: np.ndarray).

    Args:
        initial_population (np.ndarray[[int, ...], ...]): matriz de bitstrings da população inicial
        fitness_func (callable): função de avaliação de aptidão
        selection_func (callable): função de seleção
        max_gen (int): número máximo de gerações possíveis (padrão: 10000)
        max_patience (int): número máximo de iterações em que não houve melhora (padrão: 100)
        size_tournament (int, opcional se 'selection_func=__roulette_selection'): número de indivíduos selecionados aleatóriamente para o torneio (padrão: 3)
        elitism (bool): considerar ou não o elitismo (padrão: False)
        elite_size (int, opcional se 'elitism=False'): quantidade de indivíduos para elitismo (padrão: 3)
        crossover_rate (float): taxa de crossover (padrão: 0.8)
        mutation_rate (float): taxa de mutação (padrão: 0.2)

    Returns:
        best_global_population (np.ndarray[[int, ...], ...]): lista com as melhores populações globais obtidas ao decorrer das gerações
        best_local_population (np.ndarray[[int, ...], ...]): lista com as melhores populações locais obtidas ao decorrer das gerações
        best_global_fitness (np.ndarray[float, ...]): lista com as melhores aptidões globais obtidas ao decorrer das gerações
        best_local_fitness (np.ndarray[float, ...]): lista com as melhores aptidões locais obtidas ao decorrer das gerações
        all_mean_fitness (np.ndarray[float, ...]): lista com a média das aptidões obtidas ao decorrer das gerações
        generation (int): número de gerações decorridas
    
    Notes:
        - O retorno das variáveis abaixo está comentado para otimização, use por sua conta e risco!
            all_population (np.ndarray[[int, ...], ...]): lista com todas as populações obtidas, sendo a última a melhor possível
            all_fitness (np.ndarray[[int, ...], ...]): lista com todas as aptidões obtidas, sendo a última a melhor possível
    """

    print(f"{'-'*50}")
    print(f"{'Algoritmo Genético':^50}")
    print(f"{'-'*50}")

    # Avaliação da população inicial
    cur_fitness = np.array([fitness_func(individual)
                           for individual in initial_population])

    # Recuperando o melhor da população inicial e definindo valores iniciais
    best_global_fitness = [np.max(cur_fitness)]
    best_local_fitness = [np.max(cur_fitness)]
    all_mean_fitness = [np.mean(cur_fitness)]
    best_global_population = np.array([individual for individual in
                                [initial_population[np.argmax(cur_fitness)]]])
    best_local_population = np.array([individual for individual in
                                [initial_population[np.argmax(cur_fitness)]]])
    #! ONLY IF WANTED
    # all_population = np.array(
    #     [individual for individual in initial_population])
    # all_fitness = np.array([cur_fitness])

    # Início feito a partir da população inicial
    cur_population = initial_population

    # Percorrendo as gerações
    generation = 1
    patience = 1
    while (generation < max_gen and 
           patience != max_patience and 
           (initial_population.shape[1] not in best_global_fitness)):
        # Aumenta o número da geração atual
        generation = generation + 1

        # Fase de Seleção
        cur_population, cur_fitness = selection_func(cur_population=cur_population, fitness=cur_fitness, size=size_tournament)if selection_func == __tournament_selection else selection_func(
            cur_population=cur_population, fitness=cur_fitness)

        # Fase de Reprodução
        cur_population = __reproduction(
            selected_population=cur_population,
            selected_fitness=cur_fitness,
            elitism=elitism,
            elite_size=elite_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate)

        # Fase de Avaliação
        cur_fitness = np.array([
            fitness_func(individual)
            for individual in cur_population])

        # Verificação da Paciência (otimização)
        if np.max(cur_fitness) == np.max(best_global_fitness):
            patience = patience + 1

        # Atualizando valores globais
        if np.max(cur_fitness) > np.max(best_global_fitness):
            patience = 1
            best_global_fitness = np.append(
                best_global_fitness,
                np.array([np.max(cur_fitness)]),
                axis=0)
            best_global_population = np.append(
                best_global_population,
                [individual for individual in [
                    cur_population[np.argmax(cur_fitness)]]],
                axis=0)
        else:
            best_global_fitness = np.append(
                best_global_fitness,
                [best_global_fitness[-1]],
                axis=0)
            best_global_population = np.append(
                best_global_population,
                [best_global_population[-1]],
                axis=0)
        
        # Independente se conseguiu melhor resultado, salva nas listas de retorno para pós-visualização
        best_local_fitness = np.append(
            best_local_fitness,
            np.array([np.max(cur_fitness)]),
            axis=0)

        all_mean_fitness = np.append(
            all_mean_fitness,
            np.array([np.mean(cur_fitness)]),
            axis=0)

        best_local_population = np.append(
            best_local_population,
            [individual for individual in [
                cur_population[np.argmax(cur_fitness)]]],
            axis=0)

        #! ONLY IF WANTED
        # all_population = np.append(
        #     all_population,
        #     [individual for individual in cur_population],
        #     axis=0)

        # all_fitness = np.append(
        #     all_fitness,
        #     [cur_fitness],
        #     axis=0)

    print(f"Geração {generation}")
    print(f"Melhor Aptidão: {best_global_fitness[-1]}")
    print(f"Melhor Indivíduo: {best_global_population[-1]}")

    print(f"{'-'*50}")
    print(f"{'Fim do Algoritmo Genético':^50}")
    print(f"{'-'*50}")

    return best_global_population, best_local_population, best_global_fitness, best_local_fitness, all_mean_fitness, generation


########################################
#            Experimentos              #
########################################

# Criar gráfico de um dos experimentos
def plot_experiment(
    filename: str,
    alg_acronym: str,
    type_plot: str,
    num_cycle: int,
    num_experiment: int,
    population_size: int,
    all_mean_fitness: np.ndarray,
    best_fitness: np.ndarray,
    crossover_rate=-1.0,
    mutation_rate=-1.0,
):
    """Cria o gráfico de uma execução do Algoritmo Genético,
    no formato 'Melhor Aptidão x Geração' 
    (Melhor Aptidão por Geração).
    
    Args:
        filename (str): nome do arquivo/exercicio (ex: 'ex01')
        alg_acronym (str): sigla do algoritmo executado (ex: 'HC' - Hill-Climbing)
        type_plot (str): plot normal ('normal') ou pela logarítimca ('log') (padrão: 'normal')
        num_cycle (int): número do ciclo de execução
        num_experiment (int): número do experimento dentro do ciclo
        population_size (int): tamanho da população
        all_mean_fitness (np.ndarray[[int, ...], ...]): lista com todas as aptidões obtidas, sendo a última a melhor possível
        best_fitness (np.ndarray[[int], ...]): lista com as melhores aptidões obtidas ao decorrer das gerações
        crossover_rate (float, opcional): taxa de crossover (padrão: -1.0)
        mutation_rate (float, opcional): taxa de mutação (padrão: -1.0)
        
    Notes:
        Cria um arquivo de imagem do gráfico com o seguinte nome: {alg_acronym}_ciclo{num_cycle}_exp{num_experiment}_pop{population_size}_cr{crossover_text}_mt{mutation_text}.png
        Salva em um subdiretório da pasta local com o nome: {filename}
    """

    # Pacote de plotar gráficos de listas NumPy
    import matplotlib.pyplot as plt

    # Defininido o sub-diretório dos dados
    import os
    actual_dir = os.path.dirname(__file__)
    sub_directory = os.path.join(actual_dir, f'{filename}/')
    os.makedirs(sub_directory, exist_ok=True)

    # Definindo o nome do arquivo
    crossover_text = "{:02d}".format(
        int(crossover_rate*100)) if crossover_rate > 0 else ''
    mutation_text = "{:02d}".format(
        int(mutation_rate*100)) if mutation_rate > 0 else ''
    plot_name = f'{alg_acronym}_ciclo{num_cycle}_exp{num_experiment}_pop{population_size}_cr{crossover_text}_mt{mutation_text}.png'

    # Definindo os textos (nomes) do gráfico
    plt.title(
        str(f"Melhor: {best_fitness[-1]} Média: {np.mean(all_mean_fitness)}"),
        loc='center')
    plt.xlabel('Geração', loc='center')
    plt.ylabel('Aptidão', loc='center')

    # Plotando o gráfico com base nos valores
    if type_plot == 'normal':
        plt.plot(best_fitness,  label='Melhor Aptidão',
                 marker='.', linewidth=0.5)
        plt.plot(all_mean_fitness, label='Aptidão Média',
                 marker='*', linewidth=0.5)
    elif type_plot == 'log':
        plt.semilogy(best_fitness,  label='Melhor Aptidão',
                     marker='.', linewidth=0.5)
        plt.semilogy(all_mean_fitness, label='Aptidão Média',
                     marker='*', linewidth=0.5)

    # Adiciona legenda
    plt.legend()

    # Plota e salva o gráfico em um arquivo
    plt.savefig(os.path.join(sub_directory, plot_name))

    # Encerra as configurações do gráfico
    plt.close()



# Manipular arquivos CSV
def csv_table_experiment(
    filename: str,
    alg_acronym: str,
    type_exp: str,
    num_cycle: int,
    rows: np.ndarray
):
    """Escreve os experimentos em um arquivo CSV para futura análise.
    Necessário três tabelas por experimento: aptidão, gerações e tempo de execução.

    Args:
        filename (str): nome do arquivo/exercicio (ex: 'ex01')
        alg_acronym (str): sigla do algoritmo executado (ex: 'HC' - Hill-Climbing)
        type_exp (str): tipo da tabela (ex: 'aptidao')
            -> Utilize: 'aptidao', 'geracao', 'tempo'
        num_cycle (int): número do ciclo de execução
        rows (np.ndarray[[dados, ...], ...]): lista com os dados das linhas no total (ex: [['10', '0.1'], ['20', '0.2'])
        
    Notes:
        Cria um arquivo csv da tabela com o seguinte nome: {alg_acronym}_{type_exp}_ciclo{num_cycle}.csv  
        Salva em um subdiretório da pasta local com o nome: {filename}
    """

    # Defininido o sub-diretório dos dados
    import os
    actual_dir = os.path.dirname(__file__)
    sub_directory = os.path.join(actual_dir, f'{filename}/')
    os.makedirs(sub_directory, exist_ok=True)

    # Definindo o nome do arquivo
    table_name = f'{alg_acronym}_{type_exp}_ciclo{num_cycle}.csv'

    # Definindo o título da tabela com base no tipo de experimento
    table_title = 'Tamanho da População,Taxa de Crossover,Taxa de Mutação,Média,Desvio Padrão,Mediana,'
    if type_exp == 'aptidao':
        table_title = table_title + 'Melhor,Pior'
    elif type_exp == 'geracao' or type_exp == 'tempo':
        table_title = table_title + 'Mínimo,Máximo'

    # Escrevendo o arquivo com o título
    np.savetxt(fname=os.path.join(sub_directory, table_name), X=rows, fmt='%.4f', header=table_title,
               delimiter=',', comments='', encoding='UTF-8')


def main():
    """Função principal do programa
    
    Descrição dos Experimentos:
        - Fixa taxa de crossover e mutação para selecionar um tamanho de população para os próximos
        - Varia em par de taxas de crossover e mutação
        - Varia só com crossover (mutação=0.0)
        - Varia só com mutação (crossover=0.0)
        
    Restrições:
        - 4 ciclos executados, sendo um para cada ponto na descrição
        - cada ciclo: 25 vezes de execução do algoritmo
        - tam. bitstrings: 12
        - num. máx. gerações: 10000
        - num. máx. 'paciência': 50
        - quatro tamanhos de população inicial: 20, 30, 40, 50
        - método de seleção: roleta
        - crossover: 0.5, 0.6, 0.7, 0.8
        - mutação: 0.1, 0.2, 0.3, 0.4
        - sem elitismo
    
    Formato dos dados salvos:
        - Gráfico (por ciclo): apenas um para cada população, do tipo "melhor aptidão x geração"
        - Tabelas (por ciclo): aptidão, gerações e tempo de execução
    """

    #! [Debug] Definição de saída para mostrar as matrizes por completo no console se necessário.
    np.set_printoptions(threshold=np.inf)

    # Biblioteca para capturar o tempo de execução
    from time import time

    # Nome para arquivos de saída
    filename = 'ex01'

    # Definindo as restrições descritas
    fitness_func = f1
    max_cycle = 4
    max_exp_per_cycle = 25
    bitstring_size = 12
    max_gen = 10000
    max_patience = 100
    initial_population = np.array([20, 30, 40, 50], dtype=np.int64)
    selection_func = __roulette_selection
    crossover_vals = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    mutation_vals = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    elitism = False
    elite_size = 3

    ########################################
    #!      Início dos Experimentos       !#
    ########################################

    # Realização de cada ciclo
    for num_cycle in range(1, max_cycle + 1):
        # Armazenar as melhores aptidões por experimento de TODAS as populações
        # - Formato: [[tam. pop., crossover, mutação, média, mediana, melhor, pior], ...]
        cycle_best_fitness = []

        # Armazenar as melhores gerações por experimento de TODAS as populações
        # - Formato: [[tam. pop., crossover, mutação, média, mediana, mínimo, máximo], ...]
        cycle_best_generation = []

        # Armazenar o tempo de execução por experimento de TODAS as populações
        # - Formato: [[tam. pop., crossover, mutação, média, mediana, mínimo, máximo], ...]
        cycle_exec_time = []

        # Definição das taxas de crossover e mutação dos experimentos descritos
        if num_cycle == 1:
            # Ciclo 1: apenas o primeiro valor
            crossover_idx = np.zeros(crossover_vals.shape[0], dtype=np.int32)
            mutation_idx = np.zeros(mutation_vals.shape[0], dtype=np.int32)
        elif num_cycle == 2:
            # Ciclo 2: varia em pares
            crossover_idx = np.arange(crossover_vals.shape[0], dtype=np.int32)
            mutation_idx = np.arange(mutation_vals.shape[0], dtype=np.int32)
        elif num_cycle == 3:
            # Ciclo 3: somente crossover, sem mutação
            crossover_idx = np.arange(crossover_vals.shape[0], dtype=np.int32)
            mutation_idx = np.array(
                [-1] * mutation_vals.shape[0], dtype=np.int32)
        elif num_cycle == 4:
            # Ciclo 4: somente mutação, sem crossover
            crossover_idx = np.array(
                [-1] * crossover_vals.shape[0], dtype=np.int32)
            mutation_idx = np.arange(mutation_vals.shape[0], dtype=np.int32)

        # Percorre cada tamanho de população
        for rate_idx, population_size in enumerate(initial_population):
            # Armazenar as melhores aptidões de uma população
            experiment_best_fitness = []

            # Armazenar as melhores gerações de uma população
            experiment_best_generation = []

            # Armazenar o tempo de execução de uma população
            experiment_exec_time = []

            # Definindo as taxas de crossover e mutação dos experimentos
            crossover_rate = crossover_vals[crossover_idx[rate_idx]
                                            ] if crossover_idx[rate_idx] != -1 else -1.0
            mutation_rate = mutation_vals[mutation_idx[rate_idx]
                                          ] if mutation_idx[rate_idx] != -1 else -1.0

            # Obtendo um valor aletório para plotar um gráfico de um dos experimentos
            plot_rand_num = np.random.randint(1, max_exp_per_cycle + 1)

            # print(rate_idx, ' ', population_size, ' ', crossover_rate,' ', mutation_rate)

            # Percorre o número máximo de experimentos
            for num_experiment in range(1, max_exp_per_cycle + 1):
                # Registra o tempo inicial de execução
                start_timer = time()

                # Geração de população inicial aleatória (estocástico)
                gen_population = generate_population(
                    bitstring_size=bitstring_size,
                    population_size=population_size
                )

                # Aplicação do algoritmo
                best_global_population, best_local_population, best_global_fitness, best_local_fitness, all_mean_fitness, generation = genetic_algorithm(
                    initial_population=gen_population,
                    fitness_func=fitness_func,
                    selection_func=selection_func,
                    max_gen=max_gen,
                    max_patience=max_patience,
                    elitism=elitism,
                    elite_size=elite_size,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate
                )

                # Registra o tempo total de execução do algoritmo
                total_time = time() - start_timer

                # Salvando os dados nas listas
                experiment_best_fitness.append(np.max(best_local_fitness))
                experiment_best_generation.append(generation)
                experiment_exec_time.append(total_time)

                # Gerando o gráfico do experimento escolhido aleatoriamente
                if num_experiment == plot_rand_num:
                    # print(best_fitness, '\n', all_mean_fitness)

                    # Plota o gráfico
                    plot_experiment(
                        filename=filename,
                        alg_acronym='AG',
                        type_plot='normal',
                        num_cycle=num_cycle,
                        num_experiment=num_experiment,
                        population_size=population_size,
                        all_mean_fitness=all_mean_fitness,
                        best_fitness=best_global_fitness,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate
                    )

            # Salvando os dados nas listas
            cycle_best_fitness.append([
                population_size,
                crossover_rate if crossover_rate > 0.0 else 0.0,
                mutation_rate if mutation_rate > 0.0 else 0.0,
                np.mean(experiment_best_fitness),
                np.std(experiment_best_fitness),
                np.median(experiment_best_fitness),
                np.max(experiment_best_fitness),
                np.min(experiment_best_fitness)
            ])
            cycle_best_generation.append([
                population_size,
                crossover_rate if crossover_rate > 0.0 else 0.0,
                mutation_rate if mutation_rate > 0.0 else 0.0,
                np.mean(experiment_best_generation),
                np.std(experiment_best_generation),
                np.median(experiment_best_generation),
                np.min(experiment_best_generation),
                np.max(experiment_best_generation)
            ])
            cycle_exec_time.append([
                population_size,
                crossover_rate if crossover_rate > 0.0 else 0.0,
                mutation_rate if mutation_rate > 0.0 else 0.0,
                np.mean(experiment_exec_time),
                np.std(experiment_exec_time),
                np.median(experiment_exec_time),
                np.min(experiment_exec_time),
                np.max(experiment_exec_time)
            ])

            # print(experiment_best_fitness, '\n', experiment_best_generation, '\n', experiment_exec_time)

        # print(cycle_best_fitness, '\n', cycle_best_generation, '\n', cycle_exec_time)

        # Salva os resultados nos arquivos específicos

        # Tabela de Aptidão
        csv_table_experiment(
            filename=filename,
            alg_acronym='AG',
            type_exp='aptidao',
            num_cycle=num_cycle,
            rows=cycle_best_fitness
        )

        # Tabela de Geração
        csv_table_experiment(
            filename=filename,
            alg_acronym='AG',
            type_exp='geracao',
            num_cycle=num_cycle,
            rows=cycle_best_generation
        )

        # Tabela de Tempo de Execução
        csv_table_experiment(
            filename=filename,
            alg_acronym='AG',
            type_exp='tempo',
            num_cycle=num_cycle,
            rows=cycle_exec_time
        )

        # Ciclo 1: variação de tam. população, mesma taxas
        if num_cycle == 1:
            # Recuperando todas as médias de gerações, em ordem de tam. população
            all_mean_generation = np.array(cycle_best_generation)[:, 3]

            # Seleciona o melhor tam. população após término do primeiro ciclo
            idx_min_mean_generation = np.argmin(all_mean_generation)

            # Reescreve o vetor de tamanho da população com a melhor selecionada
            initial_population = np.array(
                [initial_population[idx_min_mean_generation]] * initial_population.shape[0], dtype=np.int64)


if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
