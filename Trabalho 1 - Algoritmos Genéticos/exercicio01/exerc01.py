"""
Trabalho 1 - Algoritmos Genéticos
Data de Entrega: 20/04/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np               # Matrizes e Funções Matemáticas
import matplotlib.pyplot as plt  # Plotar Gráficos


#####################################################
#              Funções de Avaliação                 #
#####################################################

# Função do Exercício 01: Aptidão bitstring [1 1 1 1 0 1 1 0 1 1 1 1]
def f1_fitness(individual_bitstring):
    """Função de Aptidão representado 
    pela bitstring [1 1 1 1 0 1 1 0 1 1 1 1].

    Args:
        individual_bitstring: vetor de bitstring do indivíduo
        
    Returns:
        hamming_dist: valor da aptidão baseado na distância de Hamming
    """

    # Criação da bitstring de verificação do exercício 01
    default_bitstring = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1])

    # Calculo da Distância de Hamming para encontrar o valor total da aptidão para bit 0
    hamming_dist = np.sum(individual_bitstring != default_bitstring)

    return (np.size(individual_bitstring) - hamming_dist)


#####################################################
#          Algoritmo Genético: Bitstring            #
#####################################################

########################################
#              0 - Geração             #
########################################

# Geração Aleatória da População
def generate_population(size_bitstring=12, size_population=10):
    """Função responsável por gerar uma nova população,
    em bitstring (vetor de bits).

    Args:       
        size_bitstring: tamanho do bitstring (padrão: 12)
        size_population: tamanho da população (padrão: 10)
        
    Returns:
        population: população inicial gerada
    """
    population = np.random.randint(2, size=(size_population, size_bitstring))
    return population


########################################
#              1 - Seleção             #
########################################

# Seleção por Roleta
def roulette_selection(cur_population: np.ndarray, all_fitness: np.ndarray, start=0, end=360):
    """Seleção de indivíduo pelo método da Roleta.

    Args:
        cur_population: vetor contendo a população atual para seleção
        all_fitness: vetor contendo todos os valores de aptidão da população
        start: início inclusivo do intervalo da roleta (padrão: 0)
        end: fim exclusivo do intervalo da roleta (padrão: 360)
        
    Returns:
        new_population: vetor contendo a nova população selecionada
    """

    # Somar a aptidão de todos indivíduos
    sum_all_fitness = np.sum(all_fitness)

    # Determinar a porção da roleta para cada indivíduo no intervalo [start; end]
    prob_roulette = np.array(((end - start) * all_fitness) / (sum_all_fitness + 1e-8))
    
    # Determinar os intervalos da roleta para cada indivíduo
    intervals_roulette = np.concatenate(([start], np.cumsum(prob_roulette)))
    intervals_roulette[-1] = end
    
    # Geração de valores aleatórios
    rand_vals = np.random.uniform(
        low=start, high=end, size=cur_population.shape[0])
    
    # Seleção dos indivíduos com base na roleta (índices)
    idx_selected = np.digitize(rand_vals, intervals_roulette) - 1
    
    return cur_population[idx_selected]


# Seleção por Torneio
def tournament_selection(cur_population: np.ndarray, all_fitness: np.ndarray, size=3):
    """Seleção de indivíduo pelo método do Torneio.

    Args:
        cur_population: vetor contendo a população atual para seleção
        all_fitness: vetor contendo todos os valores de aptidão da população
        size: número de indivíduos selecionados aleatóriamente (padrão: 3)
        
    Returns:
        new_population: vetor contendo a nova população selecionada
    """

    # Criação do vetor para nova população, com base no tamanho da população atual
    new_population = np.empty_like(cur_population)

    # Percorrendo o vetor da população atual (número de linhas)
    for i in range(cur_population.shape[0]):
        # Escolha aleatória dos indivíduos candidatos
        candidates = np.random.choice(
            cur_population.shape[0], size, replace=False)

        # Escolha do vencedor com base no maior valor obtido
        winner = np.argmax(all_fitness[candidates])

        # Salvando o vencedor na nova população
        new_population[i] = cur_population[candidates[winner]]

    return new_population


########################################
#           2 - Reprodução             #
########################################

# Crossover
def crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate=0.8):
    """Aplicação de crossover entre dois indivíduos.

    Args:
        parent1: vetor representando o primeiro indivíduo
        parent2: vetor representando o segundo indivíduo
        crossover_rate: float que representa a taxa acontecimento de crossover (padrão: 0.8)
        
    Returns:
        child1: vetor representando o primeiro filho
        child2: vetor representando o segundo filho
    """

    # Para ocorrer o crossover, um número aleatório deve ser menor ou igual a taxa
    if np.random.rand() <= crossover_rate:
        # Sorteia um ponto de corte
        crossover_point = np.random.randint(0, np.size(parent1))

        # Realização do crossover
        child1 = np.concatenate(
            (parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate(
            (parent2[:crossover_point], parent1[crossover_point:]))
    else:
        # Não ocorrência de crossover, apenas mantém os pais
        child1, child2 = np.copy(parent1), np.copy(parent2)

    return child1, child2


# Mutação
def mutation(individual: np.ndarray, mutation_rate=0.2):
    """Aplicação de mutação em um indivíduo.

    Args:
        individual: vetor representando o indivíduo a sofrer mutação
        mutation_rate: float que representa a taxa de mutação (padrão: 0.2)
        
    Returns:
        mutant: vetor representando o indivíduo com mutação
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
def reproduction(selected_population: np.ndarray, elitism=False, elite_size=3, crossover_rate=0.8, mutation_rate=0.2):
    """Reprodução de uma determinada população, em bitstring, 
    considerando crossover e mutação.

    Args:
        selected_population: vetor com a população selecionada
        elitism: considerar ou não o elitismo (padrão: False)
        elite_size: quantidade de indivíduos para elitismo (padrão: 3)
        crossover_rate: taxa de crossover (padrão: 0.8)
        mutation_rate: taxa de mutação (padrão: 0.2)
        
    Returns:
        new_population: vetor com a nova população
    """

    # Seleção de todos os pais para reprodução
    parents = selected_population

    if elitism:
        # Seleção dos indivíduos de elite (últimos 'elite_size' da população previamente ordenada...)
        elite = selected_population[:elite_size]

        # Seleção dos indivíduos sem a elite, para geração de filhos
        parents = selected_population[elite_size:]

    # Criação de novos indivíduos com crossover e mutação
    children = []
    for i in range(0, parents.shape[0]-1, 2):
        # Percorre a população em dois a dois, selecionando pares contínuos
        parent1, parent2 = parents[i], parents[i + 1]

        # Fase de crossover
        child1, child2 = crossover(parent1, parent2, crossover_rate)

        # Fase de mutação
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)

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

# Algoritmo Genético, por Seleção da Roleta
def genetic_algorithm_roulette(
    initial_population: np.ndarray,
    fitness_func,
    max_gen=10000,
    start_roulette=0,
    end_roulette=360,
    elitism=False,
    elite_size=3,
    crossover_rate=0.8,
    mutation_rate=0.2
):
    """Aplicação do Algoritmo Genético, a partir
    de uma população de bitstring. 
    Uso da Seleção por Roleta.

    Args:
        initial_population: matriz de bitstrings da população inicial
        fitness_func: função de avaliação de aptidão
        max_gen: número máximo de gerações possíveis (padrão: 10000)
        start_roulette: início exclusivo do intervalo da roleta (padrão: 0)
        end_roulette: fim inclusivo do intervalo da roleta (padrão: 360)
        elitism: considerar ou não o elitismo (padrão: False)
        elite_size: quantidade de indivíduos para elitismo (padrão: 3)
        crossover_rate: taxa de crossover (padrão: 0.8)
        mutation_rate: taxa de mutação (padrão: 0.2)
        
    Returns:
        best_population: matriz com a melhor população obtida
        best_fitness: matriz contendo as aptidões da melhor população
        generation: número de gerações decorridas
    """

    print(f"{'-'*50}")
    print(f"{'Algoritmo Genético':^50}")
    print(f"{'-'*50}")

    # Avaliação da população inicial
    cur_fitness = np.array([fitness_func(individual)
                           for individual in initial_population])

    # Percorrendo as gerações
    generation = 1
    cur_population = initial_population
    while (generation <= max_gen and (initial_population.shape[1] not in cur_fitness)):
        # Fase de Seleção
        cur_population = roulette_selection(
            cur_population, cur_fitness, start_roulette, end_roulette)

        # Fase de Reprodução
        cur_population = reproduction(
            cur_population, elitism, elite_size, crossover_rate, mutation_rate)

        # Fase de Avaliação
        cur_fitness = np.array([fitness_func(individual)
                               for individual in cur_population])

        # Aumenta o número da geração atual
        generation = generation + 1

    # Retornando a melhor população
    best_population = cur_population
    best_fitness = cur_fitness

    return best_population, best_fitness, (generation - 1)


# Algoritmo Genético, por Seleção do Torneio
def genetic_algorithm_tournament(
    initial_population: np.ndarray,
    fitness_func,
    max_gen=10000,
    size_tournament=3,
    elitism=False,
    elite_size=3,
    crossover_rate=0.8,
    mutation_rate=0.2
):
    """Aplicação do Algoritmo Genético, a partir
    de uma população de bitstring. 
    Uso da Seleção do Torneio.

    Args:
        initial_population: matriz de bitstrings da população inicial
        fitness_func: função de avaliação de aptidão
        max_gen: número máximo de gerações possíveis (padrão: 10000)
        size_tournament: número de indivíduos selecionados aleatóriamente para o torneio (padrão: 3)
        elitism: considerar ou não o elitismo (padrão: False)
        elite_size: quantidade de indivíduos para elitismo (padrão: 3)
        crossover_rate: taxa de crossover (padrão: 0.8)
        mutation_rate: taxa de mutação (padrão: 0.2)
        
    Returns:
        best_population: matriz com a melhor população obtida
        best_fitness: matriz contendo as aptidões da melhor população
        generation: número de gerações decorridas
    """

    print(f"{'-'*50}")
    print(f"{'Algoritmo Genético':^50}")
    print(f"{'-'*50}")

    # Avaliação da população inicial
    cur_fitness = np.array([fitness_func(individual)
                           for individual in initial_population])

    # Percorrendo as gerações
    generation = 1
    cur_population = initial_population
    while (generation <= max_gen and (initial_population.shape[1] not in cur_fitness)):
        # Fase de Seleção
        cur_population = tournament_selection(
            cur_population, cur_fitness, size_tournament)

        # Fase de Reprodução
        cur_population = reproduction(
            cur_population, elitism, elite_size, crossover_rate, mutation_rate)

        # Fase de Avaliação
        cur_fitness = np.array([fitness_func(individual)
                               for individual in cur_population])

        # Aumenta o número da geração atual
        generation = generation + 1

    # Retornando a melhor população
    best_population = cur_population
    best_fitness = cur_fitness

    return best_population, best_fitness, (generation - 1)


########################################
#               Testes                 #
########################################

def main():
    """Função principal do programa
    """

    # Geração da população inicial para os testes
    teste = generate_population()
    print(teste, '\n', teste.shape[0], '\n', teste.shape[1], '\n')

    t1, t2, t3 = genetic_algorithm_roulette(teste, f1_fitness)
    print(t1, '\n', t2, '\n', t3, '\n')

    t1, t2, t3 = genetic_algorithm_tournament(teste, f1_fitness)
    print(t1, '\n', t2, '\n', t3, '\n')


if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
