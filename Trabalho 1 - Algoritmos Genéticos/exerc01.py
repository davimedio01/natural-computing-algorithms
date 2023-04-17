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

    return (np.size(individual_bitstring) - hamming_dist)


#####################################################
#          Algoritmo Genético: Bitstring            #
#####################################################

########################################
#              0 - Geração             #
########################################

# Geração Aleatória da População
def generate_population(
    size_bitstring=12, 
    size_population=10
):
    """Função responsável por gerar uma nova população,
    em bitstring (vetor de bits inteiros de 0/1).

    Args:       
        size_bitstring (int): tamanho do bitstring (padrão: 12)
        size_population (int): tamanho da população (padrão: 10)
        
    Returns:
        population (np.ndarray[[int, ...], ...]): população inicial gerada
    """
    population = np.random.randint(2, size=(size_population, size_bitstring))
    return population


########################################
#              1 - Seleção             #
########################################

# Seleção por Roleta
def __roulette_selection(
    cur_population: np.ndarray, 
    fitness: np.ndarray
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
    size=3
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
    crossover_rate=0.8
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
    mutation_rate=0.2
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
    mutation_rate=0.2
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
    mutation_rate=0.2
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
        best_population (np.ndarray[[int, ...], ...]): lista com as melhores populações obtidas ao decorrer das gerações
        best_fitness (np.ndarray[[int], ...]): lista com as melhores aptidões obtidas ao decorrer das gerações
        all_population (np.ndarray[[int, ...], ...]): lista com todas as populações obtidas, sendo a última a melhor possível
        all_fitness (np.ndarray[[int, ...], ...]): lista com todas as aptidões obtidas, sendo a última a melhor possível
        generation (int): número de gerações decorridas
    """

    print(f"{'-'*50}")
    print(f"{'Algoritmo Genético':^50}")
    print(f"{'-'*50}")

    # Avaliação da população inicial
    cur_fitness = np.array([fitness_func(individual)
                           for individual in initial_population])

    # Recuperando o melhor da população inicial e definindo valores iniciais
    best_fitness = [np.max(cur_fitness)]
    best_population = np.array([individual for individual in 
                                [initial_population[np.argmax(cur_fitness)]]])#.reshape((1, initial_population.shape[0]))
    all_population = np.array([individual for individual in initial_population])#.reshape((1, initial_population.shape[0], 1))
    all_fitness = np.array([cur_fitness])
    
    # Início feito a partir da população inicial
    cur_population = initial_population

    # Percorrendo as gerações
    generation = 1
    patience = 1
    while (generation < max_gen and patience != max_patience and (initial_population.shape[1] not in cur_fitness)):
        # Aumenta o número da geração atual
        generation = generation + 1

        # Fase de Seleção
        cur_population, cur_fitness = selection_func(cur_population=cur_population, fitness=cur_fitness, size=size_tournament)if selection_func == __tournament_selection else selection_func(cur_population=cur_population, fitness=cur_fitness)
            
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

        # Atualização dos valores
        if np.max(cur_fitness) > np.max(best_fitness):
            best_fitness = np.append(
                best_fitness,
                np.array([np.max(cur_fitness)]),
                axis=0)
            best_population = np.append(
                best_population,
                [individual for individual in [cur_population[np.argmax(cur_fitness)]]],
                axis=0)
            patience = 1
        elif np.max(cur_fitness) == np.max(best_fitness):
            patience = patience + 1
            
        # Independente se conseguiu melhor resultado, salva nas listas de retorno para pós-visualização
        all_population = np.append(
            all_population,
            [individual for individual in cur_population],
            axis=0)

        all_fitness = np.append(all_fitness, [cur_fitness], axis=0)

    print(f"Geração {generation}")
    print(f"Melhor Aptidão: {best_fitness[-1]}")
    print(f"Melhor Indivíduo: {best_population[-1]}")

    print(f"{'-'*50}")
    print(f"{'Fim do Algoritmo Genético':^50}")
    print(f"{'-'*50}")

    return best_population, best_fitness, all_population, all_fitness, generation


########################################
#               Testes                 #
########################################

def main():
    """Função principal do programa
    """
    
    np.set_printoptions(threshold=np.inf)
    
    # Geração da população inicial para os testes
    teste = generate_population()
    
    t1, t2, t3, t4, t5 = genetic_algorithm(teste, f1, __roulette_selection)
    #print(t1, '\n', t2, '\n', t3, '\n', t4, '\n', t5, '\n')

    t1, t2, t3, t4, t5 = genetic_algorithm(teste, f1, __tournament_selection)
    #print(t1, '\n', t2, '\n', t3, '\n', t4, '\n', t5, '\n')


if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
