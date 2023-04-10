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

# Função do Exercício 02
def f1(x):
    return np.power(2, -2 * ((x - 0.1)/0.9) ** 2) * np.sin(5 * np.pi * x) ** 6


# Função do Exercício 03
def f2(x, y):
    return (1 - x) ** 2 + 100 * (y - (x ** 2)) ** 2


# Função de Aptidão: bitstring [1 1 1 1 0 1 1 0 1 1 1 1]
def fitness(individual_bitstring):
    """Função de Aptidão representado 
    pela bitstring [1 1 1 1 0 1 1 0 1 1 1 1].

    Args:
        individual_bitstring: vetor de bitstring do indivíduo
        
    Returns:
        hamming_dist: valor da aptidão baseado na distância de Hamming
    """
    # Criação da bitstring de verificação do exercício 01
    default_bitstring = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1])

    # Calculo da Distância de Hamming para encontrar o valor total da aptidão
    hamming_dist = np.sum(individual_bitstring != default_bitstring)

    return hamming_dist


#####################################################
#            Algoritmos Pré-Genéticos               #
#####################################################

# Subida da Colina
def hill_climbing(func, target_value: float, is_min: bool, x_min=-10.0, x_max=10.0, step_size=0.1, num_steps=1000):
    """Aplicação do algoritmo "Subida da Colina",
    em sua versão padrão, para busca local em uma função.
    Utiliza, neste caso, apenas UM PONTO da função.\n
    Para mais pontos, utilize a "Subida da Colina Iterativa".

    Args:
        func: função de avaliação
        target_value: valor alvo estipulado para função de avaliação
        is_min: booleana para atribuir algoritmo para min/max (padrão: True)
            -> Se 'True', irá verificar minimização\n
            -> Se 'False', irá verificar maximização
        x_min: valor mínimo do intervalo para geração de número aleatório (padrão: -10.0)
        x_max: valor máximo do intervalo para geração de número aleatório (padrão: 10.0)
        step_size: valor de delta X (padrão: 0.1)
        num_steps: número máximo de iterações (padrão: 1000)
        
        
    Returns:
        current_x: mínimo/máximo local encontrado
        func(current_x): valor da função no mínimo/máximo local encontrado
    """

    print(f"{'-'*50}")
    print(f"{'Hill-Climbing padrão':^50}")
    print(f"{'-'*50}")

    # Inicialização aleatória de um ponto da função
    current_x = np.random.uniform(x_min, x_max)

    # Encontrando o valor da função para o X iniciado
    current_func = func(current_x)

    # Realizando os passos iterativos
    step = 1
    while (step <= num_steps and current_func != target_value):

        # Calcula o valor da função para o X atual
        current_func = func(current_x)

        # Gera um novo ponto X, a partir da pertubação
        neighbor_x = current_x + np.random.uniform(-step_size, step_size)

        # Calcula o valor da função para o novo ponto X
        neighbor_func = func(neighbor_x)

        # Realiza a avaliação dos valores
        # Minimização
        if (is_min):
            # Se o valor da função no novo ponto for menor
            # do que o valor atual, atualiza o ponto atual
            if (neighbor_func < current_func):
                current_x = neighbor_x

        # Maximização
        else:
            # Se o valor da função no novo ponto for maior
            # do que o valor atual, atualiza o ponto atual
            if (neighbor_func > current_func):
                current_x = neighbor_x

        # Aumenta o número da iteração
        step = step + 1
    print(step)
    # Retorno do melhor ponto encontrado
    return current_x, func(current_x)


# Subida da Colina Iterativo
def iterated_hill_climbing(func, num_points: int, target_value: float, is_min: bool, x_min=-10.0, x_max=10.0, step_size=0.1, num_steps=1000):
    """Aplicação do algoritmo "Subida da Colina",
    em sua versão iterativa, para busca local em uma função.
    Utiliza, neste caso, VÁRIOS PONTOS da função.\n
    
    Para apenas um ponto, utilize a "Subida da Colina padrão".

    Args:
        func: função de avaliação
        n_start: número de pontos iniciais para aplicar no algoritmo padrão
        target_value: valor alvo estipulado para função de avaliação 
        is_min: booleana para atribuir algoritmo para min/max (padrão: True)
            -> Se 'True', irá verificar minimização\n
            -> Se 'False', irá verificar maximização
        x_min: valor mínimo do intervalo para geração de número aleatório (padrão: -10.0)
        x_max: valor máximo do intervalo para geração de número aleatório (padrão: 10.0)
        step_size: valor de delta X (padrão: 0.1)
        num_steps: número máximo de iterações (padrão: 1000)
        
        
    Returns:
        best_x: melhor mínimo/máximo local encontrado
        func(best_x): valor da função no melhor mínimo/máximo local encontrado
    """

    print(f"{'-'*50}")
    print(f"{'Hill-Climbing Iterativo':^50}")
    print(f"{'-'*50}")

    # Inicialização das variáveis para melhor valor encontrado
    best_x = None
    best_func = None

    # Geração de pontos aleatórios, com base na quantidade desejada
    # start_points = np.random.uniform(x_min, x_max, num_points)

    # Realizando os passos iterativos
    point = 1
    while (point <= num_points and best_func != target_value):
        # Geração aleatória de um ponto, com base no intervalo
        current_x = np.random.uniform(x_min, x_max)

        # Obtenção do valor da função neste ponto
        current_func = func(current_x)

        # Realizar o "Subida da Colina padrão" para este ponto
        step = 1
        while (step <= num_steps and current_func != target_value):
            # Calcula o valor da função para o X atual
            current_func = func(current_x)

            # Gera um novo ponto X, a partir da pertubação
            neighbor_x = current_x + np.random.uniform(-step_size, step_size)

            # Calcula o valor da função para o novo ponto X
            neighbor_func = func(neighbor_x)

            # Realiza a avaliação dos valores
            # Minimização
            if (is_min):
                # Se o valor da função no novo ponto for menor
                # do que o valor atual, atualiza o ponto atual
                if (neighbor_func < current_func):
                    current_x = neighbor_x

            # Maximização
            else:
                # Se o valor da função no novo ponto for maior
                # do que o valor atual, atualiza o ponto atual
                if (neighbor_func > current_func):
                    current_x = neighbor_x

            # Aumenta o número da iteração
            step = step + 1
        print(step)

        # Realiza a avaliação dos valores
        # Minimização
        if (is_min):
            # Se o valor da função no novo ponto for menor
            # do que o valor atual, atualiza o ponto atual
            if (best_func is None or func(current_x) < best_func):
                best_x = current_x
                best_func = func(current_x)

        # Maximização
        else:
            # Se o valor da função no novo ponto for maior
            # do que o valor atual, atualiza o ponto atual
            if (best_func is None or func(current_x) > best_func):
                best_x = current_x
                best_func = func(current_x)

        # Aumenta o número da iteração (ir para próximo ponto)
        point = point + 1
    print(point)
    # Retorno do melhor ponto encontrado
    return best_x, func(best_x)


# Subida da Colina Probabilístico
def stochastic_hill_climbing(func, target_value: float, is_min: bool, x_min=-10.0, x_max=10.0, step_size=0.1, num_steps=1000, t_exp=0.01):
    """Aplicação do algoritmo "Subida da Colina",
    em sua versão probabilística, para busca local em uma função.
    Utiliza, neste caso, apenas UM PONTO da função.\n
    
    Para um ponto, porém menos efetivo, utilize a "Subida da Colina padrão".
    Para mais pontos, utilize a "Subida da Colina Iterativa".

    Args:
        func: função de avaliação
        target_value: valor alvo estipulado para função de avaliação
        is_min: booleana para atribuir algoritmo para min/max (padrão: True)
            -> Se 'True', irá verificar minimização\n
            -> Se 'False', irá verificar maximização
        x_min: valor mínimo do intervalo para geração de número aleatório (padrão: -10.0)
        x_max: valor máximo do intervalo para geração de número aleatório (padrão: 10.0)
        step_size: valor de delta X (padrão: 0.1)
        num_steps: número máximo de iterações (padrão: 1000)
        t_exp: controle do decaimento da função exponencial (padrão: 1.5)
        
        
    Returns:
        current_x: mínimo/máximo local encontrado
        func(current_x): valor da função no mínimo/máximo local encontrado
    """

    print(f"{'-'*50}")
    print(f"{'Hill-Climbing Probabilístico':^50}")
    print(f"{'-'*50}")

    # Inicialização aleatória de um ponto da função
    current_x = np.random.uniform(x_min, x_max)

    # Encontrando o valor da função para o X iniciado
    current_func = func(current_x)

    # Realizando os passos iterativos
    step = 1
    while (step <= num_steps and current_func != target_value):

        # Calcula o valor da função para o X atual
        current_func = func(current_x)

        # Gera um novo ponto X, a partir da pertubação
        neighbor_x = current_x + np.random.uniform(-step_size, step_size)

        # Calcula o valor da função para o novo ponto X
        neighbor_func = func(neighbor_x)

        # Realiza a avaliação dos valores
        # Minimização
        if (is_min):
            # Calcula a probabilidade P do método
            # prob = (1 / (1 + np.exp((neighbor_func - current_func) / (t_exp + 1e-8))))
            prob = np.exp((neighbor_func - current_func) / (t_exp + 1e-8))

            # Caso o resultado seja melhor que o atual
            if current_func >= neighbor_func:
                current_x = neighbor_x
            # Do coontrário, valida o objetivo com base na probabilidade
            elif np.random.uniform() < prob:
                current_x = neighbor_x

        # Maximização
        else:
            # Calcula a probabilidade P do método
            # prob = (1 / (1 + np.exp((current_func - neighbor_func) / (t_exp + 1e-8))))
            prob = np.exp((current_func - neighbor_func) / (t_exp + 1e-8))

            # Caso o resultado seja melhor que o atual
            if current_func <= neighbor_func:
                current_x = neighbor_x
            # Do coontrário, valida o objetivo com base na probabilidade
            elif np.random.uniform() < prob:
                current_x = neighbor_x

        # Aumenta o número da iteração
        step = step + 1
    print(step)
    # Retorno do melhor ponto encontrado
    return current_x, func(current_x)


# Recozimento Simulado
def simulated_annealing(func, target_value: float, is_min: bool, x_min=-10.0, x_max=10.0, step_size=0.1, num_steps=1000, t_initial=0.01, beta=0.8):
    """Aplicação do algoritmo "Recozimento Simulado"
    para busca global em uma função.

    Args:
        func: função de avaliação
        target_value: valor alvo estipulado para função de avaliação
        is_min: booleana para atribuir algoritmo para min/max (padrão: True)
            -> Se 'True', irá verificar minimização.\n
            -> Se 'False', irá verificar maximização.
        x_min: valor mínimo do intervalo para geração de número aleatório (padrão: -10.0)
        x_max: valor máximo do intervalo para geração de número aleatório (padrão: 10.0)
        step_size: valor de delta X (padrão: 0.1)
        num_steps: número máximo de iterações (padrão: 1000)
        t_initial: valor decimal da temperatura inicial (padrão: 0.01)
        beta: taxa de resfriamento pelo decremento geométrico (padrão: 0.8)
        
        
    Returns:
        current_x: melhor mínimo/máximo global encontrado
        func(current_x): valor da função no melhor mínimo/máximo global encontrado
    """

    print(f"{'-'*50}")
    print(f"{'Recozimento Simulado':^50}")
    print(f"{'-'*50}")

    # Inicialização do ponto X aleatoriamente
    current_x = np.random.uniform(x_min, x_max)

    # Calculo do valor da função no ponto gerado
    current_func = func(current_x)

    # Realizando os passos iterativos
    step = 1
    t_actual = t_initial
    while (step <= num_steps and current_func != target_value):
        # Geração aleatória de um vizinho de X
        neighbor_x = np.random.uniform(-step_size, step_size)

        # Calculo do valor da função neste vizinho gerado
        neighbor_func = func(neighbor_x)

        # Realiza a avaliação dos valores
        # Minimização
        if (is_min):
            # Calcula a probabilidade P do método
            # prob = (1 / (1 + np.exp((neighbor_func - current_func) / (t_actual + 1e-8))))
            prob = np.exp((neighbor_func - current_func) / (t_actual + 1e-8))

            # Caso o resultado seja melhor que o atual
            if current_func >= neighbor_func:
                current_x = neighbor_x
            # Do coontrário, valida o objetivo com base na probabilidade
            elif np.random.uniform() < prob:
                current_x = neighbor_x

        # Maximização
        else:
            # Calcula a probabilidade P do método
            #prob = (1 / (1 + np.exp((current_func - neighbor_func) / (t_actual + 1e-8))))
            prob = np.exp((current_func - neighbor_func) / (t_actual + 1e-8))

            # Caso o resultado seja melhor que o atual
            if current_func <= neighbor_func:
                current_x = neighbor_x
            # Do coontrário, valida o objetivo com base na probabilidade
            elif np.random.uniform() < prob:
                current_x = neighbor_x

        # Calculando o novo valor da temperatura
        t_actual *= beta

        # Aumenta o número da iteração
        step = step + 1
    print(step)
    # Retorno do melhor ponto encontrado
    return current_x, func(current_x)


#####################################################
#               Algoritmo Genético                  #
#####################################################

########################################
#              0 - Geração             #
########################################

# Geração Aleatória da População
def generate_population():
    """Função responsável por gerar uma nova população.

    Args:       
        
        
    Returns:

    """


########################################
#              1 - Seleção             #
########################################

# Seleção por Roleta
def roulette_selection(cur_population: np.ndarray, all_fitness: np.ndarray, start=0, end=360):
    """Seleção de indivíduo pelo método da Roleta.

    Args:
        cur_population: vetor contendo a população atual para seleção
        all_fitness: vetor contendo todos os valores de aptidão da população
        start: início exclusivo do intervalo da roleta (padrão: 0)
        end: fim inclusivo do intervalo da roleta (padrão: 360)
        
    Returns:
        new_population: vetor contendo a nova população selecionada
    """

    # Somar a aptidão de todos indivíduos
    sum_all_fitness = np.sum(all_fitness)

    # Determinar a porção da roleta para cada indivíduo no intervalo (start; end]
    prob_roulette = start + np.array(((end - start) * all_fitness) / sum_all_fitness)

    # Determinar os intervalos da roleta para cada indivíduo
    intervals_roulette = np.zeros(np.size(all_fitness) + 1)
    for i in range(1, len(intervals_roulette)):
        # Inicia com a segunda posição
        intervals_roulette[i] = intervals_roulette[i-1] + prob_roulette[i-1]

    # Seleção dos indivíduos com base na roleta
    new_population = []
    for i in range(np.size(cur_population)):
        # Geração do valor aleatório
        rand_val = np.random.uniform(low=start + 1, high=end + 1)

        # Percorrendo o vetor de intervalos
        for j in range(len(intervals_roulette) - 1):
            # Verificando em que posição o valor randômico está inserido
            if intervals_roulette[j] < rand_val <= intervals_roulette[j + 1]:
                # Salva o indivíduo no novo vetor de população
                new_population.append(cur_population[j])
                break

    return np.array(new_population)


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

        # Escolha do vencedor com base no valor obtido
        winner = candidates[np.argmax(all_fitness[candidates])]

        # Salvando o vencedor na nova população
        new_population[i] = cur_population[winner]

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
        crossover_rate: float64 que representa a taxa acontecimento de crossover (padrão: 0.8)
        
    Returns:
        child1: vetor representando o primeiro filho
        child2: vetor representando o segundo filho
    """

    # Para ocorrer o crossover, um número aleatório deve ser menor ou igual a taxa
    if np.random.rand() <= crossover_rate:
        # Sorteia um ponto de corte
        crossover_point = np.random.randint(1, np.size(parent1) - 1)

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
def make_sex_without_wife(selected_population: np.ndarray, all_fitness: np.ndarray, elite_size=3, is_min=True, crossover_rate=0.8, mutation_rate=0.2):
    """Reprodução de uma determinada população, 
    considerando crossover e mutação.

    Args:
        selected_population: vetor com a população selecionada
        all_fitness: vetor contendo o valor das aptidões obtidas com a seleção
        elite_size: quantidade de indivíduos para elitismo (padrão: 3)
        is_min: booleana para atribuir algoritmo para min/max (padrão: True)
            -> Se 'True', irá verificar minimização\n
            -> Se 'False', irá verificar maximização
        crossover_rate: taxa de crossover (padrão: 0.8)
        mutation_rate: taxa de mutação (padrão: 0.2)
        
    Returns:
        new_population: vetor com a nova população
    """
    
    # Ordenação da população com base na aptidão obtida e do tipo de problema
    if is_min:
        selected_population = selected_population[np.argsort(all_fitness)]
    else:
        selected_population = selected_population[np.argsort(-all_fitness)]

    # Seleção dos indivíduos de elite (últimos 'elite_size' da ordenação)
    elite = selected_population[:elite_size]
    
    # Criação da nova população já contendo a elite
    new_population = np.copy(elite)
    
    # Criação de novos indivíduos com crossover e mutação
    for i in range((np.size(selected_population) - elite_size) // 2):
        # Percorre até a metade da população, selecionando pares aleatórios
        parent1, parent2 = np.random.choice(selected_population, 2, replace=False)
        
        # Fase de crossover
        child1, child2 = crossover(parent1, parent2, crossover_rate)
        
        # Fase de mutação
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)
        
        # Adiciona os filhos na nova população
        new_individuals = [child1, child2]
        new_population = np.append(new_population, new_individuals, axis=0)
    
    # Caso o número da população seja ímpar, adiciona o último indivíduo
    if ((np.size(selected_population) - elite_size) % 2 == 1):
        new_population = np.append(new_population, [selected_population[-1]], axis=0)         

    return new_population


########################################
#         X - Função Principal         #
########################################

# Algoritmo Genético
def genetic_algorithm():
    """Aplicação do Algoritmo Genético, a partir
    de uma população.

    Args:
        
        
        
    Returns:
        
    """

    print(f"{'-'*50}")
    print(f"{'Algoritmo Genético':^50}")
    print(f"{'-'*50}")


def main():
    """Função principal do programa
    """

    test, test2 = hill_climbing(
        f1, 1, False, x_min=0, x_max=1, num_steps=100000)
    print(f"X: {test:.5f}\nF(X): {test2:.5f}")
    test, test2 = iterated_hill_climbing(
        f1, 10, 1, False, x_min=0, x_max=1, num_steps=100000)
    print(f"X: {test:.5f}\nF(X): {test2:.5f}")
    test, test2 = stochastic_hill_climbing(
        f1, 1, False, x_min=0, x_max=1, num_steps=100000)
    print(f"X: {test:.5f}\nF(X): {test2:.5f}")
    test, test2 = simulated_annealing(
        f1, 1, False, x_min=0, x_max=1, num_steps=100000)
    print(f"X: {test:.5f}\nF(X): {test2:.5f}\n")


if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
