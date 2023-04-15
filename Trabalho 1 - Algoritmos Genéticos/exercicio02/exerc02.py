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
def f2(x):
    #return np.power(2, -2 * ((x - 0.1)/0.9) ** 2) * np.sin(5 * np.pi * x) ** 6
    return (2 ** (-2 * (((x - 0.1) / 0.9) ** 2))) * ((np.sin(5 * np.pi * x)) ** 6)


#####################################################
#            Algoritmos Pré-Genéticos               #
#####################################################

# Subida da Colina
def hill_climbing(func, target_value: float, is_min: bool, x_min=-1.0, x_max=1.0, step_size=0.1, num_steps=1000):
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
        step: número de passos necessários
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
    
    # Retorno do melhor ponto encontrado
    return current_x, func(current_x), (step - 1)


# Subida da Colina Iterativo
def iterated_hill_climbing(func, num_points: int, target_value: float, is_min: bool, x_min=-1.0, x_max=1.0, step_size=0.1, num_steps=1000):
    """Aplicação do algoritmo "Subida da Colina",
    em sua versão iterativa, para busca local em uma função.
    Utiliza, neste caso, VÁRIOS PONTOS da função.\n
    
    Para apenas um ponto, utilize a "Subida da Colina padrão".

    Args:
        func: função de avaliação
        num_points: número de pontos iniciais para aplicar no algoritmo padrão
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
        step: número de passos necessários
    """

    print(f"{'-'*50}")
    print(f"{'Hill-Climbing Iterativo':^50}")
    print(f"{'-'*50}")

    # Inicialização das variáveis para melhor valor encontrado
    best_x = None
    best_func = None

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
    return best_x, func(best_x), (step - 1)


# Subida da Colina Probabilístico
def stochastic_hill_climbing(func, target_value: float, is_min: bool, x_min=-1.0, x_max=1.0, step_size=0.1, num_steps=1000, t_exp=0.01):
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
        step: número de passos necessários
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
    
    # Retorno do melhor ponto encontrado
    return current_x, func(current_x), (step - 1)


# Recozimento Simulado
def simulated_annealing(func, target_value: float, is_min: bool, x_min=-1.0, x_max=1.0, step_size=0.1, num_steps=1000, t_initial=0.01, beta=0.8):
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
        step: número de passos necessários
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
            # prob = (1 / (1 + np.exp((current_func - neighbor_func) / (t_actual + 1e-8))))
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
    
    # Retorno do melhor ponto encontrado
    return current_x, func(current_x), (step - 1)


#####################################################
#               Conversão Bitstring                 #
#####################################################

# Float 32bit para Binário
def float_to_binary(value: float, int_digits=15, dec_digits=15):
    """Função responsável por converter um valor,
    em float 32 bit, para representação binária,
    com número máximo de casas inteiras e decimas.

    Args:       
        value: número em float máximo 32 bit
        int_digits: quantidade máxima de dígitos na parte inteira (padrão: 15)
        dec_digits: quantidade máxima de dígitos na parte decimal (padrão: 15)
        
    Returns:
        binary: número convertido em binário
    """

    # Separar a parte inteira da decimal
    signal = '1' if value < 0 else '0'
    int_part = abs(int(value))
    dec_part = abs(value) - int_part
    #print(int_part, value, dec_part)
    
    # Converter a parte inteira para binário, completando com zero à esquerda
    int_str = np.binary_repr(int_part).zfill(int_digits)
    
    # Conversão da parte decimal para binário, completando com zeros à direita
    dec_str = ""
    for i in range(dec_digits):
        # Realiza a potência de 2
        dec_part *= 2
        
        # Caso tenha ficado parte inteira, acrescenta "1"
        if dec_part >= 1:
            dec_str += "1"
            dec_part -= 1
        # Caso não tenha ficado parte inteira, acrescenta "0"
        else:
            dec_str += "0"
    
    # Salvando o resultado e retornando
        # Como há a limitância de dígitos, não acrescentar "."
    binary = signal + int_str + dec_str
    #print(signal, int_str, '.', dec_str)

    return np.array(list(binary), dtype=np.uint8)


# Binário para Float 32bit
def binary_to_float(value: np.ndarray, int_digits=15, dec_digits=15):
    """Função responsável por converter um valor,
    em representação binária, para float 32 bit,
    com número máximo de casas inteiras e decimas.

    Args:       
        value: número em representação binára 
        int_digits: quantidade máxima de dígitos na parte inteira (padrão: 15)
        dec_digits: quantidade máxima de dígitos na parte decimal (padrão: 15)
        
    Returns:
        float_converted: número convertido em float máximo 32 bit
    """
    
    # Separa a parte inteira da decimal
    signal, int_str, dec_str = value[0], value[1:int_digits + 1], value[int_digits + 1:]
    
    # Convertendo sinal
    signal_value = (-1) if signal == 1 else (1)
    
    # Converter a parte inteira para número
    int_str = "".join(str(c) for c in int_str)
    int_num = int(int_str, 2)
    
    #print(signal, int_str, '.', "".join(str(c) for c in dec_str))
    
    # Converter a parte decimal para número
    dec_num = 0
    for i in range(dec_digits):
        # Aplicando a fórmula inversa da decimal
        dec_num += int(str(dec_str[i])) * (2 ** -(i + 1))
        
    # Recuperando o número float por completo
    float_converted = signal_value * (int_num + dec_num)
    
    return float_converted
    

#####################################################
#          Algoritmo Genético: Bitstring            #
#####################################################

########################################
#              0 - Geração             #
########################################

# Geração Aleatória da População
def generate_population(int_size=15, dec_size=15, size_population=10, x_min=-1.0, x_max=1.0):
    """Função responsável por gerar uma nova população,
    em bitstring (vetor de bits), advindos do float.

    Args: 
        int_size: quantidade máxima de dígitos na parte inteira (padrão: 15)
        dec_size: quantidade máxima de dígitos na parte decimal (padrão: 15)
        size_population: tamanho da população (padrão: 10)
        x_min: valor mínimo do intervalo para geração de número aleatório (padrão: -1.0)
        x_max: valor máximo do intervalo para geração de número aleatório (padrão: 1.0)
        
    Returns:
        population: população inicial gerada
    """
    
    # Geração de Float aleatório
    individual_vals = np.random.uniform(x_min, x_max, size_population)
    
    # Conversão de Float para bitstring e salva como indivíduo
    population = [float_to_binary(value, int_size, dec_size) for value in individual_vals]
    
    return np.array(population)


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
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
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
def reproduction(
    selected_population: np.ndarray,
    elitism=False, 
    elite_size=3, 
    crossover_rate=0.8, 
    mutation_rate=0.2
):
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
    new_population = np.concatenate((elite, children)) if elitism else np.array(children)

    return new_population


########################################
#         3 - Função Principal         #
########################################

# Algoritmo Genético, por Seleção da Roleta
def genetic_algorithm_roulette(
    initial_population: np.ndarray,
    fitness_func,
    target_value: float,
    int_size=15, 
    dec_size=15,
    x_min=-1.0, 
    x_max=1.0,
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
        target_value: valor alvo estipulado para função de avaliação
        int_size: quantidade máxima de dígitos na parte inteira (padrão: 15)
        dec_size: quantidade máxima de dígitos na parte decimal (padrão: 15)
        x_min: valor mínimo do intervalo para geração de número aleatório (padrão: -1.0)
        x_max: valor máximo do intervalo para geração de número aleatório (padrão: 1.0)
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
    print(f"{'Algoritmo Genético (por Roleta)':^50}")
    print(f"{'-'*50}")

    # Avaliação da população inicial
    cur_fitness = np.array([fitness_func((binary_to_float(individual, int_size, dec_size)))
                           for individual in initial_population])
    print("PRIMEIRO LIXO", cur_fitness)
    # Percorrendo as gerações
    generation = 1
    cur_population = initial_population
    while (generation <= max_gen and (target_value not in cur_fitness)):
        # Fase de Seleção
        cur_population = roulette_selection(
            cur_population, cur_fitness, start_roulette, end_roulette)

        # Fase de Reprodução
        cur_population = reproduction(
            cur_population, elitism, elite_size, crossover_rate, mutation_rate)

        # Fase de Avaliação
        cur_fitness = np.array([fitness_func(binary_to_float(individual, int_size, dec_size))
                               for individual in cur_population])

        # Aumenta o número da geração atual
        generation = generation + 1

    # Retornando a melhor população
    best_population = [binary_to_float(individual, int_size, dec_size) for individual in cur_population]
    best_fitness = cur_fitness

    return best_population, best_fitness, (generation - 1)


# Algoritmo Genético, por Seleção do Torneio
def genetic_algorithm_tournament(
    initial_population: np.ndarray,
    fitness_func,
    target_value: float,
    int_size=15, 
    dec_size=15,
    x_min=-1.0,
    x_max=1.0,
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
        target_value: valor alvo estipulado para função de avaliação
        int_size: quantidade máxima de dígitos na parte inteira (padrão: 15)
        dec_size: quantidade máxima de dígitos na parte decimal (padrão: 15)
        x_min: valor mínimo do intervalo para geração de número aleatório (padrão: -1.0)
        x_max: valor máximo do intervalo para geração de número aleatório (padrão: 1.0)
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
    print(f"{'Algoritmo Genético (por Torneio)':^50}")
    print(f"{'-'*50}")

    # Avaliação da população inicial
    cur_fitness = np.array([fitness_func(binary_to_float(individual, int_size, dec_size))
                           for individual in initial_population])
    print("SEGUNDO LIXO", cur_fitness)
    # Percorrendo as gerações
    generation = 1
    cur_population = initial_population
    while (generation <= max_gen and (target_value not in cur_fitness)):
        # Fase de Seleção
        cur_population = tournament_selection(
            cur_population, cur_fitness, size_tournament)

        # Fase de Reprodução
        cur_population = reproduction(
            cur_population, elitism, elite_size, crossover_rate, mutation_rate)

        # Fase de Avaliação
        cur_fitness = np.array([fitness_func(binary_to_float(individual, int_size, dec_size))
                               for individual in cur_population])

        # Aumenta o número da geração atual
        generation = generation + 1

    # Retornando a melhor população
    #print(cur_population)
    best_population = [binary_to_float(
        individual, int_size, dec_size) for individual in cur_population]
    best_fitness = cur_fitness

    return best_population, best_fitness, (generation - 1)


########################################
#               Testes                 #
########################################

def main():
    """Função principal do programa
    """

    # Geração da população inicial para os testes'
    int_size = 25
    dec_size = 10
    teste = generate_population(
        int_size=int_size, dec_size=dec_size, size_population=15, x_min=-1.0, x_max=1.0)
    print(teste, '\n', teste.shape[0], '\n', len(teste[0]), '\n')

    t1, t2, t3 = genetic_algorithm_roulette(teste, f2, target_value=1, int_size=int_size, dec_size=dec_size)
    print(t1, '\n', t2, '\n', t3, '\n')

    t1, t2, t3 = genetic_algorithm_tournament(
        teste, f2, target_value=1, int_size=int_size, dec_size=dec_size)
    print(t1, '\n', t2, '\n', t3, '\n')
    

if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()


