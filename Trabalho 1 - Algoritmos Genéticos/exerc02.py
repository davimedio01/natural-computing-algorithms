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
def f2(x: list[float]) -> float:
    return (2 ** (-2 * (((x[0] - 0.1) / 0.9) ** 2))) * ((np.sin(5 * np.pi * x[0])) ** 6)


def f3(x: list[float]) -> float:
    return (1 - x[0]) ** 2 + 100 * (x[1] - (x[0] ** 2)) ** 2


#####################################################
#            Algoritmos Pré-Genéticos               #
#####################################################

# Subida da Colina
def hill_climbing(
    func: callable,
    target_value: float = float('inf'),
    is_min=True,
    bounds=np.array([[-1.0, 1.0]]),
    step_size=0.1,
    num_steps=1000
):
    """Aplicação do algoritmo "Subida da Colina",
    em sua versão padrão, para busca local em uma função.
    Utiliza, neste caso, apenas UM PONTO da função.\n
    Para mais pontos, utilize a "Subida da Colina Iterativa".

    Args:
        func (callable): função de avaliação
        target_value (float, opcional): valor alvo estipulado para função de avaliação 
        is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
            -> Se 'True', irá verificar minimização\n
            -> Se 'False', irá verificar maximização
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
        step_size (float): valor de delta X (padrão: 0.1)
        num_steps (int): número máximo de iterações (padrão: 1000)
        
    Returns:
        current_x_values (np.ndarray[float, ...]): valores de x calculados, sendo o último o mínimo/máximo local encontrado
        current_func_values (np.ndarray[float, ...]): valor da função calculados, sendo o último o melhor possível
        step (int): número de passos necessários
    """

    print(f"{'-'*50}")
    print(f"{'Hill-Climbing':^50}")
    print(f"{'-'*50}")

    # Inicialização aleatória de um ponto da função
    current_x = np.random.uniform(bounds[0][0], bounds[0][1])

    # Encontrando o valor da função para o X iniciado
    current_func = func(current_x)

    # Salva os valores de X e Y no vetor de saída
    current_x_values = np.array(current_x, dtype=np.float32)
    current_func_values = np.array(current_func, dtype=np.float32)

    # Realizando os passos iterativos
    step = 1
    while (step < num_steps and current_func != target_value):

        # Aumenta o número da iteração
        step = step + 1

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

        # Salva os valores de X e Y no vetor de saída
        current_x_values = np.append(current_x_values, current_x)
        current_func_values = np.append(current_func_values, func(current_x))

    print(f"{'-'*50}")
    print(f"{'Fim do Hill-Climbing':^50}")
    print(f"{'-'*50}")

    # Retorno do melhor ponto encontrado
    return current_x_values, current_func_values, step


# Subida da Colina Iterativo
def iterated_hill_climbing(
    func: callable,
    num_points: int,
    target_value: float = float('inf'),
    is_min=True,
    bounds=np.array([[-1.0, 1.0]]),
    step_size=0.1,
    num_steps=1000
):
    """Aplicação do algoritmo "Subida da Colina",
    em sua versão iterativa, para busca local em uma função.
    Utiliza, neste caso, VÁRIOS PONTOS da função.\n
    
    Para apenas um ponto, utilize a "Subida da Colina padrão".

    Args:
        func (callable): função de avaliação
        num_points (int): número de pontos iniciais para aplicar no algoritmo padrão
        target_value (float, opcional): valor alvo estipulado para função de avaliação
        is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
            -> Se 'True', irá verificar minimização\n
            -> Se 'False', irá verificar maximização
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
        step_size (float): valor de delta X (padrão: 0.1)
        num_steps (int): número máximo de iterações (padrão: 1000)
        
    Returns:
        current_x_values (np.ndarray[[float, float], ...]): valores de x calculados
            -> Leve em conta a cada dois valores, sendo o primeiro o ponto inicial e o segundo o melhor ponto obtido a partir do primeiro
        current_func_values (np.ndarray[[float, float], ...]): valor da função calculados, sendo o último o melhor possível
        points (int): número de pontos totais utilizados
        step (int): número de passos necessários
    """

    print(f"{'-'*50}")
    print(f"{'Hill-Climbing Iterativo':^50}")
    print(f"{'-'*50}")

    # Inicialização das variáveis para melhor valor encontrado
    best_x = None
    best_func = None
    current_x_values = []
    current_func_values = []

    # Realizando os passos iterativos
    point = 1
    while (point < num_points and best_func != target_value):
        # Aumenta o número da iteração (ir para próximo ponto)
        point = point + 1

        # Geração aleatória de um ponto, com base no intervalo
        current_x = np.random.uniform(bounds[0][0], bounds[0][1])

        # Obtenção do valor da função neste ponto
        current_func = func(current_x)

        # Salva os valores de X e Y iniciais no vetor de saída
        initial_x = current_x
        initial_func = current_func

        # Realizar o "Subida da Colina padrão" para este ponto
        step = 1
        while (step < num_steps and current_func != target_value):
            # Aumenta o número da iteração
            step = step + 1

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

        # Salva os valores de X e Y melhores no vetor de saída
        initial_x = [initial_x, best_x]
        initial_func = [initial_func, best_func]

        current_x_values.append(initial_x)
        current_func_values.append(initial_func)

    print(f"{'-'*50}")
    print(f"{'Fim do Hill-Climbing Iterativo':^50}")
    print(f"{'-'*50}")

    # Retorno do melhor ponto encontrado
    return np.array(current_x_values), np.array(current_func_values), point, step - 1


# Subida da Colina Probabilístico
def stochastic_hill_climbing(
    func: callable,
    target_value: float = float('inf'),
    is_min=True,
    bounds=np.array([[-1.0, 1.0]]),
    step_size=0.1,
    num_steps=1000,
    t_exp=0.01
):
    """Aplicação do algoritmo "Subida da Colina",
    em sua versão probabilística, para busca local em uma função.
    Utiliza, neste caso, apenas UM PONTO da função.\n
    
    Para um ponto, porém menos efetivo, utilize a "Subida da Colina padrão".
    Para mais pontos, utilize a "Subida da Colina Iterativa".

    Args:
        func (callable): função de avaliação
        target_value (float, opcional): valor alvo estipulado para função de avaliação
        is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
            -> Se 'True', irá verificar minimização\n
            -> Se 'False', irá verificar maximização
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
        step_size (float): valor de delta X (padrão: 0.1)
        num_steps (int): número máximo de iterações (padrão: 1000)
        t_exp (float): controle do decaimento da função exponencial (padrão: 1.5)
        
    Returns:
        current_x_values (np.ndarray[float, ...]): valores de x calculados, sendo o último o mínimo/máximo local encontrado
        current_func_values (np.ndarray[float, ...]): valor da função calculados, sendo o último o melhor possível
        step (int): número de passos necessários
    """

    print(f"{'-'*50}")
    print(f"{'Hill-Climbing Probabilístico':^50}")
    print(f"{'-'*50}")

    # Inicialização aleatória de um ponto da função
    current_x = np.random.uniform(bounds[0][0], bounds[0][1])

    # Encontrando o valor da função para o X iniciado
    current_func = func(current_x)

    # Salva os valores de X e Y no vetor de saída
    current_x_values = np.array(current_x, dtype=np.float32)
    current_func_values = np.array(current_func, dtype=np.float32)

    # Realizando os passos iterativos
    step = 1
    while (step < num_steps and current_func != target_value):
        # Aumenta o número da iteração
        step = step + 1

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

        # Salva os valores de X e Y no vetor de saída
        current_x_values = np.append(current_x_values, neighbor_x)
        current_func_values = np.append(current_func_values, func(neighbor_x))

    print(f"{'-'*50}")
    print(f"{'Fim do Hill-Climbing Probabilístico':^50}")
    print(f"{'-'*50}")

    # Retorno do melhor ponto encontrado
    return current_x_values, current_func_values, step


# Recozimento Simulado
def simulated_annealing(
    func: callable,
    target_value: float = float('inf'),
    is_min=True,
    bounds=np.array([[-1.0, 1.0]]),
    step_size=0.1,
    num_steps=1000,
    t_initial=0.01,
    beta=0.8
):
    """Aplicação do algoritmo "Recozimento Simulado"
    para busca global em uma função.

    Args:
        func (callable): função de avaliação
        target_value (float, opcional): valor alvo estipulado para função de avaliação
        is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
            -> Se 'True', irá verificar minimização.\n
            -> Se 'False', irá verificar maximização.
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
        step_size (float): valor de delta X (padrão: 0.1)
        num_steps (int): número máximo de iterações (padrão: 1000)
        t_initial (float): valor decimal da temperatura inicial (padrão: 0.01)
        beta (float): taxa de resfriamento pelo decremento geométrico (padrão: 0.8)
        
    Returns:
        current_x_values (np.ndarray[float, ...]): valores de x calculados, sendo o último o mínimo/máximo global encontrado
        current_func_values (np.ndarray[float, ...]): valor da função calculados, sendo o último o melhor possível
        step (int): número de passos necessários
    """

    print(f"{'-'*50}")
    print(f"{'Recozimento Simulado':^50}")
    print(f"{'-'*50}")

    # Inicialização do ponto X aleatoriamente
    current_x = np.random.uniform(bounds[0][0], bounds[0][1])

    # Calculo do valor da função no ponto gerado
    current_func = func(current_x)

    # Salva os valores de X e Y no vetor de saída
    current_x_values = np.array(current_x, dtype=np.float32)
    current_func_values = np.array(current_func, dtype=np.float32)

    # Realizando os passos iterativos
    step = 1
    t_actual = t_initial
    while (step < num_steps and current_func != target_value):
        # Aumenta o número da iteração
        step = step + 1

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
            # Do contrário, valida o objetivo com base na probabilidade
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

        # Salva os valores de X e Y no vetor de saída
        current_x_values = np.append(current_x_values, neighbor_x)
        current_func_values = np.append(current_func_values, func(neighbor_x))

    print(f"{'-'*50}")
    print(f"{'Fim do Recozimento Simulado':^50}")
    print(f"{'-'*50}")

    # Retorno do melhor ponto encontrado
    return current_x_values, current_func_values, step


#####################################################
#          Algoritmo Genético: Bitstring            #
#####################################################

#####################################################
#               Conversão Bitstring                 #
#####################################################

# Float para Binário
def __float_to_binary(
    values: np.ndarray,
    bounds: np.ndarray,
    int_digits=11,
    dec_digits=20
):
    """Função responsável por converter um valor,
    em float, para representação binária,
    com número máximo de casas inteiras e decimas.
    
    OBS: O primeiro bit da parte inteira é de sinal!
    (Considere como tamanho total = 1 + int_digits + dec_digits)

    Args:       
        values (np.ndarray[float, ...]): variáveis em número em float
        bounds (np.ndarray[[float, float], ...]): lista contendo os intervalos das variáveis da função
        int_digits (int): quantidade máxima de dígitos na parte inteira (padrão: 11)
        dec_digits (int): quantidade máxima de dígitos na parte decimal (padrão: 20)
        
    Returns:
        binary_converted (list[bitstring, ...]): lista com número convertido em binário
    """

    # Inicia a lista de saída
    # binary_converted = np.empty(((bounds.shape[0]), values.shape[0]), dtype=str)
    binary_converted = []

    # Percorrendo cada valor gerado de cada variável
    for value in values:
        # Separar a parte inteira da decimal
        signal = '1' if value < 0 else '0'
        int_part = abs(int(value))
        dec_part = abs(value) - int_part

        # Converter a parte inteira para binário, completando com zero à esquerda
        int_str = np.binary_repr(int_part).zfill(int_digits)

        # Conversão da parte decimal para binário, completando com zeros à direita
        dec_str = ""
        for _ in range(dec_digits):
            # Realiza a potência de 2
            dec_part *= 2

            # Caso tenha ficado parte inteira, acrescenta "1"
            if dec_part >= 1:
                dec_str += "1"
                dec_part -= 1
            # Caso não tenha ficado parte inteira, acrescenta "0"
            else:
                dec_str += "0"

        # Salvando o resultado na lista
        binary = signal + int_str + dec_str
        binary_converted.append(list(binary))

    return binary_converted


# Binário para Float
def __binary_to_float(
    values: np.ndarray,
    bounds: np.ndarray,
    int_digits=11,
    dec_digits=20
):
    """Função responsável por converter um valor,
    em representação binária, para float,
    com número máximo de casas inteiras e decimas.
    
    OBS: O primeiro bit da parte inteira é de sinal!
    (Considere como tamanho total = 1 + int_digits + dec_digits)

    Args:       
        values (np.ndarray[bitstring, ...]): lista com as variáveis em representação binária 
        bounds (np.ndarray[[float, float], ...]): lista contendo os intervalos das variáveis da função
        int_digits (int): quantidade máxima de dígitos na parte inteira (padrão: 11)
        dec_digits (int): quantidade máxima de dígitos na parte decimal (padrão: 20)
        
    Returns:
        float_converted (list[float, ...]): lista de variáveis convertidas em float
    """

    # Obtém o maior valor possível do binário, pela quantidade inteira (exclui bit de sinal)
    largest_binary_num = (2 ** int_digits) - 1

    # Inicia a lista de saída
    float_converted = []

    # Percorrendo a quantidade de variáveis da função
    for i in range(bounds.shape[0]):
        # Separa a parte inteira da decimal
        signal, int_str, dec_str = values[i][0], values[i][1:int_digits +
                                                           1], values[i][int_digits + 1:]

        # Convertendo sinal
        signal_value = (-1) if signal == '1' else (1)

        # Converter a parte inteira para número
        int_str = "".join(str(c) for c in int_str)
        int_num = int(int_str, 2)

        # Definir o intervalo de conversão (escala)
        scaled_int_num = bounds[i][0] + \
            (int_num / largest_binary_num) * (bounds[i][1] - bounds[i][0])

        # Converter a parte decimal para número pela fórmula inversa da decimal
        dec_num = np.sum([int(x) * 2 ** (- (k + 1))
                         for k, x in enumerate(dec_str)])

        # Recuperando o número float por completo
        float_converted.append(signal_value * (scaled_int_num + dec_num))

    return float_converted


########################################
#              0 - Geração             #
########################################

# Geração Aleatória da População
def generate_population(
    size_population=10,
    bounds=np.array([[-1.0, 1.0]]),
    int_size=11,
    dec_size=20
):
    """Função responsável por gerar uma nova população,
    em bitstring (vetor de bits), advindos do float.

    OBS: O primeiro bit da parte inteira é de sinal!
    (Considere como tamanho total = 1 + int_digits + dec_digits)

    Args: 
        size_population (int): tamanho da população (padrão: 10)
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [[-1.0, 1.0]])
        int_digits (int): quantidade máxima de dígitos na parte inteira (padrão: 11)
        dec_digits (int): quantidade máxima de dígitos na parte decimal (padrão: 20)
    
    Returns:
        population (np.ndarray[[bitstring, ...], ...]): população inicial gerada
    """

    # Geração de Float aleatório para cada variável
    individual_vals = np.random.uniform(
        bounds[:, 0], bounds[:, 1], (size_population, bounds.shape[0]))

    # Conversão de Float para bitstring e salva como indivíduo
    population = np.array([__float_to_binary(values=individual, bounds=bounds, int_digits=int_size, dec_digits=dec_size)
                          for individual in individual_vals])

    return population


########################################
#              1 - Seleção             #
########################################

# Seleção por Roleta: Problemas de MAXIMIZAÇÃO
def __roulette_selection(
    cur_population: np.ndarray,
    fitness: np.ndarray
):
    """Seleção de indivíduo pelo método da Roleta.

    Args:
        cur_population (np.ndarray[[bitstring, ...], ...]): vetor contendo a população atual para seleção
        fitness (np.ndarray[float, ...]): vetor contendo todos os valores de aptidão da população
        
    Returns:
        new_population (np.ndarray[[bitstring, ...], ...]): vetor contendo a nova população selecionada
        selected_fitness (np.ndarray[float, ...]): vetor contendo todos os valores de aptidão da população selecionada
    """

    # Determinar a porção da roleta para cada indivíduo no intervalo [start; end]
    proportions = fitness / (np.sum(fitness))

    # Posição dos indivíduos selecionados aleatoriamente com base nas porções
    idx_selected = np.random.choice(
        cur_population.shape[0], size=cur_population.shape[0], p=proportions)

    return cur_population[idx_selected], fitness[idx_selected]


# Seleção por Torneio: Problemas de MINIMIZAÇÃO
def __tournament_selection(
    cur_population: np.ndarray,
    fitness: np.ndarray,
    size=3
):
    """Seleção de indivíduo pelo método do Torneio.

    Args:
        cur_population (np.ndarray[[bitstring, ...], ...]): vetor contendo a população atual para seleção
        fitness (np.ndarray[float, ...]): vetor contendo todos os valores de aptidão da população
        size (int): número de indivíduos selecionados aleatóriamente (padrão: 3)
        
    Returns:
        new_population (np.ndarray[[bitstring, ...], ...]): vetor contendo a nova população selecionada
        selected_fitness (np.ndarray[float, ...]): vetor contendo todos os valores de aptidão da população selecionada
    """

    # Criação do vetor para nova população, com base no tamanho da população atual
    new_population = np.empty_like(cur_population)
    selected_fitness = np.empty_like(fitness)

    # Percorrendo o vetor da população atual (número de linhas)
    for i in range(cur_population.shape[0]):
        # Escolha aleatória dos indivíduos candidatos
        idx_candidates = np.random.choice(
            cur_population.shape[0], size, replace=False)

        # Escolha do vencedor com base no MENOR valor obtido
        idx_winner = np.argmin(fitness[idx_candidates])

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
        parent1 (np.ndarray[bitstring, ...]): vetor representando o primeiro indivíduo
        parent2 (np.ndarray[bitstring, ...]): vetor representando o segundo indivíduo
        crossover_rate (float): float que representa a taxa acontecimento de crossover (padrão: 0.8)
        
    Returns:
        child1 (np.ndarray[bitstring, ...]): vetor representando o primeiro filho
        child2 (np.ndarray[bitstring, ...]): vetor representando o segundo filho
    """

    # Cria os vetores iniciais para abrigar os filhos
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)

    # Percorrendo a quantidade de variáveis
    for i in range(parent1.shape[0]):

        # Para ocorrer o crossover, um número aleatório deve ser menor ou igual a taxa
        if np.random.rand() <= crossover_rate:
            # Sorteia um ponto de corte
            crossover_point = np.random.randint(0, parent1[i].shape[0])

            # Realização do crossover
            child1[i] = np.concatenate(
                (parent1[i][:crossover_point], parent2[i][crossover_point:]))
            child2[i] = np.concatenate(
                (parent2[i][:crossover_point], parent1[i][crossover_point:]))
        else:
            # Não ocorrência de crossover, apenas mantém os pais
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    return child1, child2


# Mutação
def __mutation(
    individual: np.ndarray,
    mutation_rate=0.2
):
    """Aplicação de mutação em um indivíduo.

    Args:
        individual (np.ndarray[bitstring, ...]): vetor representando o indivíduo a sofrer mutação
        mutation_rate (float): float que representa a taxa de mutação (padrão: 0.2)
        
    Returns:
        mutant (np.ndarray[bitstring, ...]): vetor representando o indivíduo com mutação
    """

    # Cria o vetor inicial para abrigar o mutante
    mutant = np.copy(individual)

    # Percorrendo cada bitstring do indivíduo
    for i in range(mutant.shape[0]):
        # Percorrendo cada caractere na bitstring
        for j in range(mutant.shape[1]):
            # Verifica se o número aleatório é menor ou igual à taxa de mutação
            if np.random.uniform() <= mutation_rate:
                # Troca o bit dependendo do valor
                mutant[i, j] = '1' if mutant[i, j] == '0' else '0'

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
        selected_population (np.ndarray[[bitstring, ...], ...]): vetor com a população selecionada
        selected_fitness (np.ndarray[float, ...]): vetor contendo todos os valores de aptidão da população
        elitism (bool): considerar ou não o elitismo (padrão: False)
        elite_size (int, opcional se 'elitism=False'): quantidade de indivíduos para elitismo (padrão: 3)
        crossover_rate (float): taxa de crossover (padrão: 0.8)
        mutation_rate (float): taxa de mutação (padrão: 0.2)
        
    Returns:
        new_population (np.ndarray[[bitstring, ...], ...]): vetor com a nova população
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
    target_value: float = float('inf'),
    is_min=True,
    bounds=np.array([[-1.0, 1.0]]),
    int_size=11,
    dec_size=20,
    max_gen=10000,
    max_patience=20,
    size_tournament: int = 3,
    elitism=False,
    elite_size: int = 3,
    crossover_rate=0.8,
    mutation_rate=0.2
):
    """Aplicação do Algoritmo Genético, a partir
    de uma população de bitstring, para min/max de uma 
    função multivariável.

    Seleção por Torneio para Minimização.
        -> 'size_tournament': define o tamanho do torneio

    Seleção por Roleta para Maximização.

    Args:
        initial_population (np.ndarray[[bitstring, ...], ...]): matriz de bitstrings da população inicial
        fitness_func (callable): função de avaliação de aptidão
        target_value (float, opcional): valor alvo estipulado para função de avaliação (padrão: infinito)
        is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
        int_size (int): quantidade máxima de dígitos na parte inteira (padrão: 11)
        dec_size (int): quantidade máxima de dígitos na parte decimal (padrão: 20)
            -> O primeiro bit da parte inteira é de sinal! (tamanho total = 1 + int_digits + dec_digits)
        max_gen (int): número máximo de gerações possíveis (padrão: 10000)
        max_patience (int): número máximo de iterações em que não houve melhora (padrão: 100)
        size_tournament (int, opcional se 'is_min=False'): número de indivíduos selecionados aleatóriamente para o torneio (padrão: 3)
        elitism (bool): considerar ou não o elitismo (padrão: False)
        elite_size (int, opcional se 'elitism=False'): quantidade de indivíduos para elitismo (padrão: 3)
        crossover_rate (float): taxa de crossover (padrão: 0.8)
        mutation_rate (float): taxa de mutação (padrão: 0.2)

    Returns:
        best_population (np.ndarray[[float, ...], ...]): lista com as melhores populações obtidas ao decorrer das gerações
        best_fitness (np.ndarray[[float], ...]): lista com as melhores aptidões obtidas ao decorrer das gerações
        all_population (np.ndarray[[float, ...], ...]): lista com todas as populações obtidas, sendo a última a melhor possível
        all_fitness (np.ndarray[[float, ...], ...]): lista com todas as aptidões obtidas, sendo a última a melhor possível
        generation (int): número de gerações decorridas
    """

    print(f"{'-'*50}")
    print(f"{'Algoritmo Genético':^50}")
    print(f"{'-'*50}")

    # Avaliação da população inicial
    cur_fitness = np.array([fitness_func(__binary_to_float(individual, bounds=bounds,
                           int_digits=int_size, dec_digits=dec_size)) for individual in initial_population])

    # Recuperando o melhor da população inicial e definindo valores iniciais
    best_fitness = [np.min(cur_fitness) if is_min else np.max(cur_fitness)]
    best_population = np.array([__binary_to_float(individual, bounds=bounds, int_digits=int_size, dec_digits=dec_size) for individual in [
                               initial_population[np.argmin(cur_fitness)] if is_min else initial_population[np.argmax(cur_fitness)]]]).reshape((1, bounds.shape[0]))
    all_population = np.array([__binary_to_float(individual, bounds=bounds, int_digits=int_size, dec_digits=dec_size)
                              for individual in initial_population]).reshape((1, initial_population.shape[0], bounds.shape[0]))
    all_fitness = np.array([cur_fitness])

    # Início feito a partir da população inicial
    cur_population = initial_population

    # Percorrendo as gerações
    generation = 1
    patience = 1
    while (generation < max_gen and patience != max_patience and (target_value not in best_fitness)):
        # Aumenta o número da geração atual
        generation = generation + 1

        # Minimização
        if is_min:
            # Fase de Seleção: uso do Torneio
            cur_population, cur_fitness = __tournament_selection(
                cur_population=cur_population,
                fitness=cur_fitness,
                size=size_tournament)

            # Fase de Reprodução
            cur_population = __reproduction(
                selected_population=cur_population,
                selected_fitness=cur_fitness,
                elitism=elitism,
                elite_size=elite_size,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate)

            # Fase de Avaliação
            cur_fitness = np.array(
                [fitness_func(__binary_to_float(
                    individual,
                    bounds=bounds,
                    int_digits=int_size,
                    dec_digits=dec_size))
                 for individual in cur_population])

            # Atualização dos valores
            if np.min(cur_fitness) < np.min(best_fitness):
                best_fitness = np.append(
                    best_fitness,
                    np.array([np.min(cur_fitness)]),
                    axis=0)
                best_population = np.append(
                    best_population,
                    [__binary_to_float(
                        individual,
                        bounds=bounds,
                        int_digits=int_size,
                        dec_digits=dec_size)
                     for individual in [cur_population[np.argmin(cur_fitness)]]],
                    axis=0)
                patience = 1
            elif np.min(cur_fitness) == np.min(best_fitness):
                patience = patience + 1

        # Maximização
        else:
            # Fase de Seleção: uso da Roleta
            cur_population, cur_fitness = __roulette_selection(
                cur_population=cur_population,
                fitness=cur_fitness)

            # Fase de Reprodução
            cur_population = __reproduction(
                selected_population=cur_population,
                selected_fitness=cur_fitness,
                elitism=elitism,
                elite_size=elite_size,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate)

            # Fase de Avaliação
            cur_fitness = np.array(
                [fitness_func(__binary_to_float(
                    individual,
                    bounds=bounds,
                    int_digits=int_size,
                    dec_digits=dec_size))
                 for individual in cur_population])

            # Atualização dos valores
            if np.max(cur_fitness) > np.max(best_fitness):
                best_fitness = np.append(
                    best_fitness,
                    np.array([np.max(cur_fitness)]),
                    axis=0)
                best_population = np.append(
                    best_population,
                    [__binary_to_float(
                        individual,
                        bounds=bounds,
                        int_digits=int_size,
                        dec_digits=dec_size)
                     for individual in
                     [cur_population[np.argmax(cur_fitness)]]],
                    axis=0)
                patience = 1
            elif np.max(cur_fitness) == np.max(best_fitness):
                patience = patience + 1

        # Independente se conseguiu melhor resultado, salva nas listas de retorno para pós-visualização
        all_population = np.append(
            all_population,
            [[__binary_to_float(
                individual,
                bounds=bounds,
                int_digits=int_size,
                dec_digits=dec_size)
              for individual in cur_population]],
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
    # Convertendo a taxa de mutação para ser relacionada ao número de bits e da população
    # mutation_rate = 1.0 / (float(1 + int_size + dec_size) * len(bounds))
    
    print("\nFUNÇÃO 2")  # (0.1; 1)
    bounds = np.array([[-10.0, 10.0]])
    t1, t2, t3, t4, t5 = genetic_algorithm(generate_population(
        bounds=bounds), f2, is_min=False, bounds=bounds, max_gen=10000)
    #print('\n', t1, '\n', t2, '\n', t3, '\n', t4, '\n', t5, '\n')


if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
