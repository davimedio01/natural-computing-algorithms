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

# Função do Exercício 02
def f2(x: list[float]) -> float:
    return (2 ** (-2 * (((x[0] - 0.1) / 0.9) ** 2))) * ((np.sin(5 * np.pi * x[0])) ** 6)


#####################################################
#            Algoritmos Pré-Genéticos               #
#####################################################

# Subida da Colina
def hill_climbing(
    fitness_func: callable,
    target_value: float = float('inf'),
    is_min=True,
    bounds=np.array([[-1.0, 1.0]]),
    max_gen=10000,
    max_patience=100,
    step_size=0.1,
):
    """Aplicação do algoritmo "Subida da Colina",
    em sua versão padrão, para busca local em uma função multivariável.
    
    Args:
        fitness_func (callable): função de avaliação de aptidão
        target_value (float, opcional): valor alvo estipulado para função de avaliação (padrão: infinito)
        is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
            - Se 'True', irá verificar minimização\n
            - Se 'False', irá verificar maximização
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
        max_gen (int): número máximo de gerações possíveis (padrão: 10000)
        max_patience (int): número máximo de iterações em que não houve melhora (padrão: 100)
        step_size (float): valor de delta X (padrão: 0.1)
        
    Returns:
        best_global_population (np.ndarray[float, ...]): lista com as melhores populações globais obtidas ao decorrer das gerações
        best_local_population (np.ndarray[float, ...]): lista com as melhores populações locais obtidas ao decorrer das gerações
        best_global_fitness (np.ndarray[float, ...]): lista com as melhores aptidões globais obtidas ao decorrer das gerações
        best_local_fitness (np.ndarray[float, ...]): lista com as melhores aptidões locais obtidas ao decorrer das gerações
        all_mean_fitness (np.ndarray[float, ...]): lista com a média das aptidões obtidas ao decorrer das gerações
        generation (int): número de gerações decorridas
    """

    print(f"{'-'*50}")
    print(f"{'Hill-Climbing':^50}")
    print(f"{'-'*50}")

    # Geração do indivíduo (valor) inicial, de forma randômica e dentro do intervalo definido
    initial_population = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

    # Avaliação do indivíduo inicial
    cur_fitness = np.array([fitness_func(initial_population)])
 
    # Recuperando o melhor da população inicial e definindo valores iniciais
    best_global_fitness = [np.min(cur_fitness) if is_min else np.max(cur_fitness)]
    best_local_fitness = [np.min(cur_fitness) if is_min else np.max(cur_fitness)]
    all_mean_fitness = [np.mean(cur_fitness)]
    best_global_population = np.array([individual for individual in np.array([initial_population[np.argmin(
        cur_fitness)] if is_min else initial_population[np.argmax(cur_fitness)]])])
    best_local_population = np.array([individual for individual in [initial_population[np.argmin(
        cur_fitness)] if is_min else initial_population[np.argmax(cur_fitness)]]])

    # Início feito a partir da população inicial
    cur_population = initial_population

    # Percorrendo as gerações
    generation = 1
    patience = 1
    while (generation < max_gen and
           patience != max_patience and
           (target_value not in best_global_fitness)):
        # Aumenta o número da geração atual
        generation = generation + 1
        
        # Gera um vizinho da população inicial, a partir de uma pertubação gaussiana
        neighbor_population = cur_population + np.random.uniform(-step_size, step_size, cur_population.shape)

        # Realiza a avaliação dos valores
        neighbor_fitness = np.array([fitness_func(neighbor_population)])
        
        #print(neighbor_population, ' ', neighbor_fitness)
        
        # Minimização
        if is_min:
            # Verificação da Paciência (otimização)
            if np.min(neighbor_fitness) == np.min(best_global_fitness):
                patience = patience + 1

            # Atualizando valores globais
            if np.min(neighbor_fitness) < np.min(best_global_fitness):
                patience = 1
                cur_population = np.copy(neighbor_population)
                best_global_fitness = np.append(
                    best_global_fitness,
                    np.array([np.min(neighbor_fitness)]),
                    axis=0)
                best_global_population = np.append(
                    best_global_population,
                    np.array([cur_population[np.argmin(neighbor_fitness)]]),
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
                np.array([np.min(neighbor_fitness)]),
                axis=0)

            best_local_population = np.append(
                best_local_population,
                np.array([cur_population[np.argmin(neighbor_fitness)]]),
                axis=0)

        # Maximização
        else:
            # Verificação da Paciência (otimização)
            if np.max(neighbor_fitness) == np.max(best_global_fitness):
                patience = patience + 1

            # Atualizando valores globais
            if np.max(neighbor_fitness) > np.max(best_global_fitness):
                patience = 1
                cur_population = np.copy(neighbor_population)
                best_global_fitness = np.append(
                    best_global_fitness,
                    np.array([np.max(neighbor_fitness)]),
                    axis=0)
                best_global_population = np.append(
                    best_global_population,
                    np.array([cur_population[np.argmax(neighbor_fitness)]]),
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
                np.array([np.max(neighbor_fitness)]),
                axis=0)

            best_local_population = np.append(
                best_local_population,
                np.array([cur_population[np.argmax(neighbor_fitness)]]),
                axis=0)

        # Salvando as médias das aptidões
        all_mean_fitness = np.append(
            all_mean_fitness,
            np.array([np.mean(neighbor_fitness)]),
            axis=0)
        

    print(f"Geração {generation}")
    print(f"Melhor Aptidão: {best_global_fitness[-1]}")
    print(f"Melhor Indivíduo: {best_global_population[-1]}")

    print(f"{'-'*50}")
    print(f"{'Fim do Hill-Climbing':^50}")
    print(f"{'-'*50}")

    return best_global_population, best_local_population, best_global_fitness, best_local_fitness, all_mean_fitness, generation


# # Subida da Colina Iterativo
# def iterated_hill_climbing(
#     func: callable,
#     num_points: int,
#     target_value: float = float('inf'),
#     is_min=True,
#     bounds=np.array([[-1.0, 1.0]]),
#     step_size=0.1,
#     num_steps=1000
# ):
#     """Aplicação do algoritmo "Subida da Colina",
#     em sua versão iterativa, para busca local em uma função.
#     Utiliza, neste caso, VÁRIOS PONTOS da função.\n
    
#     Para apenas um ponto, utilize a "Subida da Colina padrão".

#     Args:
#         func (callable): função de avaliação
#         num_points (int): número de pontos iniciais para aplicar no algoritmo padrão
#         target_value (float, opcional): valor alvo estipulado para função de avaliação
#         is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
#             -> Se 'True', irá verificar minimização\n
#             -> Se 'False', irá verificar maximização
#         bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
#         step_size (float): valor de delta X (padrão: 0.1)
#         num_steps (int): número máximo de iterações (padrão: 1000)
        
#     Returns:
#         current_x_values (np.ndarray[[float, float], ...]): valores de x calculados
#             -> Leve em conta a cada dois valores, sendo o primeiro o ponto inicial e o segundo o melhor ponto obtido a partir do primeiro
#         current_func_values (np.ndarray[[float, float], ...]): valor da função calculados, sendo o último o melhor possível
#         points (int): número de pontos totais utilizados
#         step (int): número de passos necessários
#     """

#     print(f"{'-'*50}")
#     print(f"{'Hill-Climbing Iterativo':^50}")
#     print(f"{'-'*50}")

#     # Inicialização das variáveis para melhor valor encontrado
#     best_x = None
#     best_func = None
#     current_x_values = []
#     current_func_values = []

#     # Realizando os passos iterativos
#     point = 1
#     while (point < num_points and best_func != target_value):
#         # Aumenta o número da iteração (ir para próximo ponto)
#         point = point + 1

#         # Geração aleatória de um ponto, com base no intervalo
#         current_x = np.random.uniform(bounds[0][0], bounds[0][1])

#         # Obtenção do valor da função neste ponto
#         current_func = func(current_x)

#         # Salva os valores de X e Y iniciais no vetor de saída
#         initial_x = current_x
#         initial_func = current_func

#         # Realizar o "Subida da Colina padrão" para este ponto
#         step = 1
#         while (step < num_steps and current_func != target_value):
#             # Aumenta o número da iteração
#             step = step + 1

#             # Calcula o valor da função para o X atual
#             current_func = func(current_x)

#             # Gera um novo ponto X, a partir da pertubação
#             neighbor_x = current_x + np.random.uniform(-step_size, step_size)

#             # Calcula o valor da função para o novo ponto X
#             neighbor_func = func(neighbor_x)

#             # Realiza a avaliação dos valores
#             # Minimização
#             if (is_min):
#                 # Se o valor da função no novo ponto for menor
#                 # do que o valor atual, atualiza o ponto atual
#                 if (neighbor_func < current_func):
#                     current_x = neighbor_x

#             # Maximização
#             else:
#                 # Se o valor da função no novo ponto for maior
#                 # do que o valor atual, atualiza o ponto atual
#                 if (neighbor_func > current_func):
#                     current_x = neighbor_x

#         # Realiza a avaliação dos valores
#         # Minimização
#         if (is_min):
#             # Se o valor da função no novo ponto for menor
#             # do que o valor atual, atualiza o ponto atual
#             if (best_func is None or func(current_x) < best_func):
#                 best_x = current_x
#                 best_func = func(current_x)

#         # Maximização
#         else:
#             # Se o valor da função no novo ponto for maior
#             # do que o valor atual, atualiza o ponto atual
#             if (best_func is None or func(current_x) > best_func):
#                 best_x = current_x
#                 best_func = func(current_x)

#         # Salva os valores de X e Y melhores no vetor de saída
#         initial_x = [initial_x, best_x]
#         initial_func = [initial_func, best_func]

#         current_x_values.append(initial_x)
#         current_func_values.append(initial_func)

#     print(f"{'-'*50}")
#     print(f"{'Fim do Hill-Climbing Iterativo':^50}")
#     print(f"{'-'*50}")

#     # Retorno do melhor ponto encontrado
#     return np.array(current_x_values), np.array(current_func_values), point, step - 1


# # Subida da Colina Probabilístico
# def stochastic_hill_climbing(
#     func: callable,
#     target_value: float = float('inf'),
#     is_min=True,
#     bounds=np.array([[-1.0, 1.0]]),
#     step_size=0.1,
#     num_steps=1000,
#     t_exp=0.01
# ):
#     """Aplicação do algoritmo "Subida da Colina",
#     em sua versão probabilística, para busca local em uma função.
#     Utiliza, neste caso, apenas UM PONTO da função.\n
    
#     Para um ponto, porém menos efetivo, utilize a "Subida da Colina padrão".
#     Para mais pontos, utilize a "Subida da Colina Iterativa".

#     Args:
#         func (callable): função de avaliação
#         target_value (float, opcional): valor alvo estipulado para função de avaliação
#         is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
#             -> Se 'True', irá verificar minimização\n
#             -> Se 'False', irá verificar maximização
#         bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
#         step_size (float): valor de delta X (padrão: 0.1)
#         num_steps (int): número máximo de iterações (padrão: 1000)
#         t_exp (float): controle do decaimento da função exponencial (padrão: 1.5)
        
#     Returns:
#         current_x_values (np.ndarray[float, ...]): valores de x calculados, sendo o último o mínimo/máximo local encontrado
#         current_func_values (np.ndarray[float, ...]): valor da função calculados, sendo o último o melhor possível
#         step (int): número de passos necessários
#     """

#     print(f"{'-'*50}")
#     print(f"{'Hill-Climbing Probabilístico':^50}")
#     print(f"{'-'*50}")

#     # Inicialização aleatória de um ponto da função
#     current_x = np.random.uniform(bounds[0][0], bounds[0][1])

#     # Encontrando o valor da função para o X iniciado
#     current_func = func(current_x)

#     # Salva os valores de X e Y no vetor de saída
#     current_x_values = np.array(current_x, dtype=np.float32)
#     current_func_values = np.array(current_func, dtype=np.float32)

#     # Realizando os passos iterativos
#     step = 1
#     while (step < num_steps and current_func != target_value):
#         # Aumenta o número da iteração
#         step = step + 1

#         # Calcula o valor da função para o X atual
#         current_func = func(current_x)

#         # Gera um novo ponto X, a partir da pertubação
#         neighbor_x = current_x + np.random.uniform(-step_size, step_size)

#         # Calcula o valor da função para o novo ponto X
#         neighbor_func = func(neighbor_x)

#         # Realiza a avaliação dos valores
#         # Minimização
#         if (is_min):
#             # Calcula a probabilidade P do método
#             # prob = (1 / (1 + np.exp((neighbor_func - current_func) / (t_exp + 1e-8))))
#             prob = np.exp((neighbor_func - current_func) / (t_exp + 1e-8))

#             # Caso o resultado seja melhor que o atual
#             if current_func >= neighbor_func:
#                 current_x = neighbor_x
#             # Do coontrário, valida o objetivo com base na probabilidade
#             elif np.random.uniform() < prob:
#                 current_x = neighbor_x

#         # Maximização
#         else:
#             # Calcula a probabilidade P do método
#             # prob = (1 / (1 + np.exp((current_func - neighbor_func) / (t_exp + 1e-8))))
#             prob = np.exp((current_func - neighbor_func) / (t_exp + 1e-8))

#             # Caso o resultado seja melhor que o atual
#             if current_func <= neighbor_func:
#                 current_x = neighbor_x
#             # Do coontrário, valida o objetivo com base na probabilidade
#             elif np.random.uniform() < prob:
#                 current_x = neighbor_x

#         # Salva os valores de X e Y no vetor de saída
#         current_x_values = np.append(current_x_values, neighbor_x)
#         current_func_values = np.append(current_func_values, func(neighbor_x))

#     print(f"{'-'*50}")
#     print(f"{'Fim do Hill-Climbing Probabilístico':^50}")
#     print(f"{'-'*50}")

#     # Retorno do melhor ponto encontrado
#     return current_x_values, current_func_values, step


# Recozimento Simulado
def simulated_annealing(
    fitness_func: callable,
    target_value: float=float('inf'),
    is_min=True,
    bounds=np.array([[-1.0, 1.0]]),
    max_gen=10000,
    max_patience=100,
    step_size=0.1,
    t_initial=0.01,
    beta=0.8
):
    """Aplicação do algoritmo "Recozimento Simulado"
    para busca global em uma função.
    
    Args:
        fitness_func (callable): função de avaliação de aptidão
        target_value (float, opcional): valor alvo estipulado para função de avaliação (padrão: infinito)
        is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
            - Se 'True', irá verificar minimização\n
            - Se 'False', irá verificar maximização
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
        max_gen (int): número máximo de gerações possíveis (padrão: 10000)
        max_patience (int): número máximo de iterações em que não houve melhora (padrão: 100)
        step_size (float): valor de delta X (padrão: 0.1)
        t_initial (float): valor decimal da temperatura inicial (padrão: 0.01)
        beta (float): taxa de resfriamento pelo decremento geométrico (padrão: 0.8)
        
        
    Returns:
        best_global_population (np.ndarray[float, ...]): lista com as melhores populações globais obtidas ao decorrer das gerações
        best_local_population (np.ndarray[float, ...]): lista com as melhores populações locais obtidas ao decorrer das gerações
        best_global_fitness (np.ndarray[float, ...]): lista com as melhores aptidões globais obtidas ao decorrer das gerações
        best_local_fitness (np.ndarray[float, ...]): lista com as melhores aptidões locais obtidas ao decorrer das gerações
        all_mean_fitness (np.ndarray[float, ...]): lista com a média das aptidões obtidas ao decorrer das gerações
        generation (int): número de gerações decorridas
    """

    print(f"{'-'*50}")
    print(f"{'Recozimento Simulado':^50}")
    print(f"{'-'*50}")

    # Geração do indivíduo (valor) inicial, de forma randômica e dentro do intervalo definido
    initial_population = np.random.uniform(
        bounds[:, 0], bounds[:, 1], bounds.shape[0])

    # Avaliação do indivíduo inicial
    cur_fitness = np.array([fitness_func(initial_population)])

    # Recuperando o melhor da população inicial e definindo valores iniciais
    best_global_fitness = [
        np.min(cur_fitness) if is_min else np.max(cur_fitness)]
    best_local_fitness = [
        np.min(cur_fitness) if is_min else np.max(cur_fitness)]
    all_mean_fitness = [np.mean(cur_fitness)]
    best_global_population = np.array([individual for individual in np.array([initial_population[np.argmin(
        cur_fitness)] if is_min else initial_population[np.argmax(cur_fitness)]])])
    best_local_population = np.array([individual for individual in [initial_population[np.argmin(
        cur_fitness)] if is_min else initial_population[np.argmax(cur_fitness)]]])

    # Início feito a partir da população inicial
    cur_population = initial_population

    # Percorrendo as gerações
    generation = 1
    patience = 1
    t_actual = t_initial
    while (generation < max_gen and
           patience != max_patience and
           (target_value not in best_global_fitness)):
        # Aumenta o número da geração atual
        generation = generation + 1

        # Gera um vizinho da população inicial, a partir de uma pertubação gaussiana
        neighbor_population = cur_population + np.random.uniform(-step_size, step_size, cur_population.shape)

        # Realiza a avaliação dos valores
        neighbor_fitness = np.array([fitness_func(neighbor_population)])

        # print(neighbor_population, ' ', neighbor_fitness)

        # Avaliação dos resultados
        delta_fitness = neighbor_fitness - cur_fitness

        # Minimização
        if is_min:
            # Verifica se o vizinho encontrado é melhor que o atual
            if delta_fitness < 0:
                # Salva o novo valor como melhor
                cur_population = np.copy(neighbor_population)
                cur_fitness = np.copy(neighbor_fitness)
            else:
                # Aplica a fórmula da probabilidade
                prob = np.exp(delta_fitness / t_actual * 1e-8)
                if np.random.rand() < prob:
                    cur_population = np.copy(neighbor_population)
                    cur_fitness = np.copy(neighbor_fitness)
                
            # Verificação da Paciência (otimização)
            if (np.min(cur_fitness) == np.min(best_global_fitness)):
                patience = patience + 1

            # Atualizando valores globais
            if (np.min(cur_fitness) < np.min(best_global_fitness)):
                patience = 1
                best_global_fitness = np.append(
                    best_global_fitness,
                    np.array([np.min(cur_fitness)]),
                    axis=0)
                best_global_population = np.append(
                    best_global_population,
                    np.array([cur_population[np.argmin(cur_fitness)]]),
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
            
        # Maximização
        else:
            # Verifica se o vizinho encontrado é melhor que o atual
            if delta_fitness > 0:
                # Salva o novo valor como melhor
                cur_population = np.copy(neighbor_population)
                cur_fitness = np.copy(neighbor_fitness)
            else:
                # Aplica a fórmula da probabilidade
                prob = np.exp(delta_fitness / t_actual * 1e-8)
                if np.random.rand() < prob:
                    cur_population = np.copy(neighbor_population)
                    cur_fitness = np.copy(neighbor_fitness)
                
            # Verificação da Paciência (otimização)
            if (np.max(cur_fitness) == np.max(best_global_fitness)):
                patience = patience + 1

            # Atualizando valores globais
            if (np.max(cur_fitness) > np.max(best_global_fitness)):
                patience = 1
                best_global_fitness = np.append(
                    best_global_fitness,
                    np.array([np.max(cur_fitness)]),
                    axis=0)
                best_global_population = np.append(
                    best_global_population,
                    np.array([cur_population[np.argmax(cur_fitness)]]),
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
            np.array(
                [np.min(cur_fitness) if is_min else np.max(cur_fitness)]),
            axis=0)

        best_local_population = np.append(
            best_local_population,
            np.array([cur_population[np.argmin(cur_fitness)
                        if is_min else np.argmax(cur_fitness)]]),
            axis=0)

        # Atualiza o valor da temperatura
        t_actual *= beta

        # Salvando as médias das aptidões
        all_mean_fitness = np.append(
            all_mean_fitness,
            np.array([np.mean(neighbor_fitness)]),
            axis=0)

    print(f"Geração {generation}")
    print(f"Melhor Aptidão: {best_global_fitness[-1]}")
    print(f"Melhor Indivíduo: {best_global_population[-1]}")

    print(f"{'-'*50}")
    print(f"{'Fim do Recozimento Simulado':^50}")
    print(f"{'-'*50}")

    return best_global_population, best_local_population, best_global_fitness, best_local_fitness, all_mean_fitness, generation


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
    bitstring_size=np.array([11, 20]),
):
    """Função responsável por converter um valor,
    em float, para representação binária,
    com número máximo de casas inteiras e decimas.
    
    OBS: O primeiro bit da parte inteira é de sinal!
    (Considere como tamanho total = 1 + int_digits + dec_digits)

    Args:       
        values (np.ndarray[float, ...]): variáveis em número em float
        bounds (np.ndarray[[float, float], ...]): lista contendo os intervalos das variáveis da função
        bitstring_size (np.ndarray[int, int]): lista a quantidade máx. de dígitos para parte inteira [0] e decimal [1] (padrão: [11, 20])
        
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
        int_str = np.binary_repr(int_part).zfill(bitstring_size[0])

        # Conversão da parte decimal para binário, completando com zeros à direita
        dec_str = ""
        for _ in range(bitstring_size[1]):
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
    bitstring_size=np.array([11, 20]),
):
    """Função responsável por converter um valor,
    em representação binária, para float,
    com número máximo de casas inteiras e decimas.
    
    OBS: O primeiro bit da parte inteira é de sinal!
    (Considere como tamanho total = 1 + int_digits + dec_digits)

    Args:       
        values (np.ndarray[bitstring, ...]): lista com as variáveis em representação binária 
        bounds (np.ndarray[[float, float], ...]): lista contendo os intervalos das variáveis da função
        bitstring_size (np.ndarray[int, int]): lista a quantidade máx. de dígitos para parte inteira [0] e decimal [1] (padrão: [11, 20])
        
    Returns:
        float_converted (list[float, ...]): lista de variáveis convertidas em float
    """

    # Obtém o maior valor possível do binário, pela quantidade inteira (exclui bit de sinal)
    largest_binary_num = (2 ** bitstring_size[0]) - 1

    # Inicia a lista de saída
    float_converted = []

    # Percorrendo a quantidade de variáveis da função
    for i in range(bounds.shape[0]):
        # Separa a parte inteira da decimal
        signal, int_str, dec_str = values[i][0], values[i][1:bitstring_size[0] +
                                                           1], values[i][bitstring_size[0] + 1:]

        # Convertendo sinal
        signal_value = (-1) if signal == '1' else (1)

        # Converter a parte inteira para número
        int_str = "".join(str(c) for c in int_str)
        int_num = int(int_str, 2)

        # Definir o intervalo de conversão (escala)
        scaled_int_num = bounds[i][0] + (int_num / largest_binary_num) * (bounds[i][1] - bounds[i][0])

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
    bitstring_size=np.array([11, 20]),
    population_size=10,
    bounds=np.array([[-1.0, 1.0]]),
):
    """Função responsável por gerar uma nova população,
    em bitstring (vetor de bits), advindos do float.

    OBS: O primeiro bit da parte inteira é de sinal!
    (Considere como tamanho total = 1 + int_digits + dec_digits)

    Args: 
        bitstring_size (np.ndarray[int, int]): lista a quantidade máx. de dígitos para parte inteira [0] e decimal [1] (padrão: [11, 20])
        population_size (int): tamanho da população (padrão: 10)
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [[-1.0, 1.0]])
        
    Returns:
        population (np.ndarray[[bitstring, ...], ...]): população inicial gerada
    """

    # Geração de Float aleatório para cada variável
    individual_vals = np.random.uniform(
        bounds[:, 0], bounds[:, 1], (population_size, bounds.shape[0]))

    # Conversão de Float para bitstring e salva como indivíduo
    population = np.array([__float_to_binary(values=individual, bounds=bounds, bitstring_size=bitstring_size)
                          for individual in individual_vals])

    return population


########################################
#              1 - Seleção             #
########################################

# Seleção por Roleta: Problemas de MAXIMIZAÇÃO
def __roulette_selection(
    cur_population: np.ndarray,
    fitness: np.ndarray,
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
    size=3,
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
    crossover_rate=0.8,
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
    mutation_rate=0.2,
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
    mutation_rate=0.2,
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
    bitstring_size=np.array([11, 20]),
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

    Seleção por Torneio para Minimização.
        -> 'size_tournament': define o tamanho do torneio

    Seleção por Roleta para Maximização.

    Args:
        initial_population (np.ndarray[[bitstring, ...], ...]): matriz de bitstrings da população inicial
        fitness_func (callable): função de avaliação de aptidão
        target_value (float, opcional): valor alvo estipulado para função de avaliação (padrão: infinito)
        is_min (bool): booleana para atribuir algoritmo para min/max (padrão: True)
            - Se 'True', irá verificar minimização\n
            - Se 'False', irá verificar maximização
        bounds (np.ndarray[[float, float], ...]): lista com valores do intervalo da função (padrão: [-1.0 1.0])
        bitstring_size (np.ndarray[int, int]): lista a quantidade máx. de dígitos para parte inteira [0] e decimal [1] (padrão: [11, 20])
        max_gen (int): número máximo de gerações possíveis (padrão: 10000)
        max_patience (int): número máximo de iterações em que não houve melhora (padrão: 100)
        size_tournament (int, opcional se 'is_min=False'): número de indivíduos selecionados aleatóriamente para o torneio (padrão: 3)
        elitism (bool): considerar ou não o elitismo (padrão: False)
        elite_size (int, opcional se 'elitism=False'): quantidade de indivíduos para elitismo (padrão: 3)
        crossover_rate (float): taxa de crossover (padrão: 0.8)
        mutation_rate (float): taxa de mutação (padrão: 0.2)

    Returns:
        best_global_population (np.ndarray[[float, ...], ...]): lista com as melhores populações globais obtidas ao decorrer das gerações
        best_local_population (np.ndarray[[float, ...], ...]): lista com as melhores populações locais obtidas ao decorrer das gerações
        best_global_fitness (np.ndarray[[float], ...]): lista com as melhores aptidões globais obtidas ao decorrer das gerações
        best_local_fitness (np.ndarray[[float], ...]): lista com as melhores aptidões locais obtidas ao decorrer das gerações
        all_mean_fitness (np.ndarray[[float], ...]): lista com a média das aptidões obtidas ao decorrer das gerações
        generation (int): número de gerações decorridas
    
    Notes:
        - O retorno das variáveis abaixo está comentado para otimização, use por sua conta e risco!
            all_population (np.ndarray[[float, ...], ...]): lista com todas as populações obtidas, sendo a última a melhor possível
            all_fitness (np.ndarray[[float, ...], ...]): lista com todas as aptidões obtidas, sendo a última a melhor possível
    """

    print(f"{'-'*50}")
    print(f"{'Algoritmo Genético':^50}")
    print(f"{'-'*50}")

    # Avaliação da população inicial
    cur_fitness = np.array([fitness_func(__binary_to_float(
        individual, bounds=bounds, bitstring_size=bitstring_size)) for individual in initial_population])

    # Recuperando o melhor da população inicial e definindo valores iniciais
    best_global_fitness = [
        np.min(cur_fitness) if is_min else np.max(cur_fitness)]
    best_local_fitness = [
        np.min(cur_fitness) if is_min else np.max(cur_fitness)]
    all_mean_fitness = [np.mean(cur_fitness)]
    best_global_population = np.array([__binary_to_float(individual, bounds=bounds, bitstring_size=bitstring_size) for individual in [
        initial_population[np.argmin(cur_fitness)] if is_min else initial_population[np.argmax(cur_fitness)]]])  # .reshape((1, bounds.shape[0]))
    best_local_population = np.array([__binary_to_float(individual, bounds=bounds, bitstring_size=bitstring_size) for individual in [
        initial_population[np.argmin(cur_fitness)] if is_min else initial_population[np.argmax(cur_fitness)]]])  # .reshape((1, bounds.shape[0]))
    #! ONLY IF WANTED
    # all_population = np.array([__binary_to_float(individual, bounds=bounds, bitstring_size=bitstring_size)
    #                           for individual in initial_population])#.reshape((1, initial_population.shape[0], bounds.shape[0]))
    # all_fitness = np.array([cur_fitness])

    # Início feito a partir da população inicial
    cur_population = initial_population

    # Percorrendo as gerações
    generation = 1
    patience = 1
    while (generation < max_gen and
           patience != max_patience and
           (target_value not in best_global_fitness)):
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
                    bitstring_size=bitstring_size))
                 for individual in cur_population])

            # Atualização dos valores
            if np.min(cur_fitness) == np.min(best_global_fitness):
                patience = patience + 1

            # Atualizando valores globais
            if np.min(cur_fitness) < np.min(best_global_fitness):
                patience = 1
                best_global_fitness = np.append(
                    best_global_fitness,
                    np.array([np.min(cur_fitness)]),
                    axis=0)
                best_global_population = np.append(
                    best_global_population,
                    [__binary_to_float(
                        individual,
                        bounds=bounds,
                        bitstring_size=bitstring_size)
                        for individual in
                        [cur_population[np.argmin(cur_fitness)]]],
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
                np.array([np.min(cur_fitness)]),
                axis=0)

            best_local_population = np.append(
                best_local_population,
                [__binary_to_float(
                    individual,
                    bounds=bounds,
                    bitstring_size=bitstring_size)
                    for individual in [cur_population[np.argmin(cur_fitness)]]],
                axis=0)

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
                    bitstring_size=bitstring_size))
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
                    [__binary_to_float(
                        individual,
                        bounds=bounds,
                        bitstring_size=bitstring_size)
                        for individual in
                        [cur_population[np.argmax(cur_fitness)]]],
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

            best_local_population = np.append(
                best_local_population,
                [__binary_to_float(
                    individual,
                    bounds=bounds,
                    bitstring_size=bitstring_size)
                    for individual in
                    [cur_population[np.argmax(cur_fitness)]]],
                axis=0)

        # Salvando as médias das aptidões
        all_mean_fitness = np.append(
            all_mean_fitness,
            np.array([np.mean(cur_fitness)]),
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
    crossover_text = "{:02d}".format(int(crossover_rate*100)) if crossover_rate > 0 else ''
    mutation_text = "{:02d}".format(int(mutation_rate*100)) if mutation_rate > 0 else ''
    plot_name = f'{alg_acronym}_ciclo{num_cycle}_exp{num_experiment}_pop{population_size}_cr{crossover_text}_mt{mutation_text}.png'

    # Definindo os textos (nomes) do gráfico
    plt.title(
        str(f"Melhor: {best_fitness[-1]} Média: {np.mean(all_mean_fitness)}"),
        loc='center')
    plt.xlabel('Geração', loc='center')
    plt.ylabel('Aptidão', loc='center')

    # Plotando o gráfico com base nos valores
    if type_plot == 'normal':
        plt.plot(best_fitness,  label='Melhor Aptidão', marker='.', linewidth=0.5)
        plt.plot(all_mean_fitness, label='Aptidão Média', marker='*', linewidth=0.5)
    elif type_plot == 'log':
        plt.semilogy(best_fitness,  label='Melhor Aptidão', marker='.', linewidth=0.5)
        plt.semilogy(all_mean_fitness, label='Aptidão Média', marker='*', linewidth=0.5)

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
    
    - Algoritmos Pré-Genéticos: Subida da Colina e Recozimento Simulado
        - Serão executados separadamente e seguindo as restrições abaixo.
        
        - Passos:
            - Para cada taxa de variação inicial, realiza os experimentos e obtém a melhor taxa
            - Utiliza essa taxa em outro ciclo de experimentos e obtém os dados finais
                
        Obs: Os nomes de saída são adaptados para serem iguais ao do Algoritmo Genético (evitar conflitos nos parâmetros)
            - Exemplo: 'passo de iteração' <=> 'geração'
        
        - Restrições:
            - 2 ciclos executados, sendo um para cada ponto na descrição
            - cada ciclo: 25 vezes de execução do algoritmo
            - tipo de problema (min/máx): maximização
            - num. máx. gerações: 1000
            - num. máx. 'paciência': 20
            - um valor de população inicial: gerado aleatoriamente do intervalo da função
            - taxa de variação inicial: 0.1, 0.2, 0.3, 0.4
            - [Recozimento Simulado] 'T': 0.01, 0.02, 0.03, 0.04
            - [Recozimento Simulado] beta: 0.8, 0.7, 0.6, 0.5
    
    - Algoritmo Genético
        - Passos:
            - Fixa taxa de crossover e mutação para selecionar um tamanho de população para os próximos
            - Varia em par de taxas de crossover e mutação
        
        - Restrições:
            - 2 ciclos executados, sendo um para cada ponto na descrição
            - cada ciclo: 25 vezes de execução do algoritmo
            - float p/bitstrings: 1 bit de sinal + 11 bits parte inteira + 20 bits parte decimal (32 bits)
            - tipo de problema (min/máx): maximização
            - num. máx. gerações: 1000
            - num. máx. 'paciência': 20
            - quatro tamanhos de população inicial: 20, 30, 40, 50
            - método de seleção: roleta (maximização) ou torneio (minimização)
            - crossover: 0.5, 0.6, 0.7, 0.8
            - mutação: 0.1, 0.2, 0.3, 0.4
            - sem elitismo
            
    - OBS: O critério de convergência para melhor população é a menor média de gerações entre as populações
        
    Função: g(x) = (2 ** (-2 * (((x - 0.1) / 0.9) ** 2))) * ((sin(5 * pi * x)) ** 6)
        - intervalo definido por: [[-1.0, 1.0]]
        - melhores valores esperados: [0.1] = 1
    
    Formato dos dados salvos:
        - Gráfico (por ciclo): apenas um para cada população, do tipo "melhor aptidão x geração"
        - Tabelas (por ciclo): aptidão, gerações e tempo de execução
    """

    #! [Debug] Definição de saída para mostrar as matrizes por completo no console se necessário.
    np.set_printoptions(threshold=np.inf)

    # Biblioteca para capturar o tempo de execução
    from time import time

    # Nome para arquivos de saída
    filename = 'ex02'

    ########################################
    #!      Algoritmos Pré-Genéticos      !#
    ########################################
    
    # Definição das variáveis comuns aos algoritmos
    fitness_func = f2
    bounds = np.array([[-1.0, 1.0]])
    target_value = float('inf')
    max_cycle = 2
    max_exp_per_cycle = 25
    is_min = False
    is_min = False
    max_gen = 1000
    max_patience = 20

    ########################################
    #!        Hill-Climbing Padrão        !#
    ########################################

    step_size_vals = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    
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

        # Percorre cada taxa de variação
        for rate_idx, step_size in enumerate(step_size_vals):
            # Armazenar as melhores aptidões de uma população
            experiment_best_fitness = []

            # Armazenar as melhores gerações de uma população
            experiment_best_generation = []

            # Armazenar o tempo de execução de uma população
            experiment_exec_time = []
            
            # Obtendo um valor aletório para plotar um gráfico de um dos experimentos
            plot_rand_num = np.random.randint(1, max_exp_per_cycle + 1)

            # print(rate_idx, ' ', population_size, ' ', crossover_rate,' ', mutation_rate)

            # Percorre o número máximo de experimentos
            for num_experiment in range(1, max_exp_per_cycle + 1):
                # Registra o tempo inicial de execução
                start_timer = time()

                # Aplicação do algoritmo
                best_global_population, best_local_population, best_global_fitness, best_local_fitness, all_mean_fitness, generation = hill_climbing(
                    fitness_func=fitness_func,
                    target_value=target_value,
                    is_min=is_min,
                    bounds=bounds,
                    max_gen=max_gen,
                    max_patience=max_patience,
                    step_size=step_size
                )

                # Registra o tempo total de execução do algoritmo
                total_time = time() - start_timer

                # Salvando os dados nas listas
                experiment_best_fitness.append(np.min(best_local_fitness) if is_min else np.max(best_local_fitness))
                experiment_best_generation.append(generation)
                experiment_exec_time.append(total_time)

                # Gerando o gráfico do experimento escolhido aleatoriamente
                if num_experiment == plot_rand_num:
                    # print(best_fitness, '\n', all_mean_fitness)

                    # Plota o gráfico
                    plot_experiment(
                        filename=filename,
                        alg_acronym='HC',
                        type_plot='normal',
                        num_cycle=num_cycle,
                        num_experiment=num_experiment,
                        population_size=step_size,
                        all_mean_fitness=all_mean_fitness,
                        best_fitness=best_global_fitness
                    )

            # Salvando os dados nas listas
            cycle_best_fitness.append([
                step_size,
                0.0,
                0.0,
                np.mean(experiment_best_fitness),
                np.std(experiment_best_fitness),
                np.median(experiment_best_fitness),
                np.max(experiment_best_fitness) if not is_min else np.min(experiment_best_fitness),
                np.min(experiment_best_fitness) if not is_min else np.max(experiment_best_fitness)
            ])
            cycle_best_generation.append([
                step_size,
                0.0,
                0.0,
                np.mean(experiment_best_generation),
                np.std(experiment_best_generation),
                np.median(experiment_best_generation),
                np.min(experiment_best_generation),
                np.max(experiment_best_generation)
            ])
            cycle_exec_time.append([
                step_size,
                0.0,
                0.0,
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
            alg_acronym='HC',
            type_exp='aptidao',
            num_cycle=num_cycle,
            rows=cycle_best_fitness
        )

        # Tabela de Geração
        csv_table_experiment(
            filename=filename,
            alg_acronym='HC',
            type_exp='geracao',
            num_cycle=num_cycle,
            rows=cycle_best_generation
        )

        # Tabela de Tempo de Execução
        csv_table_experiment(
            filename=filename,
            alg_acronym='HC',
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
            step_size_vals = np.array([step_size_vals[idx_min_mean_generation]])
    

    ########################################
    #!        Recozimento Simulado        !#
    ########################################
        
    step_size_vals = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    
    # Definição das variáveis específicas
    t_initial_vals = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    beta_vals = np.array([0.8, 0.7, 0.6, 0.5], dtype=np.float64)

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
        
        # Definição de T e beta dos experimentos descritos
        if num_cycle == 1:
            # Ciclo 1: apenas o primeiro valor
            t_initial_idx = np.zeros(t_initial_vals.shape[0], dtype=np.int32)
            beta_idx = np.zeros(beta_vals.shape[0], dtype=np.int32)
        elif num_cycle == 2:
            # Ciclo 2: varia em pares
            t_initial_idx = np.arange(t_initial_vals.shape[0], dtype=np.int32)
            beta_idx = np.arange(beta_vals.shape[0], dtype=np.int32)

        # Percorre cada taxa de variação
        for rate_idx, step_size in enumerate(step_size_vals):
            # Armazenar as melhores aptidões de uma população
            experiment_best_fitness = []

            # Armazenar as melhores gerações de uma população
            experiment_best_generation = []

            # Armazenar o tempo de execução de uma população
            experiment_exec_time = []

            # Definindo T e beta dos experimentos
            t_initial = t_initial_vals[t_initial_idx[rate_idx]] if t_initial_idx[rate_idx] != -1 else -1.0
            beta = beta_vals[beta_idx[rate_idx]] if beta_idx[rate_idx] != -1 else -1.0
            
            # Obtendo um valor aletório para plotar um gráfico de um dos experimentos
            plot_rand_num = np.random.randint(1, max_exp_per_cycle + 1)

            # print(rate_idx, ' ', population_size, ' ', crossover_rate,' ', mutation_rate)
            
            # Percorre o número máximo de experimentos
            for num_experiment in range(1, max_exp_per_cycle + 1):
                # Registra o tempo inicial de execução
                start_timer = time()

                # Aplicação do algoritmo
                best_global_population, best_local_population, best_global_fitness, best_local_fitness, all_mean_fitness, generation = simulated_annealing(
                    fitness_func=fitness_func,
                    target_value=target_value,
                    is_min=is_min,
                    bounds=bounds,
                    max_gen=max_gen,
                    max_patience=max_patience,
                    step_size=step_size,
                    t_initial=t_initial,
                    beta=beta
                )

                # Registra o tempo total de execução do algoritmo
                total_time = time() - start_timer

                # Salvando os dados nas listas
                experiment_best_fitness.append(np.min(best_local_fitness) if is_min else np.max(best_local_fitness))
                experiment_best_generation.append(generation)
                experiment_exec_time.append(total_time)

                # Gerando o gráfico do experimento escolhido aleatoriamente
                if num_experiment == plot_rand_num:
                    # print(best_fitness, '\n', all_mean_fitness)

                    # Plota o gráfico
                    plot_experiment(
                        filename=filename,
                        alg_acronym='RS',
                        type_plot='normal',
                        num_cycle=num_cycle,
                        num_experiment=num_experiment,
                        population_size=step_size,
                        all_mean_fitness=all_mean_fitness,
                        best_fitness=best_global_fitness,
                        crossover_rate = t_initial,
                        mutation_rate = beta
                    )

            # Salvando os dados nas listas
            cycle_best_fitness.append([
                step_size,
                t_initial,
                beta,
                np.mean(experiment_best_fitness),
                np.std(experiment_best_fitness),
                np.median(experiment_best_fitness),
                np.max(experiment_best_fitness) if not is_min else np.min(experiment_best_fitness),
                np.min(experiment_best_fitness) if not is_min else np.max(experiment_best_fitness)
            ])
            cycle_best_generation.append([
                step_size,
                t_initial,
                beta,
                np.mean(experiment_best_generation),
                np.std(experiment_best_generation),
                np.median(experiment_best_generation),
                np.min(experiment_best_generation),
                np.max(experiment_best_generation)
            ])
            cycle_exec_time.append([
                step_size,
                t_initial,
                beta,
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
            alg_acronym='RS',
            type_exp='aptidao',
            num_cycle=num_cycle,
            rows=cycle_best_fitness
        )

        # Tabela de Geração
        csv_table_experiment(
            filename=filename,
            alg_acronym='RS',
            type_exp='geracao',
            num_cycle=num_cycle,
            rows=cycle_best_generation
        )

        # Tabela de Tempo de Execução
        csv_table_experiment(
            filename=filename,
            alg_acronym='RS',
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
            step_size_vals = np.array([step_size_vals[idx_min_mean_generation]] * t_initial_vals.shape[0])
    
    
    ########################################
    #!        Algoritmo Genético          !#
    ########################################
    
    # Definindo as restrições descritas
    fitness_func = f2
    bounds = np.array([[-1.0, 1.0]])
    target_value = float('inf')
    max_cycle = 2
    max_exp_per_cycle = 25
    bitstring_size = np.array([11, 20])
    is_min = False
    max_gen = 1000
    max_patience = 20
    initial_population = np.array([20, 30, 40, 50], dtype=np.int64)
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
                    population_size=population_size,
                    bounds=bounds
                )

                # Aplicação do algoritmo
                best_global_population, best_local_population, best_global_fitness, best_local_fitness, all_mean_fitness, generation = genetic_algorithm(
                    initial_population=gen_population,
                    fitness_func=fitness_func,
                    target_value=target_value,
                    is_min=is_min,
                    bounds=bounds,
                    bitstring_size=bitstring_size,
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
                experiment_best_fitness.append(np.min(best_local_fitness) if is_min else np.max(best_local_fitness))
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
                np.max(experiment_best_fitness) if not is_min else np.min(experiment_best_fitness),
                np.min(experiment_best_fitness) if not is_min else np.max(experiment_best_fitness)
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
