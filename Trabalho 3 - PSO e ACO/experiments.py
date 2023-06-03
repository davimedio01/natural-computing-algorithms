"""
Trabalho 3 - PSO e ACO
Data de Entrega: 22/06/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np               # Matrizes e Funções Matemáticas
from copy import copy as cp      # Copiar objetos (não somente a referência)
import matplotlib.pyplot as plt  # Criação de Gráficos

# Ex01: PSO e GA
from pso import PSO
from genetic_algorithm import GA

# Ex02: ACO
import tsplib95                  # Para importar arquivos do TSPLIB
from aco_tsp import ACO_TSP


# Exercício 01: Função de Rosenbrock
def ex01_rosenbrock_func_(x: list[float]) -> float:
    """Função de Rosenbrock relacionada para
    otimização de algoritmos em problemas de minimização.

    Args:
        x : list[float]
            Lista contendo os valores das variáveis (X,Y) de entrada para a função \n
    
    Returns:
        result : float
            Resultado de F(X,Y), no formato float \n
    
    Notes:
        
    """
    
    return (1 - x[0]) ** 2 + 100 * (x[1] - (x[0] ** 2)) ** 2


# Exercício 02: Traveling Salesperson Problems (TSPLIB)
def ex02_tsp(tsp_filename: str):
    """Recupera os dados de um arquivo TSP advindo
    do TSPLIB¹ e retorna os dados das distâncias em formato NumPy.
    Abordagem por matriz de adjacência.

    Args:
        tsp_filename : str 
            Nome do arquivo TSP, sem extensão (ex: 'berlin52')
    
    Returns:
        tsp_problem : tsplib95.Problem
            Variável representativa do problema TSP carregado. Utilização da biblioteca 'tsplib95'².\n
            O acesso as informações dos dados pode ser feita pelo método 'render()'.\n
            O conjunto de nós e suas coordenadas podem ser acessadas por 'get_nodes()' e retorna um 'dict'.\n
        node_coords : np.ndarray[[float, float], ...]
            Conjunto de coordenadas do problema (ex: [[1.0, 2.0], [2.0, 4.0]])\n
    
    Notes:
        ¹: Base de Dados TSPLIB disponível em http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ \n
        ²: Biblioteca 'tsplib95' disponível em https://github.com/rhgrant10/tsplib95 \n
    """
    
    # Importando o arquivo TSP com os dados do problema
    tsp_problem = tsplib95.load(tsp_filename + '.tsp')
    
    # Recuperando os nós (pares de coordenadas) do TSP
    nodes = tsp_problem.node_coords.values()
    node_coords = np.array([coords for coords in nodes])
    
    # # Recuperando a quantidade de cidades
    # num_cities = tsp_problem.dimension
    
    return tsp_problem, node_coords


########################################
#            Experimentos              #
########################################

# Criar gráfico de um dos experimentos do PSO
def plot_pso_experiment(
    filename: str,
    alg_name_acronym: str,
    type_plot: str,
    num_cycle: int,
    num_experiment: int,
    best_values: np.ndarray,
    mean_values: np.ndarray,
):
    """Cria o gráfico para os experimentos com PSO e GA.
    
    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex01') \n
        alg_name_acronym : str
            Nome ou sigla do algoritmo executado (ex: 'PSO', 'GA' ou 'ACO') \n
        type_plot : str
            Tipo de plot a ser utilizada (ex: 'normal' ou 'log') \n
        num_cycle : int 
            Número do ciclo de execução \n
        num_experiment : int 
            Número do experimento dentro do ciclo \n
        best_values : np.ndarray 
            Melhores valores de aptidão ao longo das iterações \n
        mean_values : np.ndarray 
            Valores médios de aptidão ao longo das iterações \n
        
    Notes:
        Cria um arquivo de imagem do gráfico com o seguinte nome: {alg_name_acronym}_ciclo{num_cycle}_exp{num_experiment}.png \n
        Salva em um subdiretório da pasta local com o nome: {filename} \n
    """

    # Pacote de plotar gráficos de listas NumPy
    import matplotlib.pyplot as plt

    # Defininido o sub-diretório dos dados
    import os
    actual_dir = os.path.dirname(__file__)
    sub_directory = os.path.join(actual_dir, f'{filename}/')
    os.makedirs(sub_directory, exist_ok=True)

    # Definindo o nome do arquivo
    plot_name = f'{alg_name_acronym}_ciclo{num_cycle}_exp{num_experiment}.png'

    # Definindo os textos (nomes) do gráfico
    title = 'Execução do ' + alg_name_acronym
    plt.title(title, loc='center')
    plt.xlabel('Iteração', loc='center')
    plt.ylabel('Aptidão', loc='center')

    # Plotando o gráfico com base nos valores
    if type_plot == 'normal':
        plt.plot(best_values, label='Melhor Aptidão', c='b')
        plt.plot(mean_values, label='Aptidão Média', c='r')
    elif type_plot == 'semilog':
        plt.semilogy(best_values, label='Melhor Aptidão', c='b')
        plt.semilogy(mean_values, label='Aptidão Média', c='r')

    # Adiciona legenda
    plt.legend()

    # Plota e salva o gráfico em um arquivo
    plt.savefig(os.path.join(sub_directory, plot_name))
    #plt.show()

    # Encerra as configurações do gráfico
    plt.close()


# Criar gráfico de um dos experimentos do ACO
def plot_aco_experiment(
    filename: str,
    tsp_filename: str,
    num_cycle: int,
    num_experiment: int,
    best_dist: np.ndarray = None,
    cities_x_coords: np.ndarray = None,
    cities_y_coords: np.ndarray = None,
    best_path_x_coords: np.ndarray = None,
    best_path_y_coords: np.ndarray = None,
):
    """Cria o gráfico para os experimentos com PSO e GA.
    
    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex01') \n
        tsp_filename : str
            Nome do arquivo do TSP, sem extensão (ex: 'berlin52') \n
        num_cycle : int 
            Número do ciclo de execução \n
        num_experiment : int 
            Número do experimento dentro do ciclo \n
        best_itr : int
            Valores da melhor iteração do ACO. \n
        best_dist : np.ndarray 
            Valores das melhores distâncias por iteração do ACO. \n
        cities_x_coords : np.ndarray 
            Valores X das coordenadas das cidades do TSP. \n
        cities_y_coords : np.ndarray 
            Valores Y das coordenadas das cidades do TSP. \n
        best_path_x_coords : np.ndarray 
            Valores X das coordenadas do melhor caminho do ACO. \n
        best_path_y_coords : np.ndarray 
            Valores Y das coordenadas do melhor caminho do ACO. \n
        
    Notes:
        Cria um arquivo de imagem do gráfico com o seguinte nome: {tsp_filename}_{type_exp}_ciclo{num_cycle}_exp{num_experiment}.png \n
        Salva em um subdiretório da pasta local com o nome: {filename} \n
    """

    # Pacote de plotar gráficos de listas NumPy
    import matplotlib.pyplot as plt

    # Defininido o sub-diretório dos dados
    import os
    actual_dir = os.path.dirname(__file__)
    sub_directory = os.path.join(actual_dir, f'{filename}/')
    os.makedirs(sub_directory, exist_ok=True)
    
    #######################################
    #! Gráfico de Distâncias por Iteração #
    #######################################
    
    # Configurando o tamanho do gráfico para comportar as cidades, se necessário
    #plt.figure(figsize=(8, 6))        
    
    # Definindo o nome do arquivo
    plot_name = f'{tsp_filename}_itr_ciclo{num_cycle}_exp{num_experiment}.png'
    
    # Plotando as distâncias obtidas, por iteração
    plt.plot(best_dist, c='b')
    
    # Definindo o título do gráfico
    suptitle = 'ACO-TSP'
    title = f'Distância Mínima: {best_dist[-1]:.4f} - Iteração {np.argmin(best_dist)}'
    
    # Definindo os nomes do gráfico
    plt.suptitle(suptitle)
    plt.title(title, loc='center')
    plt.xlabel('Iteração', loc='center')
    plt.ylabel('Distância do Menor Caminho', loc='center')

    # Plota e salva o gráfico em um arquivo
    plt.savefig(os.path.join(sub_directory, plot_name))
    #plt.show()

    # Encerra as configurações do gráfico
    plt.close()
    
    
    ##############################################################
    #! Gráfico de Coordenadas das Cidades com a Distância Mínima #
    ##############################################################
    
    # Configurando o tamanho do gráfico para comportar as cidades, se necessário
    #plt.figure(figsize=(8, 6))
    
    # Definindo o nome do arquivo
    plot_name = f'{tsp_filename}_path-city_ciclo{num_cycle}_exp{num_experiment}.png'
    
    # Plotando as cidades, de acordo com as coordenadas
    plt.scatter(cities_x_coords, cities_y_coords, c='b', s=15, marker='o') # Coordenadas das cidades em pontos
    
    # Plotando o caminho mínimo, de acordo com o resultado
    plt.plot(best_path_x_coords, best_path_y_coords, c='r', linewidth=0.8, linestyle="--")
    plt.plot([best_path_x_coords[-1], best_path_x_coords[0]], [best_path_y_coords[-1], best_path_y_coords[0]], c='r', linewidth=0.8, linestyle="--")
    
    # Definindo o título do gráfico
    suptitle = f'ACO-TSP: Caminho Mínimo para {tsp_filename}'
    title = f'Distância Mínima: {best_dist[-1]:.4f} - Iteração {np.argmin(best_dist)}'

    # Definindo os nomes do gráfico
    plt.xlabel('Latitude', loc='center')
    plt.ylabel('Longitude', loc='center')

    # Definindo o título do gráfico
    plt.suptitle(suptitle)
    plt.title(title, loc='center')
    
    # Plota e salva o gráfico em um arquivo
    plt.savefig(os.path.join(sub_directory, plot_name))
    #plt.show()

    # Encerra as configurações do gráfico
    plt.close()

# Criar arquivos CSV para dados das tabelas
def create_csv_table(
    filename: str,
    alg_name_acronym: str,
    type_exp: str,
    rows: list,
):
    """Escreve os experimentos em um arquivo CSV para futuras tabelas do relatório.

    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex01') \n
        alg_name_acronym : str
            Nome ou sigla do algoritmo executado (ex: 'PSO', 'GA' ou 'ACO-TSP') \n
        type_exp (str): tipo da tabela (ex: 'acerto')
            -> Utilize: 
        rows : list [[dados, ...], ...] 
            Lista com os dados das linhas no total (ex: [['10', '0.1'], ['20', '0.2']) \n
        
    Notes:
        Cria um arquivo csv da tabela com o seguinte nome: {alg_name_acronym}_{type_exp}.csv \n
        Salva em um subdiretório da pasta local do código com o nome: {filename} \n
    """
    
    # Defininido o sub-diretório dos dados
    import os
    actual_dir = os.path.dirname(__file__)
    sub_directory = os.path.join(actual_dir, f'{filename}/')
    os.makedirs(sub_directory, exist_ok=True)

    # Definindo o nome do arquivo
    table_name = f'{alg_name_acronym}_{type_exp}.csv'
    
    # Definindo o título da tabela com base no tipo de algoritmo
    if alg_name_acronym == 'PSO':
        table_title = 'Num. Partículas,V.Min,V.Máx'
    elif alg_name_acronym == 'GA':
        table_title = 'Tam. da População,Taxa de Crossover,Taxa de Mutação'
    elif alg_name_acronym == 'ACO-TSP':
        table_title = 'Alfa,Beta'
    table_title += ',Média,Desvio Padrão,Mínimo,Mediana,Máximo'

    # Salvando o arquivo CSV
    np.savetxt(fname=os.path.join(sub_directory, table_name), X=rows, fmt='%.4f', header=table_title,
               delimiter=',', comments='', encoding='UTF-8')


# Criar arquivos texto para dados do melhor experimento
def create_txt_best(
    filename: str,
    alg_name_acronym: str,
    type_exp: str,
    rows: list,
):
    """Escreve os experimentos em um arquivo TXT para futuras tabelas do relatório.

    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex01') \n
        alg_name_acronym : str
            Nome ou sigla do algoritmo executado (ex: 'PSO' ou 'ACO') \n
        type_exp (str): tipo da tabela (ex: 'acerto')
            -> Utilize: 
        rows : list [[dados, ...], ...] 
            Lista com os dados das linhas no total (ex: [['10', '0.1'], ['20', '0.2']) \n
        
    Notes:
        Cria um arquivo txt com o seguinte nome: {alg_name_acronym}_{type_exp}.txt \n
        Salva em um subdiretório da pasta local do código com o nome: {filename} \n
    """
    
    # Defininido o sub-diretório dos dados
    import os
    actual_dir = os.path.dirname(__file__)
    sub_directory = os.path.join(actual_dir, f'{filename}/')
    os.makedirs(sub_directory, exist_ok=True)

    # Definindo o nome do arquivo
    table_name = f'{alg_name_acronym}_{type_exp}.txt'
    
    # Salvando no txt
    with open(os.path.join(sub_directory, table_name), 'w') as file:
        for item in rows:
            # Escreve cada item da lista (um ciclo) em uma "linha"
            file.write("%s\n" % item)


def main():
    """Função principal do programa
    
    Restrições:
        - Hiperparâmetros definidos pelo estado da arte dos algoritmos
        
    Descrição dos Experimentos:
        - Execução única de cada algoritmo para seu respectivo problema \n
        - Parâmetros do PSO \n
            -- 
        - Parâmetros do AG \n
            -- 
        - Parâmetros do ACO \n
            -- 
    
    Formato dos dados salvos:
        - Gráficos \n
            -> Exercício 01 (PSO): um gráfico contendo os valores mínimo e médio de F(X,Y) ao longo das iterações \n
                -- Realizar o mesmo para o Algoritmo Genético \n
            -> Exercício 02 (ACO): dois gráficos \n
                -- O primeiro com a distância total do menor caminho encontrado ao longo das iterações \n
                -- O segundo contendo as cidades e o traçado do menor caminho encontrado \n

    """

    #! [Debug] Definição de saída para mostrar as matrizes por completo no console se necessário.
    np.set_printoptions(threshold=np.inf)
    
    # Definindo condições gerais de todos os exercícios
    max_cycle = 4
    max_exp_per_cycle = 25
    
    ########################################
    #!       Exercício 01: PSO e GA       !#
    ########################################
    print(f"{'-'*75}")
    print(f"{'Exercício 01':^75}")
    print(f"{'-'*75}")
    
    # Definindo consistência dos números aleatórios
    np.random.seed(42)
    
    # Definindo as condições iniciais e comuns do exercício
    filename = 'ex1_PSO'
    fitness_func = ex01_rosenbrock_func_
    is_min = True
    bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]])
    num_particles = np.random.randint(10, 51) # Um tamanho de particula, aleatório, para todos os experimentos
    max_it = max_gen = 1000
    max_patience = 100
    
    #!###########################################################
    #! PSO
    #!###########################################################
    # Definindo os hiperparâmetros do PSO
    VMIN = np.array([-5.0, -2.5, -10.0, -1.0])
    VMAX = np.array([5.0, 2.5, 10.0, 1.0])
    W = 0.7
    AC2 = 2.05
    AC1 = 2.05
    
    #!###########################################################
    #! Ciclos de Execução!
    #!###########################################################
    # Variáveis para salvar o gráfico de convergência do melhor algoritmo executado
    best_global_pso = None       # Relativo a todos os ciclos
    best_global_fitness = np.inf # Melhor aptidão obtida
    best_cycle = 0               # Melhor ciclo de execução
    
    # Variáveis para dados das tabelas
    cycle_fitness = []    # Aptidões
    cycle_itr = []        # Iterações/Gerações
    cycle_exec_time = []  # Tempo de Execução
    
    for cycle in range(max_cycle):
        print(f"{'-'*75}")
        print(f"{'Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}")
        
        #!###########################################################
        #! Experimentos
        #!###########################################################
        
        # Definindo os mesmos números aleatórios para NumPy durante a execução do código
        np.random.seed(42)
        
        # Variáveis auxiliares para salvar o melhor local (a cada 25 exp.)
        best_local_pso = None       # Relativo a cada ciclo
        best_local_fitness = np.inf # Melhor aptidão do conj. exp.
        best_num_exp = -1           # Número do melhor experimento
        
        # Variáveis para salvar os dados de cada experimento (linhas das tabelas)
        experiment_fitness = []     # Valor de aptidão
        experiment_itr = []         # Iterações/Gerações
        experiment_exec_time = []   # Tempo de Execução

        for num_experiment in range(max_exp_per_cycle):        
            # Execução do PSO
            pso = PSO(
                VMIN=VMIN[cycle],
                VMAX=VMAX[cycle],
                W=W,
                AC1=AC1,
                AC2=AC2
            )
            pso.optimize(
                fitness_func=fitness_func,
                is_min=is_min,
                bounds=bounds,
                num_particles=num_particles,
                max_it=max_it,
                max_patience=max_patience
            )
            
            # Salvando o melhor PSO local (por ciclo)
            if (is_min and pso.best_global_fitness[-1] < best_local_fitness) or (not is_min and pso.best_global_fitness[-1] > best_local_fitness):
                best_local_pso = cp(pso)
                best_local_fitness = cp(pso.best_global_fitness[-1])
                best_num_exp = cp(num_experiment)

            # Salvando dados para as tabelas
            experiment_fitness.append(pso.best_global_fitness[-1])
            experiment_itr.append(pso.itr)
            experiment_exec_time.append(pso.exec_time)
            
            print(f"{'-'*50}") 
        
        # Salvando o melhor PSO global (de todos os ciclos)
        if (is_min and best_local_fitness < best_global_fitness) or (not is_min and best_local_fitness > best_global_fitness):
            best_global_pso = cp(best_local_pso)
            best_global_fitness = cp(best_local_fitness)
            best_cycle = cp(cycle)
            
        # Salvando dados do ciclo para tabela
        cycle_fitness.append([  # Aptidão
            num_particles, # Número de Partículas
            VMIN[cycle],   # Velocidade Mínima
            VMAX[cycle],   # Velocidade Máxima
            np.mean(experiment_fitness), # Média
            np.std(experiment_fitness), # Desvio Padrão
            np.min(experiment_fitness), # Mínimo
            np.median(experiment_fitness), # Mediana
            np.max(experiment_fitness), # Máximo
        ])
        
        cycle_itr.append([  # Iterações/Gerações
            num_particles, # Número de Partículas
            VMIN[cycle],   # Velocidade Mínima
            VMAX[cycle],   # Velocidade Máxima
            np.mean(experiment_itr), # Média
            np.std(experiment_itr), # Desvio Padrão
            np.min(experiment_itr), # Mínimo
            np.median(experiment_itr), # Mediana
            np.max(experiment_itr),  # Máximo
        ])
        
        cycle_exec_time.append([  # Tempo de Execução
            num_particles,  # Número de Partículas
            VMIN[cycle],   # Velocidade Mínima
            VMAX[cycle],   # Velocidade Máxima
            np.mean(experiment_exec_time),  # Média
            np.std(experiment_exec_time),  # Desvio Padrão
            np.min(experiment_exec_time),  # Mínimo
            np.median(experiment_exec_time),  # Mediana
            np.max(experiment_exec_time),  # Máximo
        ])
        
        print(f"{'-'*75}")
        print(f"{'Fim do Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}\n")
    
    # Salvando os dados gráficos
    plot_pso_experiment(
        filename=filename,
        alg_name_acronym='PSO',
        type_plot='semilog',
        num_cycle=best_cycle+1,
        num_experiment=best_num_exp+1,
        best_values=best_global_pso.best_global_fitness,
        mean_values=best_global_pso.best_mean_fitness,
    )
    
    # Salvando os dados das tabelas
    create_csv_table(  # Aptidão
        filename=filename,
        alg_name_acronym='PSO',
        type_exp='aptidao',
        rows=cycle_fitness,
    )
    create_csv_table(  # Iterações/Gerações
        filename=filename,
        alg_name_acronym='PSO',
        type_exp='iteracao',
        rows=cycle_itr,
    )
    create_csv_table(  # Aptidão
        filename=filename,
        alg_name_acronym='PSO',
        type_exp='tempo',
        rows=cycle_exec_time,
    )
    
    
    #!###########################################################
    #! GA
    #!###########################################################
    # Definindo os hiperparâmetros do GA
    population_size = 30
    bitstring_size = np.array([11, 20]) # Float 32 bits (IEEE)
    size_tournament = 3
    elitism = False
    elite_size = 3
    crossover_rate = np.array([0.5, 0.6, 0.7, 0.8])
    mutation_rate = np.array([0.1, 0.2, 0.3, 0.4])
    
    #!###########################################################
    #! Ciclos de Execução!
    #!###########################################################
    # Variáveis para salvar o gráfico de convergência do melhor algoritmo executado
    best_global_ga = None        # Relativo a todos os ciclos
    best_global_fitness = np.inf # Melhor aptidão obtida
    best_cycle = 0               # Melhor ciclo de execução

    # Variáveis para dados das tabelas
    cycle_fitness = []    # Aptidões
    cycle_itr = []        # Iterações/Gerações
    cycle_exec_time = []  # Tempo de Execução

    for cycle in range(max_cycle):
        print(f"{'-'*75}")
        print(f"{'Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}")

        #!###########################################################
        #! Experimentos
        #!###########################################################

        # Definindo os mesmos números aleatórios para NumPy durante a execução do código
        np.random.seed(42)

        # Variáveis auxiliares para salvar o melhor local (a cada 25 exp.)
        best_local_ga = None        # Relativo a cada ciclo
        best_local_fitness = np.inf # Melhor aptidão do conj. exp.
        best_num_exp = -1           # Número do melhor experimento

        # Variáveis para salvar os dados de cada experimento (linhas das tabelas)
        experiment_fitness = []     # Valor de aptidão
        experiment_itr = []         # Iterações/Gerações
        experiment_exec_time = []   # Tempo de Execução

        for num_experiment in range(max_exp_per_cycle):
            # Execução do GA
            ga = GA()
            ga.generate_population(
                bounds=bounds,
                population_size=population_size,
                bitstring_size=bitstring_size,
            )
            ga.optimize(
                fitness_func=fitness_func,
                is_min=is_min,
                max_gen=max_gen,
                max_patience=max_patience,
                size_tournament=size_tournament,
                elitism=elitism,
                elite_size=elite_size,
                crossover_rate=crossover_rate[cycle],
                mutation_rate=mutation_rate[cycle],
            )

            # Salvando o melhor GA local (por ciclo)
            if (is_min and ga.best_global_fitness[-1] < best_local_fitness) or (not is_min and ga.best_global_fitness[-1] > best_local_fitness):
                best_local_ga = cp(ga)
                best_local_fitness = cp(ga.best_global_fitness[-1])
                best_num_exp = cp(num_experiment)

            # Salvando dados para as tabelas
            experiment_fitness.append(ga.best_global_fitness[-1])
            experiment_itr.append(ga.best_generation)
            experiment_exec_time.append(ga.exec_time)

            print(f"{'-'*50}")

        # Salvando o melhor PSO global (de todos os ciclos)
        if (is_min and best_local_fitness < best_global_fitness) or (not is_min and best_local_fitness > best_global_fitness):
            best_global_ga = cp(best_local_ga)
            best_global_fitness = cp(best_local_fitness)
            best_cycle = cp(cycle)

        # Salvando dados do ciclo para tabela
        cycle_fitness.append([  # Aptidão
            population_size, # Tam. da População
            crossover_rate[cycle], # Taxa de Crossover
            mutation_rate[cycle], # Taxa de Mutação
            np.mean(experiment_fitness),  # Média
            np.std(experiment_fitness),  # Desvio Padrão
            np.min(experiment_fitness),  # Mínimo
            np.median(experiment_fitness),  # Mediana
            np.max(experiment_fitness),  # Máximo
        ])

        cycle_itr.append([  # Iterações/Gerações
            population_size,  # Tam. da População
            crossover_rate[cycle],  # Taxa de Crossover
            mutation_rate[cycle],  # Taxa de Mutação
            np.mean(experiment_itr),  # Média
            np.std(experiment_itr),  # Desvio Padrão
            np.min(experiment_itr),  # Mínimo
            np.median(experiment_itr),  # Mediana
            np.max(experiment_itr),  # Máximo
        ])

        cycle_exec_time.append([  # Tempo de Execução
            population_size,  # Tam. da População
            crossover_rate[cycle],  # Taxa de Crossover
            mutation_rate[cycle],  # Taxa de Mutação
            np.mean(experiment_exec_time),  # Média
            np.std(experiment_exec_time),  # Desvio Padrão
            np.min(experiment_exec_time),  # Mínimo
            np.median(experiment_exec_time),  # Mediana
            np.max(experiment_exec_time),  # Máximo
        ])

        print(f"{'-'*75}")
        print(f"{'Fim do Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}\n")

    # Salvando os dados gráficos
    plot_pso_experiment(
        filename=filename,
        alg_name_acronym='GA',
        type_plot='semilog',
        num_cycle=best_cycle+1,
        num_experiment=best_num_exp+1,
        best_values=best_global_ga.best_global_fitness,
        mean_values=best_global_ga.all_mean_fitness,
    )

    # Salvando os dados das tabelas
    create_csv_table(  # Aptidão
        filename=filename,
        alg_name_acronym='GA',
        type_exp='aptidao',
        rows=cycle_fitness,
    )
    create_csv_table(  # Iterações/Gerações
        filename=filename,
        alg_name_acronym='GA',
        type_exp='iteracao',
        rows=cycle_itr,
    )
    create_csv_table(  # Aptidão
        filename=filename,
        alg_name_acronym='GA',
        type_exp='tempo',
        rows=cycle_exec_time,
    )
    
    
    ########################################
    #!       Exercício 02: ACO-TSP        !#
    ########################################
    print(f"{'-'*75}")
    print(f"{'Exercício 02':^75}")
    print(f"{'-'*75}")

    # Definindo consistência dos números aleatórios
    np.random.seed(42)
    
    # Definindo as condições iniciais do exercício
    filename = 'ex2_ACO-TSP'
    
    # Recuperando os dados do arquivo TSP
    tsp_filename = 'berlin52' # Optimal Path: 7542
    _, node_coords = ex02_tsp(tsp_filename=tsp_filename)
    
    # Definindo os hiperparâmetros do ACO-TSP
    alpha = np.array([1.0, 5.0, 2.0, 2.0]) # Peso da trilha de feromônio (tau)
    beta = np.array([5.0, 1.0, 2.0, 5.0])  # Peso do desejo heurístico (eta)
    rho = 0.5          # Taxa de evaporação do feromônio
    Q = 100            # Quantidade de feromônio depositado por uma formiga
    elite_ant = 5      # Número de formigas elitistas
    tau_init = 1e-6    # Trilha de feromônio inicial
    max_it = 150       # Número máximo de iterações
    max_patience = 10  # Número máximo para estagnação ('paciência')

    #!###########################################################
    #! Ciclos de Execução!
    #!###########################################################
    # Variáveis para salvar o gráfico de convergência do melhor algoritmo executado
    best_global_aco_tsp = None       # Relativo a todos os ciclos
    best_global_path_dist = np.inf # Melhor distância obtida
    best_cycle = 0               # Melhor ciclo de execução

    # Variáveis para dados das tabelas
    cycle_fitness = []    # Aptidões
    cycle_itr = []        # Iterações/Gerações
    cycle_exec_time = []  # Tempo de Execução

    for cycle in range(max_cycle):
        print(f"{'-'*75}")
        print(f"{'Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}")

        #!###########################################################
        #! Experimentos
        #!###########################################################

        # Definindo os mesmos números aleatórios para NumPy durante a execução do código
        np.random.seed(42)

        # Variáveis auxiliares para salvar o melhor local (a cada 25 exp.)
        best_local_aco_tsp = None       # Relativo a cada ciclo
        best_local_path_dist = np.inf # Melhor distância do conj. exp.
        best_num_exp = -1           # Número do melhor experimento

        # Variáveis para salvar os dados de cada experimento (linhas das tabelas)
        experiment_fitness = []     # Valor de aptidão
        experiment_itr = []         # Iterações/Gerações
        experiment_exec_time = []   # Tempo de Execução

        for num_experiment in range(max_exp_per_cycle):
            # Execução do ACO-TSP
            aco_tsp = ACO_TSP(
                alpha=alpha[cycle],
                beta=beta[cycle],
                rho=rho,
                Q=Q,
                elite_ant=elite_ant
            )
            aco_tsp.optimize(
                node_coords=node_coords,
                tau_init=tau_init,
                max_it=max_it,
                max_patience=max_patience
            )

            # Salvando o melhor ACO-TSP local (por ciclo)
            if aco_tsp.best_path_distance[-1] < best_local_path_dist:
                best_local_aco_tsp = cp(aco_tsp)
                best_local_path_dist = cp(aco_tsp.best_path_distance[-1])
                best_num_exp = cp(num_experiment)

            # Salvando dados para as tabelas
            experiment_fitness.append(aco_tsp.best_path_distance[-1])
            experiment_itr.append(aco_tsp.itr)
            experiment_exec_time.append(aco_tsp.exec_time)

            print(f"{'-'*50}")

        # Salvando o melhor ACO-TSP global (de todos os ciclos)
        if best_local_path_dist < best_global_path_dist:
            best_global_aco_tsp = cp(best_local_aco_tsp)
            best_global_path_dist = cp(best_local_path_dist)
            best_cycle = cp(cycle)

        # Salvando dados do ciclo para tabela
        cycle_fitness.append([  # Aptidão
            alpha[cycle],  # Alfa
            beta[cycle],  # Beta
            np.mean(experiment_fitness),  # Média
            np.std(experiment_fitness),  # Desvio Padrão
            np.min(experiment_fitness),  # Mínimo
            np.median(experiment_fitness),  # Mediana
            np.max(experiment_fitness),  # Máximo
        ])

        cycle_itr.append([  # Iterações/Gerações
            alpha[cycle],  # Alfa
            beta[cycle],  # Beta
            np.mean(experiment_itr),  # Média
            np.std(experiment_itr),  # Desvio Padrão
            np.min(experiment_itr),  # Mínimo
            np.median(experiment_itr),  # Mediana
            np.max(experiment_itr),  # Máximo
        ])

        cycle_exec_time.append([  # Tempo de Execução
            alpha[cycle], # Alfa
            beta[cycle], # Beta
            np.mean(experiment_exec_time),  # Média
            np.std(experiment_exec_time),  # Desvio Padrão
            np.min(experiment_exec_time),  # Mínimo
            np.median(experiment_exec_time),  # Mediana
            np.max(experiment_exec_time),  # Máximo
        ])

        print(f"{'-'*75}")
        print(f"{'Fim do Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}\n")

    # Salvando os dados gráficos
    cities_x_coords, cities_y_coords = node_coords.T
    best_path_x_coords, best_path_y_coords = node_coords[best_global_aco_tsp.best_path_nodes].T
    plot_aco_experiment(
        filename=filename,
        tsp_filename=tsp_filename,
        num_cycle=best_cycle+1,
        num_experiment=best_num_exp+1,
        best_dist=best_global_aco_tsp.best_path_distance,
        cities_x_coords=cities_x_coords,
        cities_y_coords=cities_y_coords,
        best_path_x_coords=best_path_x_coords,
        best_path_y_coords=best_path_y_coords,
    )

    # Salvando os dados das tabelas
    create_csv_table(  # Aptidão
        filename=filename,
        alg_name_acronym='ACO-TSP',
        type_exp='aptidao',
        rows=cycle_fitness,
    )
    create_csv_table(  # Iterações/Gerações
        filename=filename,
        alg_name_acronym='ACO-TSP',
        type_exp='iteracao',
        rows=cycle_itr,
    )
    create_csv_table(  # Aptidão
        filename=filename,
        alg_name_acronym='ACO-TSP',
        type_exp='tempo',
        rows=cycle_exec_time,
    )
    
    
    
if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
