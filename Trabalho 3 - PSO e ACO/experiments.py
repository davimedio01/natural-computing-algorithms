"""
Trabalho 3 - PSO e ACO
Data de Entrega: 22/06/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np               # Matrizes e Funções Matemáticas
from copy import copy as cp      # Copiar objetos (não somente a referência)
import matplotlib.pyplot as plt  # Criação de Gráficos

# Ex02: PSO e GA
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
    elif type_plot == 'log':
        plt.log(best_values, label='Melhor Aptidão', c='b')
        plt.log(mean_values, label='Aptidão Média', c='r')
    elif type_plot == 'semilog':
        plt.semilogx(best_values, label='Melhor Aptidão', c='b')
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
    #plt.savefig(os.path.join(sub_directory, plot_name))
    plt.show()

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
    #plt.savefig(os.path.join(sub_directory, plot_name))
    plt.show()

    # Encerra as configurações do gráfico
    plt.close()

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
    num_particles = np.random.randint(10, 51)
    max_it = max_gen = 10000
    max_patience = 100
    
    # Definindo os hiperparâmetros do PSO
    VMIN = -5.0
    VMAX = 5.0
    W = 0.7
    AC2 = 2.05
    AC1 = 2.05
        
    # Execução do PSO
    pso = PSO(
        VMIN=VMIN,
        VMAX=VMAX,
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
    
    # Salvando dados do PSO
    plot_pso_experiment(
        filename=filename,
        alg_name_acronym='PSO',
        type_plot='log',
        num_cycle=0,
        num_experiment=0,
        best_values=pso.best_global_fitness,
        mean_values=pso.best_mean_fitness,
    )
    
    # Definindo consistência dos números aleatórios
    np.random.seed(42)
    
    # Definindo os hiperparâmetros do GA
    population_size = 30
    bitstring_size = np.array([11, 20]) # Float 32 bits (IEEE)
    size_tournament = 3
    elitism = False
    elite_size = 3
    crossover_rate = 0.5
    mutation_rate = 0.25
     
    # Execução do GA
    ga = GA()
    ga.generate_population(
        bounds=bounds,
        population_size=population_size,
        bitstring_size=bitstring_size
    )
    ga.optimize(
        fitness_func=fitness_func,
        is_min=is_min,
        max_gen=max_gen,
        max_patience=max_patience,
        size_tournament=size_tournament,
        elitism=elitism,
        elite_size=elite_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )
    
    # Salvando dados do GA
    plot_pso_experiment(
        filename=filename,
        alg_name_acronym='GA',
        type_plot='log',
        num_cycle=0,
        num_experiment=0,
        best_values=ga.best_global_fitness,
        mean_values=ga.all_mean_fitness,
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
    tsp_problem, node_coords = ex02_tsp(tsp_filename=tsp_filename)
    
    # Definindo os hiperparâmetros do ACO-TSP
    alpha = 1         # Peso da trilha de feromônio (tau)
    beta = 5          # Peso do desejo heurístico (eta)
    rho = 0.5         # Taxa de evaporação do feromônio
    Q = 100           # Quantidade de feromônio depositado por uma formiga
    elite_ant = 5     # Número de formigas elitistas
    tau_init = 1e-6   # Trilha de feromônio inicial
    max_it = 100      # Número máximo de iterações
    max_patience = 50 # Número máximo para estagnação ('paciência')

    # Execução do ACO-TSP
    aco_tsp = ACO_TSP(
        alpha=alpha,
        beta=beta,
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

    # Salvando os gráficos para o ACO-TSP
    cities_x_coords, cities_y_coords = node_coords.T
    best_path_x_coords, best_path_y_coords = node_coords[aco_tsp.best_path_nodes].T
    plot_aco_experiment(
        filename=filename,
        tsp_filename=tsp_filename,
        num_cycle=0,
        num_experiment=0,
        best_dist=aco_tsp.best_path_distance,
        cities_x_coords=cities_x_coords,
        cities_y_coords=cities_y_coords,
        best_path_x_coords=best_path_x_coords,
        best_path_y_coords=best_path_y_coords,
    )
    
    
if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
