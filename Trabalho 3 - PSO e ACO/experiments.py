"""
Trabalho 3 - PSO e ACO
Data de Entrega: 22/06/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np # Matrizes e Funções Matemáticas
from copy import copy as cp  # Copiar objetos (não somente a referência)
import sklearn.datasets as sk_data # Recuperar os Datasets
from sklearn.preprocessing import StandardScaler  # Normalização dos Dados de Pré-Processamento com Z Score
from sklearn.model_selection import train_test_split # Separar conjuntos de treinamento, testes e validação
from sklearn.metrics import accuracy_score, confusion_matrix # Métricas de Acurácia e Matriz de Confusão para Experimentos
import matplotlib.pyplot as plt # Criação de Gráficos

# AG, PSO e ACO
from genetic_algorithm import GA
from pso import PSO
from aco import ACO


########################################
#            Experimentos              #
########################################

# Executa os ciclos propostos, cada qual com vários experimentos
def run_cycle_experiments(
    filename: str,
    alg_name_acronym: str,
    max_cycle: int,
    max_exp_per_cycle: int,
):
    """Executa todos os 'max_cycle' ciclos dos experimentos propostos.
    Cada ciclo possui 'max_exp_per_cycle' de experimentos executados.

    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex_PSO' ou 'ex_ACO')
        alg_name_acronym : str
            Nome ou sigla do algoritmo executado (ex: 'PSO' ou 'ACO')
        max_cycle : int
            Número máximo de ciclos de execução.
        max_exp_per_cycle : int
            Número máximo de experimentos executados por ciclo.
        
        
    Notes:
        
    """
        
    # Biblioteca para capturar o tempo de execução
    from time import time

    
    #!########################################################### 
    #! Ciclos de Execução!
    #!###########################################################  
    # 
    best_cycle = 0     # Melhor ciclo de execução
    
    # Variáveis para salvar as tabelas gerais do ciclo
    
    
    for cycle in range(max_cycle):
        print(f"{'-'*75}")
        print(f"{'Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}")
        
        #!###########################################################
        #! Experimentos
        #!###########################################################
        
        # Definindo os mesmos números aleatórios para NumPy durante a execução do código
        np.random.seed(42)

        # 
        best_num_experiment = -1     # Número do melhor experimento
        
        # Variáveis para salvar os dados de cada experimento
        
        for num_experiment in range(max_exp_per_cycle):
                        
            # Registra o tempo inicial de execução
            start_timer = time()
            
            # Para cada experimento,
            
            
            # Registra o tempo total de execução do algoritmo
            total_time = time() - start_timer
            print(f"Tempo de Execução (s): {total_time :.4f}")
            
            # Salvando o melhor exp. local (por ciclo)
            

            # Salvando os dados para as tabelas
            

            print(f"{'-'*50}")            
        
        # Salvando o melhor exp. global (de todos ciclos)
        
        
        # Salvando outros dados relevantes do ciclo
        
        
        print(f"{'-'*75}")
        print(f"{'Fim do Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}\n")
    
    # Salvar os dados para as tabelas
    


# Criar gráfico de um dos experimentos
def plot_experiment(
    filename: str,
    alg_name_acronym: str,
    num_cycle: int,
    num_experiment: int,
):
    """Cria o gráfico X.
    
    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex01')
        alg_name_acronym : str
            Nome ou sigla do algoritmo executado (ex: 'PSO' ou 'ACO')
        num_cycle : int 
            Número do ciclo de execução
        num_experiment : int 
            Número do experimento dentro do ciclo
        
        
    Notes:
        Cria um arquivo de imagem do gráfico com o seguinte nome: {alg_name_acronym}_ciclo{num_cycle}_exp{num_experiment}.png
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
    plot_name = f'{alg_name_acronym}_ciclo{num_cycle}_exp{num_experiment}.png'

    # Definindo os textos (nomes) do gráfico
    title = 'X'
    plt.title(title, loc='center')
    plt.xlabel('X', loc='center')
    plt.ylabel('Y', loc='center')

    # Plotando o gráfico com base nos valores
    #plt.plot(None, label='None', c='r')

    # Adiciona legenda
    plt.legend()

    # Plota e salva o gráfico em um arquivo
    plt.savefig(os.path.join(sub_directory, plot_name))

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
            Nome do arquivo/exercicio (ex: 'ex01')
        alg_name_acronym : str
            Nome ou sigla do algoritmo executado (ex: 'PSO' ou 'ACO')
        type_exp (str): tipo da tabela (ex: 'acerto')
            -> Utilize: 
        rows : list [[dados, ...], ...] 
            Lista com os dados das linhas no total (ex: [['10', '0.1'], ['20', '0.2'])
        
    Notes:
        Cria um arquivo csv da tabela com o seguinte nome: {alg_name_acronym}_{type_exp}.txt 
        Salva em um subdiretório da pasta local do código com o nome: {filename}
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


# Manipular arquivos CSV
def create_csv_cycle(
    filename: str,
    alg_name_acronym: str,
    type_exp: str,
    rows: list
):
    """Escreve os experimentos em um arquivo CSV para futuras tabelas do relatório.

    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex01')
        alg_name_acronym : str
            Nome ou sigla do algoritmo executado (ex: 'PSO' ou 'ACO')
        type_exp (str): tipo da tabela (ex: 'acerto')
            -> Utilize: 
        rows : list [[dados, ...], ...] 
            Lista com os dados das linhas no total (ex: [['10', '0.1'], ['20', '0.2'])
        
    Notes:
        Cria um arquivo csv da tabela com o seguinte nome: {alg_name_acronym}_{type_exp}.csv 
        Salva em um subdiretório da pasta local do código com o nome: {filename}
    """

    # Defininido o sub-diretório dos dados
    import os
    actual_dir = os.path.dirname(__file__)
    sub_directory = os.path.join(actual_dir, f'{filename}/')
    os.makedirs(sub_directory, exist_ok=True)

    # Definindo o nome do arquivo
    table_name = f'{alg_name_acronym}_{type_exp}.csv'

    # Definindo o título da tabela com base no tipo de experimento
    table_title = ''#'Taxa de Aprendizado,Média,Desvio Padrão,Mínimo,Mediana,Máximo'

    # Escrevendo o arquivo com o título
    np.savetxt(fname=os.path.join(sub_directory, table_name), X=rows, fmt='%.4f', header=table_title,
               delimiter=',', comments='', encoding='UTF-8')


def main():
    """Função principal do programa
    
    Restrições:
        
        
    Descrição dos Experimentos:
        - 4 ciclos executados
        - cada ciclo: 25 vezes de execução do algoritmo
        
    
    Formato dos dados salvos:
        - Gráfico (por ciclo):
        - Tabelas (por ciclo):

    """

    #! [Debug] Definição de saída para mostrar as matrizes por completo no console se necessário.
    np.set_printoptions(threshold=np.inf)
    
    # Definindo as condições gerais e comuns de todos exercícios
    max_cycle = 0
    max_exp_per_cycle = 25
    
    ########################################
    #!       Exercício 01: PSO e GA       !#
    ########################################

    # Definindo as condições iniciais do exercício
    filename = 'ex_PSO'
    
    
    ########################################
    #!         Exercício 02: ACO          !#
    ########################################

    # Definindo as condições iniciais do exercício
    filename = 'ex_ACO'
    


if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
