"""
Trabalho 2 - Redes Neurais
Data de Entrega: 11/05/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np # Matrizes e Funções Matemáticas
import sklearn.datasets as sk_data # Recuperar os Datasets
from sklearn.preprocessing import StandardScaler  # Normalização dos Dados de Pré-Processamento com Z Score
from sklearn.model_selection import train_test_split # Separar conjuntos de treinamento, testes e validação
from sklearn.metrics import accuracy_score # Métrica de Acurácia para Experimentos

from perceptron import Perceptron # Rede Neural: Perceptron

import matplotlib.pyplot as plt # Criação de Gráficos
import pandas as pd # Manipulação e Visualização das Tabelas


#####################################################
#                  Base de Dados                    #
#####################################################

# Exercício 01: Iris Dataset (https://archive.ics.uci.edu/ml/datasets/Iris)
def load_iris_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Carrega o "Iris" dataset, a partir
    da biblioteca "sklearn". 
    
    - Informações do dataset em: https://archive.ics.uci.edu/ml/datasets/Iris
    
    Returns:
        (data, target) : (np.ndarray, np.ndarray)
            tupla de ndarray do tipo:
            data: matriz 2D de forma (num_amostras, num_características) com cada linha representando uma amostra e cada coluna representando as características.
            target: matriz 1D de forma (num_amostras) contendo as amostras de destino.
         
    """

    return sk_data.load_iris(return_X_y=True)


# Exercício 02: Wine Data Set (https://archive.ics.uci.edu/ml/datasets/Wine)
def load_wine_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Carrega o "Wine" dataset, a partir
    da biblioteca "sklearn". 
    
    - Informações do dataset em: https://archive.ics.uci.edu/ml/datasets/Wine
    
    Returns:
        (data, target) : (np.ndarray, np.ndarray)
            tupla de ndarray do tipo:
            data: matriz 2D de forma (num_amostras, num_características) com cada linha representando uma amostra e cada coluna representando as características.
            target: matriz 1D de forma (num_amostras) contendo as amostras de destino.
         
    """

    return sk_data.load_wine(return_X_y=True)


# Exercício 03:


########################################
#            Experimentos              #
########################################

# Treinamento do Perceptron (Experimento)
def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
    initial_W: np.ndarray,
    initial_bias: np.ndarray,
    initial_learning_rate: np.ndarray,
    max_it: int,
):
    """ Realização dos experimentos com Perceptron

    Args:
        X : np.ndarray
            matriz 2D de forma (num_amostras, num_características) com cada linha representando uma amostra e cada coluna representando as características.
        y : np.ndarray
            matriz 1D de forma (num_amostras) contendo as amostras de destino.
        initial_W : np.ndarray
            Vetor de pesos iniciais do Perceptron, que é atualizado durante a fase de treinamento (fit).
        initial_bias : float
            Valor inicial de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
        initial_learning_rate : float
            Taxa inicial de aprendizado para atualização dos pesos.
        max_it : int
            Número máximo de iterações como critério de parada.
            
    Notes:
        
    """
    return None


# Manipular arquivos CSV (tabelas)
def create_csv_table(
    filename: str,
    rows: np.ndarray
):
    """Escreve os experimentos em um arquivo CSV para tabelas do relatório.

    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex01')
        rows : np.ndarray[[dados, ...], ...] 
            Lista com os dados das linhas no total (ex: [['10', '0.1'], ['20', '0.2'])
        
    Notes:
        Cria um arquivo csv da tabela com o seguinte nome: {}.csv  
        Salva em um subdiretório da pasta local do código com o nome: {filename}
    """

    # Defininido o sub-diretório dos dados
    import os
    actual_dir = os.path.dirname(__file__)
    sub_directory = os.path.join(actual_dir, f'{filename}/')
    os.makedirs(sub_directory, exist_ok=True)

    # Definindo o nome do arquivo
    table_name = f'.csv'

    # Definindo o título da tabela com base no tipo de experimento
    table_header = 'Title,'

    # Escrevendo o arquivo com o título
    np.savetxt(fname=os.path.join(sub_directory, table_name), X=rows, header=table_header,
               delimiter=',', comments='', encoding='UTF-8')  # , fmt='%.4f')


def main():
    """Função principal do programa
    
    Restrições:
        - Para cada ciclo, obtém os melhores pesos e taxas (bias e aprendizado)
        - Após a execução total do ciclo, executa-se mais uma vez com os melhores parâmetros obtidos para salvar os dados
        - Repete o processo para a quantidade total de ciclos
        
    Descrição dos Experimentos:
        - 3 ciclos executados
        - cada ciclo: 15+1 vezes de execução do algoritmo
        - conjunto de dados iniciais normalizado com z score
        - valores aletórios para os subconjuntos de treinamento, validação e testes
        - num. máx. iterações/épocas: 1000
        - valores iniciais de W (pesos): 0
        - taxa inicial de "bias": 0
        - taxa inicial de "Learning Rate": 0.1
        
    
    Formato dos dados salvos:
        - Gráfico (por ciclo): erro médio quadrático da convergência com a última rede do ciclo
        - Tabelas (por exercício): 
            - matrizes de confusão: treinamento, validação e teste
            - parâmetros iniciais e finais das redes: *pesos, *bias, taxa de aprendizado, taxas de acerto e erro
    """

    #! [Debug] Definição de saída para mostrar as matrizes por completo no console se necessário.
    np.set_printoptions(threshold=np.inf)

    # Biblioteca para capturar o tempo de execução
    from time import time

    # Definindo as condições gerais dos exercícios
    max_cycle = 4
    max_exp_per_cycle = 25
    max_it = 1000
    
    ########################################
    #!     Exercício 01: Iris Dataset     !#
    ########################################

    # Definindo as condições iniciais do exercício
    filename = 'ex01'
    X, y = load_iris_dataset()
    W_initial = np.zeros((X.shape[0], X.shape[1]))
    bias_initial = np.array([0.0, 0.1, 0.2, 0.3])
    learning_rate_initial = np.array([0.1, 0.2, 0.3, 0.4])

    print(W_initial)

    ########################################
    #!     Exercício 02: Wine Dataset     !#
    ########################################



    ########################################
    #!            Exercício 03            !#
    ########################################




if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
