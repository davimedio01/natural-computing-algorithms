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
from sklearn.metrics import accuracy_score, confusion_matrix # Métricas de Acurácia e Matriz de Confusão para Experimentos

from perceptron import Perceptron

import matplotlib.pyplot as plt # Criação de Gráficos
import pandas as pd # Manipulação e Visualização das Tabelas

#! [Debug]
from sklearn.metrics import classification_report # Métricas Gerais para Teste

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
            target: matriz 1D de forma (num_classe_amostras) contendo o valor da classe das amostras de destino.
         
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
            target: matriz 1D de forma (num_classe_amostras) contendo o valor da classe das amostras de destino.
         
    """

    return sk_data.load_wine(return_X_y=True)


# Exercício 03:


########################################
#            Experimentos              #
########################################


# Executa os ciclos propostos, cada qual com vários experimentos
def run_cycle_experiments(
    filename: str,
    max_cycle: int,
    max_exp_per_cycle: int,
    X: np.ndarray,
    y: np.ndarray,
    initial_learning_rate: np.ndarray,
    max_epoch: int,
    max_patience: int,
):
    """Executa todos os 'max_cycle' ciclos dos experimentos propostos.
    Cada ciclo possui 'max_exp_per_cycle' de experimentos executados.

    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex01')
        max_cycle : int
            Número máximo de ciclos de execução.
        max_exp_per_cycle : int
            Número máximo de experimentos executados por ciclo.
        X : np.ndarray
            Matriz 2D de forma (num_amostras, num_características) com cada linha representando uma amostra e cada coluna representando as características.
        y : np.ndarray
            Vetor de forma (num_classe_amostras) contendo o valor da classe das amostras de destino.   
        initial_learning_rate : np.ndarray (float)
            Vetor contendo os valores iniciais de "learning_rate" para cada ciclo de execução do algoritmo.
        max_epoch : int
            Número máximo de épocas/iterações como critério de parada do algoritmo.
        max_patience : int
            Número máximo de "paciência" como critério de parada do algoritmo.
        
    Notes:
        O pré-processamento do Perceptron, para um exercício, ocorre nesta função.
    """
        
    # Biblioteca para capturar o tempo de execução
    from time import time

    # Obtendo a quantidade de classes do conj. amostras
    n_class = np.unique(y).shape[0]
    
    # Normalizando os dados com Z Score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
        
    # Separando os subconjuntos de treinamento (70%), validação (15%) e teste (15%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    # print(X_train, '\n', y_train, '\n', X_train.shape, ' ', y_train.shape)
    # print(X_val, '\n', y_val, '\n', X_val.shape, ' ', y_val.shape)
    # print(X_test, '\n', y_test, '\n', X_test.shape, ' ', y_test.shape)
    #!########################################################### 
    #! Ciclos de Execução!
    #!###########################################################
    
    # initial_W = np.random.uniform(size=(X.shape[1], n_class))
    # initial_bias = np.random.uniform(size=n_class)
    initial_W = np.zeros(shape=(n_class, X.shape[1]))
    initial_bias = np.zeros(shape=(1, n_class))
    
    print(initial_W)
    print(initial_W.shape)
    print(initial_bias)
    print(initial_bias.shape)
    
    my_dick = Perceptron(
        initial_W, 
        initial_bias, 
        initial_learning_rate[1], 
        n_class, 
        max_epoch,
        max_patience
    )
    my_dick.fit(X_train, y_train, X_val, y_val)
    
    print(my_dick.W)
    print(my_dick.W.shape)
    print(my_dick.bias)
    print(my_dick.bias.shape)
    
    plt.title('Convergência do Perceptron', loc='center')
    plt.xlabel('Época', loc='center')
    # plt.ylabel('Taxas de Acero/Erro', loc='center')
    # plt.plot(my_dick.all_acc_val, label='Acerto', marker='.', linewidth=0.3)
    # plt.plot(my_dick.all_error_val, label='Erro', marker='*', linewidth=0.3)
    plt.ylabel('Erro Acumulado', loc='center')
    plt.plot(my_dick.mse_train, label='Treino', marker='.', linewidth=0.3)
    plt.plot(my_dick.mse_val, label='Validação', marker='*', linewidth=0.3)
    plt.legend()
    plt.show()
    plt.close()
    
    my_dick_predict = my_dick.predict(X_test)
    print(y_test)
    print(y_test.shape)
    print(my_dick_predict)
    print(my_dick_predict.shape)
    
    print(classification_report(y_test, my_dick_predict, zero_division=True))
    
    for cycle in range(max_cycle):
        
        # Gerados por ciclo
        initial_W = np.random.uniform(size=(X.shape[1], n_class))
        initial_bias = np.random.uniform(size=n_class)

        # print(initial_W)
        # print(initial_W.shape)
        # print(initial_bias)
        # print(initial_bias.shape)

        #!###########################################################
        #! Experimentos com Perceptron
        #!###########################################################


    
    
# # Um Experimento: Treinamento do Perceptron
# def run_experiment(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     initial_W: np.ndarray,
#     initial_bias: np.ndarray,
#     initial_learning_rate: np.ndarray,
#     max_epoch: int,
#     max_patience: int
# ):
#     """ Realização dos experimentos com Perceptron

#     Args:
#         X_train : np.ndarray
#             Matriz 2D de forma (num_amostras, num_características) do subconjunto de treinamento
#         y_train : np.ndarray 
#             Vetor de forma (num_classe_amostras) com as saídas desejadas do subconjunto de treinamento
#         X_val : np.ndarray
#             Matriz 2D de forma (num_amostras, num_características) do subconjunto de validação
#         y_val : np.ndarray 
#             Vetor de forma (num_classe_amostras) com as saídas desejadas do subconjunto de validação
#         initial_W : np.ndarray
#             Vetor de pesos iniciais do Perceptron, que é atualizado durante a fase de treinamento (fit).
#         initial_bias : float
#             Valor inicial de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
#         initial_learning_rate : float
#             Taxa inicial de aprendizado para atualização dos pesos.
#         max_epoch : int
#             Número máximo de épocas/iterações como critério de parada.
#         max_patience : int
#             Número máximo de "paciência" como critério de parada do algoritmo.
           
#     Notes:
        
#     """
#     return None


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
        - Valores iniciais de W (pesos) e bias definidos aleatoriamente entre [0, 1)
        - Taxa de "Learning Rate" fixa durante o treinamento do Perceptron
        
        
    Descrição dos Experimentos:
        - 4 ciclos executados
        - cada ciclo: 25 vezes de execução do algoritmo
        - conjunto de amostras iniciais normalizado com Z Score
        - subconjuntos de treinamento, validação e testes com proporções respectivas de: 70%, 15% e 15%
        - num. máx. iterações/épocas: 1000
        - num. máx. "paciência" (estagnação do algoritmo): 10
        - valores iniciais de W (pesos) por ciclo: aleatórios entre 0 e 1
        - taxa inicial de "bias" por ciclo: aleatório entre 0 e 1
        - taxas iniciais de "Learning Rate" para cada ciclo (índice corresponde ao ciclo): [1, 0.1, 0.01, 0.001]
        
    
    Formato dos dados salvos:
        - Gráfico (por ciclo): erro médio quadrático da convergência (MSE do treino e MSE da validação) com a melhor rede do ciclo
        - Tabelas (por ciclo): 
            - melhor rede executada: parâmetros iniciais e finais (peso e bias), learning_rate, matrizes de confusão (treinamento, validação e teste)
            - taxa de acerto/erro (todas as redes executadas): média, desvio padrão, mínimo, mediana, máximo
            - número de épocas (todas as redes executadas): média, desvio padrão, mínimo, mediana, máximo
            - tempo de execução (todas as redes executadas): média, desvio padrão, mínimo, mediana, máximo
    """

    #! [Debug] Definição de saída para mostrar as matrizes por completo no console se necessário.
    np.set_printoptions(threshold=np.inf)

    # Definindo as condições gerais e comuns de todos exercícios
    max_cycle = 4
    max_exp_per_cycle = 25
    max_epoch = 1000
    max_patience = 100
    initial_learning_rate = np.array([1, 0.1, 0.01, 0.001])
    
    ########################################
    #!     Exercício 01: Iris Dataset     !#
    ########################################

    # Definindo as condições iniciais do exercício
    filename = 'ex01'
    X, y = load_iris_dataset()
    
    # Execução dos ciclos do exercício
    run_cycle_experiments(
        filename=filename,
        max_cycle=max_cycle,
        max_exp_per_cycle=max_exp_per_cycle,
        X=X,
        y=y,
        initial_learning_rate=initial_learning_rate,
        max_epoch=max_epoch,
        max_patience=max_patience,
    )

    ########################################
    #!     Exercício 02: Wine Dataset     !#
    ########################################
    # # Definindo as condições iniciais do exercício
    # filename = 'ex02'
    # X, y = load_wine_dataset()
    
    # # Execução dos ciclos do exercício
    # run_cycle_experiments(
    #     filename=filename,
    #     max_cycle=max_cycle,
    #     max_exp_per_cycle=max_exp_per_cycle,
    #     max_epoch=max_epochs,
    #     max_patience=max_patience,
    #     X=X,
    #     y=y,
    #     initial_learning_rate=initial_learning_rate
    # )


    ########################################
    #!            Exercício 03            !#
    ########################################




if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
