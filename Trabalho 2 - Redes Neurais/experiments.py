"""
Trabalho 2 - Redes Neurais
Data de Entrega: 11/05/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np # Matrizes e Funções Matemáticas
from copy import copy as cp  # Copiar objetos (não somente a referência)
import sklearn.datasets as sk_data # Recuperar os Datasets
from sklearn.preprocessing import StandardScaler  # Normalização dos Dados de Pré-Processamento com Z Score
from sklearn.model_selection import train_test_split # Separar conjuntos de treinamento, testes e validação
from sklearn.metrics import accuracy_score, confusion_matrix # Métricas de Acurácia e Matriz de Confusão para Experimentos

from perceptron import Perceptron
from sklearn.preprocessing import OneHotEncoder # Abordagem: "Um vs Todos" (OVR) necessária ao Perceptron

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


# Exercício 03: Blood Transfusion Service Center (https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)
def load_btsc_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Carrega o "Blood Transfusion Service Center" dataset, 
    a partir da biblioteca "sklearn" pelo site OpenML. 
    
    - Informações do dataset em: https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
    
    Returns:
        (data, target) : (np.ndarray, np.ndarray)
            tupla de ndarray do tipo:
            data: matriz 2D de forma (num_amostras, num_características) com cada linha representando uma amostra e cada coluna representando as características.
            target: matriz 1D de forma (num_classe_amostras) contendo o valor da classe das amostras de destino.
         
    """

    return sk_data.fetch_openml(data_id=1464, return_X_y=True, as_frame=False, parser="liac-arff")


########################################
#            Experimentos              #
########################################

# Executa os ciclos propostos, cada qual com vários experimentos
def run_perceptron_cycle_experiments(
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

    # Normalizando os dados com Z Score
    X = StandardScaler().fit_transform(X)
    
    # Aplicando "One Hot Encoder": abordagem "One vs All"
    y = OneHotEncoder(sparse_output=False, dtype=np.int32).fit_transform(y.reshape(-1, 1))
    
    # Obtendo a quantidade de classes do conj. amostras
    n_class = y.shape[1]
   
    # Separando os subconjuntos de treinamento (70%), validação (15%) e teste (15%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    #!########################################################### 
    #! Ciclos de Execução!
    #!###########################################################  
    # Variáveis para salvar as tabelas gerais do ciclo
    cycle_best_perceptron = [] # Dados espec. do melhor Perceptron
    cycle_acc_test = []        # Taxa de Acerto do Conj. Teste
    cycle_error_test = []      # Taxa de Erro do Conj. Teste
    cycle_epoch_train = []     # Número de Épocas/Iterações do Treinamento
    cycle_exec_time = []       # Tempo de Execução

    for cycle in range(max_cycle):
        print(f"{'-'*75}")
        print(f"{'Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}")
        
        #!###########################################################
        #! Experimentos com Perceptron
        #!###########################################################
        # Geração do peso e bias aleatoriamente
        initial_W = np.random.uniform(size=(X.shape[1], n_class))
        initial_bias = np.random.uniform(size=n_class)

        # Variáveis para salvar o gráfico de convergência do melhor Perceptron executado
        best_perceptron = None
        best_acc_test = -1.0        # Melhor acurácia do conj. teste
        best_num_experiment = -1    # Número do melhor experimento
        
        # Variáveis para salvar os dados de cada experimento
        experiment_acc_test = []    # Taxa de Acerto do Conj. Teste
        experiment_error_test = []  # Taxa de Erro do Conj. Teste
        experiment_epoch_train = [] # Número de Épocas/Iterações do Treinamento
        experiment_exec_time = []   # Tempo de Execução
        
        for num_experiment in range(max_exp_per_cycle):
            # Registra o tempo inicial de execução
            start_timer = time()
            
            # Para cada experimento, aplicar o Perceptron e salvar os dados necessários
            perceptron = Perceptron(
                W=initial_W,
                bias=initial_bias,
                learning_rate=initial_learning_rate[cycle],
                n_class=n_class,
                max_epoch=max_epoch,
                max_patience=max_patience
            )
            
            # Realizando treinamento e validação
            perceptron.fit(
                X_train=X_train, 
                y_train=y_train, 
                X_val=X_val, 
                y_val=y_val,
            )
            
            # Testando e obtendo a acurácia
            test_pred = perceptron.predict(X=X_test)
            acc_test = accuracy_score(y_true=y_test, y_pred=test_pred)
            print(f"Acurácia no Conj. de Teste: {acc_test * 100 :.2f}%")
            
            # Registra o tempo total de execução do algoritmo
            total_time = time() - start_timer
            print(f"Tempo de Execução (s): {total_time :.4f}")
            
            # Salvando o melhor Perceptron para gráfico
            if acc_test > best_acc_test:
                best_perceptron = cp(perceptron)
                best_acc_test = cp(acc_test)
                best_num_experiment = cp(num_experiment)

            # Salvando os dados para as tabelas
            experiment_acc_test.append(acc_test)
            experiment_error_test.append(1 - acc_test)  
            experiment_epoch_train.append(perceptron.mse_train.shape[0])  
            experiment_exec_time.append(total_time)

            print(f"{'-'*50}")            

        # Após os experimentos, criar gráfico de conv. do melhor Perceptron
        plot_experiment(
            filename=filename, 
            alg_name_acronym='P', 
            num_cycle=cycle+1, 
            num_experiment=best_num_experiment,
            learning_rate=best_perceptron.learning_rate,
            mse_train=best_perceptron.mse_train, 
            mse_val=best_perceptron.mse_val
        )
        
        # Também, salvar dados relevantes da melhor rede executada
        y_train_pred = best_perceptron.predict(X=X_train)
        y_val_pred = best_perceptron.predict(X=X_val)
        y_test_pred = best_perceptron.predict(X=X_test)
        
        cycle_best_perceptron.append([
            initial_W,     # W inicial do ciclo
            initial_bias,  # Bias inicial do ciclo
            best_perceptron.W, # W final do ciclo
            best_perceptron.bias, # Bias final do ciclo
            best_perceptron.learning_rate, # Taxa de Learning Rate
            confusion_matrix(np.argmax(y_train, axis=1), np.argmax(y_train_pred, axis=1)), # Matriz de Confusão: Treinamento
            confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_val_pred, axis=1)),     # Matriz de Confusão: Validação
            confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1)),   # Matriz de Confusão: Teste
        ])
        
        # Salvando outros dados relevantes do ciclo
        cycle_acc_test.append([
            initial_W,     # W inicial do ciclo
            initial_bias,  # Bias inicial do ciclo
            best_perceptron.W,  # W final do ciclo
            best_perceptron.bias,  # Bias final do ciclo
            initial_learning_rate[cycle],   # Taxa de Learning Rate
            np.mean(experiment_acc_test),   # Média
            np.std(experiment_acc_test),    # Desvio Padrão
            np.min(experiment_acc_test),    # Mínimo
            np.median(experiment_acc_test), # Mediana
            np.max(experiment_acc_test),    # Máximo
        ])
        cycle_error_test.append([
            initial_W,     # W inicial do ciclo
            initial_bias,  # Bias inicial do ciclo
            best_perceptron.W, # W final do ciclo
            best_perceptron.bias, # Bias final do ciclo
            initial_learning_rate[cycle],     # Taxa de Learning Rate
            np.mean(experiment_error_test),   # Média
            np.std(experiment_error_test),    # Desvio Padrão
            np.min(experiment_error_test),    # Mínimo
            np.median(experiment_error_test), # Mediana
            np.max(experiment_error_test),    # Máximo
        ])
        cycle_epoch_train.append([
            initial_W,     # W inicial do ciclo
            initial_bias,  # Bias inicial do ciclo
            best_perceptron.W,  # W final do ciclo
            best_perceptron.bias,  # Bias final do ciclo
            initial_learning_rate[cycle],      # Taxa de Learning Rate
            np.mean(experiment_epoch_train),   # Média
            np.std(experiment_epoch_train),    # Desvio Padrão
            np.min(experiment_epoch_train),    # Mínimo
            np.median(experiment_epoch_train), # Mediana
            np.max(experiment_epoch_train),    # Máximo
        ])
        cycle_exec_time.append([
            initial_W,     # W inicial do ciclo
            initial_bias,  # Bias inicial do ciclo
            best_perceptron.W,  # W final do ciclo
            best_perceptron.bias,  # Bias final do ciclo
            initial_learning_rate[cycle],    # Taxa de Learning Rate
            np.mean(experiment_exec_time),   # Média
            np.std(experiment_exec_time),    # Desvio Padrão
            np.min(experiment_exec_time),    # Mínimo
            np.median(experiment_exec_time), # Mediana
            np.max(experiment_exec_time),    # Máximo
        ])

        print(f"{'-'*75}")
        print(f"{'Fim do Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}\n")
    
    # Salvar os dados para as tabelas
    
    # Melhor Perceptron
    create_txt(
        filename=filename,
        alg_name_acronym='P',
        type_exp='melhorP',
        rows=cycle_best_perceptron,
    )
    
    # Taxa de Acerto
    create_txt(
        filename=filename,
        alg_name_acronym='P',
        type_exp='acerto',
        rows=cycle_acc_test,
    )
    
    # Taxa de Erro
    create_txt(
        filename=filename,
        alg_name_acronym='P',
        type_exp='erro',
        rows=cycle_error_test,
    )
    
    # Número de Épocas
    create_txt(
        filename=filename,
        alg_name_acronym='P',
        type_exp='epocas',
        rows=cycle_epoch_train,
    )
    
    # Tempo de Execução (s)
    create_txt(
        filename=filename,
        alg_name_acronym='P',
        type_exp='tempo',
        rows=cycle_exec_time,
    )


# Criar gráfico de um dos experimentos
def plot_experiment(
    filename: str,
    alg_name_acronym: str,
    num_cycle: int,
    num_experiment: int,
    learning_rate: float,
    mse_train: np.ndarray,
    mse_val: np.ndarray,
):
    """Cria o gráfico de convergência do Perceptron,
    no formato "Época x MSE".
    
    Args:
        filename : str 
            Nome do arquivo/exercicio (ex: 'ex01')
        alg_name_acronym : str
            Nome ou sigla do algoritmo executado (ex: 'Percp' - Perceptron)
        num_cycle : int 
            Número do ciclo de execução
        num_experiment : int 
            Número do experimento dentro do ciclo
        mse_train : np.ndarray 
            Vetor que armazena os valores do erro médio quadrático (MSE) para o conjunto de treinamento do Perceptron.
        mse_val : np.ndarray 
            Vetor que armazena os valores do erro médio quadrático (MSE) para o conjunto de validação do Perceptron.
            
        
    Notes:
        Cria um arquivo de imagem do gráfico com o seguinte nome: {alg_name_acronym}_ciclo{num_cycle}_exp{num_experiment}_lr{learning_rate}.png
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
    plot_name = f'{alg_name_acronym}_ciclo{num_cycle}_exp{num_experiment}_lr{learning_rate}.png'

    # Definindo os textos (nomes) do gráfico
    plt.title('Convergência do Perceptron', loc='center')
    plt.xlabel('Época', loc='center')
    plt.ylabel('Erro Médio Quadrático (MSE)', loc='center')

    # Plotando o gráfico com base nos valores
    plt.plot(mse_train, label='Treino', c='b')
    plt.plot(mse_val, label='Validação', c='r')

    # Adiciona legenda
    plt.legend()

    # Plota e salva o gráfico em um arquivo
    plt.savefig(os.path.join(sub_directory, plot_name))

    # Encerra as configurações do gráfico
    plt.close()


# Criar arquivos texto para listas
def create_txt(
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
            Nome ou sigla do algoritmo executado (ex: 'Percp' - Perceptron)
        type_exp (str): tipo da tabela (ex: 'acerto')
            -> Utilize: 'melhorP'/'melhorMLP', 'acerto', 'erro', 'epoca', 'tempo'
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
    
    # Definindo números sempre aleatórios para NumPy durante a execução do código
    np.random.seed(42)
    
    # Definindo as condições gerais e comuns de todos exercícios
    max_cycle = 4
    max_exp_per_cycle = 25
    max_epoch = 1000
    max_patience = 10
    initial_learning_rate = np.array([1, 0.1, 0.01, 0.001])
    
    ########################################
    #!     Exercício 01: Iris Dataset     !#
    ########################################

    # Definindo as condições iniciais do exercício
    filename = 'ex01'
    X, y = load_iris_dataset()
    
    # Execução dos ciclos do exercício
    run_perceptron_cycle_experiments(
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
    
    # Definindo as condições iniciais do exercício
    filename = 'ex02'
    X, y = load_wine_dataset()
    
    # Execução dos ciclos do exercício
    run_perceptron_cycle_experiments(
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
    #!            Exercício 03            !#
    ########################################
    
    # Definindo as condições iniciais do exercício
    filename = 'ex03'
    X, y = load_btsc_dataset()
    
    ##################################
    #*          Perceptron          *#
    ##################################
    
    # Execução dos ciclos do exercício
    run_perceptron_cycle_experiments(
        filename=filename,
        max_cycle=max_cycle,
        max_exp_per_cycle=max_exp_per_cycle,
        X=X,
        y=y,
        initial_learning_rate=initial_learning_rate,
        max_epoch=max_epoch,
        max_patience=max_patience,
    )
    
    ##################################
    #*     MultiLayer Perceptron    *#
    ##################################
    
    # Importanto o MLP do SciKit-Learn
    from sklearn.neural_network import MLPClassifier

    # Biblioteca para capturar o tempo de execução
    from time import time

    # Normalizando os dados com Z Score
    X = StandardScaler().fit_transform(X)
    
    # Separando os subconjuntos de treinamento (70%), validação (15%) e teste (15%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    #!###########################################################
    #! Ciclos de Execução!
    #!###########################################################
    # Variáveis para salvar as tabelas gerais do ciclo
    cycle_best_mlp = []        # Dados espec. do melhor MLP
    cycle_acc_test = []        # Taxa de Acerto do Conj. Teste
    cycle_error_test = []      # Taxa de Erro do Conj. Teste
    cycle_epoch_train = []     # Número de Épocas/Iterações do Treinamento
    cycle_exec_time = []       # Tempo de Execução

    for cycle in range(max_cycle):
        print(f"{'-'*75}")
        print(f"{'Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}")

        #!###########################################################
        #! Experimentos com MLP
        #!###########################################################
        # Variáveis para salvar o gráfico de convergência do melhor MLP executado
        best_mlp = None
        best_acc_test = -1.0        # Melhor acurácia do conj. teste

        # Variáveis para salvar os dados de cada experimento
        experiment_acc_test = []    # Taxa de Acerto do Conj. Teste
        experiment_error_test = []  # Taxa de Erro do Conj. Teste
        experiment_epoch_train = []  # Número de Épocas/Iterações do Treinamento
        experiment_exec_time = []   # Tempo de Execução

        for num_experiment in range(max_exp_per_cycle):
            # Registra o tempo inicial de execução
            start_timer = time()

            print(f"\n{'-'*50}")
            print(f"{'MLP':^50}")
            print(f"{'-'*50}")
        
            # Treinando o MLP
            multi_perceptron = MLPClassifier(
                hidden_layer_sizes=(X_train.shape[1] * 2),
                activation='relu',
                solver='adam',
                learning_rate_init=initial_learning_rate[cycle],
                max_iter=max_epoch,
                random_state=42,
                n_iter_no_change=max_patience,
            )
            multi_perceptron.fit(X_train, y_train)

            # Testando e obtendo a acurácia
            test_pred = multi_perceptron.predict(X=X_test)
            acc_test = accuracy_score(y_true=y_test, y_pred=test_pred)
            print(f"Acurácia no Conj. de Teste: {acc_test * 100 :.2f}%")

            # Registra o tempo total de execução do algoritmo
            total_time = time() - start_timer
            print(f"Tempo de Execução (s): {total_time :.4f}")

            # Salvando o melhor Perceptron para gráfico
            if acc_test > best_acc_test:
                best_mlp = cp(multi_perceptron)
                best_acc_test = cp(acc_test)

            # Salvando os dados para as tabelas
            experiment_acc_test.append(acc_test)
            experiment_error_test.append(1 - acc_test)
            experiment_epoch_train.append(multi_perceptron.n_iter_)
            experiment_exec_time.append(total_time)

            print(f"{'-'*50}")

        # Também, salvar dados relevantes da melhor rede executada
        y_train_pred = best_mlp.predict(X=X_train)
        y_val_pred = best_mlp.predict(X=X_val)
        y_test_pred = best_mlp.predict(X=X_test)
        
        cycle_best_mlp.append([
            best_mlp.coefs_, # W final do ciclo
            best_mlp.intercepts_,  # Bias final do ciclo
            initial_learning_rate[cycle],  # Taxa de Learning Rate
            confusion_matrix(y_train, y_train_pred), # Matriz de Confusão: Treinamento
            confusion_matrix(y_val, y_val_pred),     # Matriz de Confusão: Validação
            confusion_matrix(y_test, y_test_pred),   # Matriz de Confusão: Teste
        ])
        
        # Salvando outros dados relevantes do ciclo
        cycle_acc_test.append([
            best_mlp.coefs_,  # W final do ciclo
            best_mlp.intercepts_,  # Bias final do ciclo
            initial_learning_rate[cycle],   # Taxa de Learning Rate
            np.mean(experiment_acc_test),   # Média
            np.std(experiment_acc_test),    # Desvio Padrão
            np.min(experiment_acc_test),    # Mínimo
            np.median(experiment_acc_test),  # Mediana
            np.max(experiment_acc_test),    # Máximo
        ])
        cycle_error_test.append([
            best_mlp.coefs_,  # W final do ciclo
            best_mlp.intercepts_,  # Bias final do ciclo
            initial_learning_rate[cycle],     # Taxa de Learning Rate
            np.mean(experiment_error_test),   # Média
            np.std(experiment_error_test),    # Desvio Padrão
            np.min(experiment_error_test),    # Mínimo
            np.median(experiment_error_test),  # Mediana
            np.max(experiment_error_test),    # Máximo
        ])
        cycle_epoch_train.append([
            best_mlp.coefs_, # W final do ciclo
            best_mlp.intercepts_, # Bias final do ciclo
            initial_learning_rate[cycle],      # Taxa de Learning Rate
            np.mean(experiment_epoch_train),   # Média
            np.std(experiment_epoch_train),    # Desvio Padrão
            np.min(experiment_epoch_train),    # Mínimo
            np.median(experiment_epoch_train),  # Mediana
            np.max(experiment_epoch_train),    # Máximo
        ])
        cycle_exec_time.append([
            best_mlp.coefs_,  # W final do ciclo
            best_mlp.intercepts_,  # Bias final do ciclo
            initial_learning_rate[cycle],    # Taxa de Learning Rate
            np.mean(experiment_exec_time),   # Média
            np.std(experiment_exec_time),    # Desvio Padrão
            np.min(experiment_exec_time),    # Mínimo
            np.median(experiment_exec_time),  # Mediana
            np.max(experiment_exec_time),    # Máximo
        ])

        print(f"{'-'*75}")
        print(f"{'Fim do Ciclo %d' % (cycle+1):^75}")
        print(f"{'-'*75}\n")

    # Salvar os dados para as tabelas

    # Melhor Perceptron
    create_txt(
        filename=filename,
        alg_name_acronym='MLP',
        type_exp='melhorMLP',
        rows=cycle_best_mlp,
    )

    # Taxa de Acerto
    create_txt(
        filename=filename,
        alg_name_acronym='MLP',
        type_exp='acerto',
        rows=cycle_acc_test,
    )

    # Taxa de Erro
    create_txt(
        filename=filename,
        alg_name_acronym='MLP',
        type_exp='erro',
        rows=cycle_error_test,
    )

    # Número de Épocas
    create_txt(
        filename=filename,
        alg_name_acronym='MLP',
        type_exp='epocas',
        rows=cycle_epoch_train,
    )

    # Tempo de Execução (s)
    create_txt(
        filename=filename,
        alg_name_acronym='MLP',
        type_exp='tempo',
        rows=cycle_exec_time,
    )
    
    

if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    # Chama a função principal
    main()
