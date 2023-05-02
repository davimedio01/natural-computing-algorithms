"""
Trabalho 2 - Redes Neurais
Data de Entrega: 11/05/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np                               # Matrizes e Funções Matemáticas
from copy import copy as cp                      # Copiar objetos (não somente a referência)
from sklearn.metrics import accuracy_score       # Cálculo da Acurácia

#####################################################
#             Rede Neural: Perceptron               #
#####################################################
class Perceptron:
    """Perceptron Simples para Classificação de Padrões
    por meio da abordagem "Um vs Resto" (One vs Rest - OVR).
    Necessário aplicar OneHotEncoder para o conjunto de rótulos de um dataset.

    Args:
        W : np.ndarray (n_linhas: n_características, n_colunas: n_classes)
            Matriz de pesos do Perceptron para cada classe, que é atualizado durante a fase de treinamento (fit).
        bias : np.ndarray (n_linhas: n_classes)
            Vetor com valores de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
        learning_rate : float
            Taxa de aprendizado para atualização dos pesos.
        n_class : int
            Número de classes dos conjuntos de dados.
        max_epoch : int
            Número máximo de épocas/iterações como critério de parada.
        max_patience : int
            Número máximo de iterações em que não houve melhora (padrão: 100)
    
    Attributes:
        W : np.ndarray (n_linhas: n_características, n_colunas: n_classes)
            Matriz de pesos do Perceptron para cada classe, que é atualizado durante a fase de treinamento (fit).
        bias : np.ndarray (n_linhas: n_classes)
            Vetor com valores de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
        learning_rate : float
            Taxa de aprendizado para atualização dos pesos.
        n_class : int
            Número de classes dos conjuntos de dados.
        max_epoch : int
            Número máximo de épocas/iterações como critério de parada.
        max_patience : int
            Número máximo de iterações em que não houve melhora (padrão: 100)
        activation_function : function
            Função de ativação utilizada pelo Perceptron. Padrão: função limiar de passo.
        mse_train : np.ndarray 
            Vetor que armazena os valores do erro médio quadrático (MSE) para o conjunto de treinamento. 
            Cada elemento do vetor corresponde ao MSE calculado após uma iteração (ou época) do algoritmo.
            Para o Perceptron, quanto menor o valor do MSE, melhor é o ajuste da rede aos dados de treinamento.
        mse_val : np.ndarray 
            Vetor que armazena os valores do erro médio quadrático (MSE) para o conjunto de validação. 
            Cada elemento do vetor corresponde ao MSE calculado após uma iteração (ou época) do algoritmo.
            Para o Perceptron, quanto menor o valor do MSE, melhor é o ajuste da rede aos dados de validação.
        all_acc_val : np.ndarray
            Vetor que armazena todos os valores das taxas de acerto para o conjunto de validação.
        all_error_val : np.ndarray
            Vetor que armazena todos os valores das taxas de erro para o conjunto de validação.
        all_best_acc_val : np.ndarray
            Vetor que armazena todos os melhores valores das taxas de acerto para o conjunto de validação.
        all_best_error_val : np.ndarray
            Vetor que armazena todos os melhores valores das taxas de erro para o conjunto de validação.
             
    Methods:
        fit (X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray)
            Realiza o treinamento do Perceptron, com base nos parâmetros de inicialização.
                X_train: matriz do subconjunto de treinamento
                y_train: vetor de saídas desejadas do subconjunto de treinamento, no formato OneHotEncoder
                X_val: matriz do subconjunto de validação
                y_val: vetor de saídas desejadas do subconjunto de validação, no formato OneHotEncoder
        
        predict (X: np.ndarray) 
            Recebe um ndarray e calcula o Perceptron equivalente, após a fase de treinamento 
                X: matriz com os valores de entrada
            Return:
                y_pred: vetor com valores das classes preditadas, no formato OneHotEncoder
    
    Notes:
        - Atributos marcados com "*" foram removidos na versão final.
    """

    def __init__(
        self,
        W: np.ndarray,
        bias: np.ndarray,
        learning_rate: float,
        n_class: int,
        max_epoch: int,
        max_patience=100,
    ) -> None:
        print(f"\n{'-'*50}")
        print(f"{'Perceptron':^50}")
        print(f"{'-'*50}")

        # Definição e inicialização dos atributos da classe
        self.W = W
        self.bias = bias
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_epoch = max_epoch
        self.max_patience = max_patience
        self.activation_function = self.__step_function__
        self.mse_train = None
        self.mse_val = None
        self.all_acc_val = None
        self.all_error_val = None
        self.all_best_acc_val = None
        self.all_best_error_val = None

    def __step_function__(self, x: np.ndarray):
        return np.where(x > 0, 1, 0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        #! Inicialização de W (pesos) e bias deve ocorrer antes da chamada da função. !#
        print(f"{'Treinando a rede'}", end='')
        print(f"{'.'*34}")

        # Inicializando variáveis dos melhores valores
        best_W = np.copy(self.W)
        best_bias = np.copy(self.bias)
        best_acc_val = -1.0 # Melhor taxa de acerto do conj. validação
        best_mse_val = np.Inf # Melhor MSE do conj. validação

        # # Inicializando os vetores de MSE para treinamento e validação
        self.mse_train = np.array([])
        self.mse_val = np.array([])

        # Inicializando o vetor das taxas acerto/erro da validação
        self.all_acc_val = np.array([])
        self.all_error_val = np.array([])
        self.all_best_acc_val = np.array([])
        self.all_best_error_val = np.array([])
        
        # Percorrendo as épocas/iterações
        epoch = 1    # Época atual
        patience = 1  # Paciência atual

        while epoch < self.max_epoch and patience < self.max_patience:
            
            ########################################
            #!         Fase de Treinamento        !#
            ########################################
            
            # Erros acumulados do treinamento
            E_train = 0.0

            # Percorrendo o padrão de treinamento para a época/iteração 't'
            for i in range(X_train.shape[0]):
                # Obtendo a saída da rede para X_train
                y_train_pred = self.activation_function(np.dot(X_train[i], self.W) + self.bias)

                # Determinando o erro para X_train[i]
                error_train = y_train[i] - y_train_pred
                
                # Atualizando vetor de pesos
                self.W = self.W + (self.learning_rate * X_train[i].reshape((-1, 1)) * error_train).reshape(self.W.shape)

                # Atualizando o bias
                self.bias = self.bias + (self.learning_rate * error_train).reshape(self.bias.shape)

                # Acumulando o erro quadrático para treinamento
                E_train = E_train + np.sum(error_train ** 2)

            # Salvando o erro médio quadrático do treinamento de uma época
            self.mse_train = np.append(self.mse_train, E_train / X_train.shape[0])
            
            ########################################
            #!          Fase de Validação         !#
            ########################################

            # Erros acumulados da validação
            E_val = 0.0

            # Calculando o erro médio quadrático da validação
            for i in range(X_val.shape[0]):
                # Obtendo a saída da rede para X_val[i]
                y_val_pred = self.activation_function(np.dot(X_val[i], self.W) + self.bias)

                # Determinando o erro para X_val[i]
                error_val = y_val[i] - y_val_pred

                # Acumulando o erro quadrático para treinamento
                E_val = E_val + np.sum(error_val ** 2)

            # Salvando o erro médio quadrático da validação de uma época
            self.mse_val = np.append(self.mse_val, E_val / X_val.shape[0])

            # # Realizando a predição da validação
            y_val_pred = self.predict(X=X_val)
            
            # # Encontrando a acurácia atual (acerto e erro) para cada classe (One vs All)
            acc_val = accuracy_score(y_true=y_val, y_pred=y_val_pred)
            error_val = (1.0 - acc_val)

            # Modificando hiperparâmetros (pesos e bias) com base no MSE acumulativo
            if self.mse_val[-1] < best_mse_val:
                # Caso o erro tenha melhorado, salva os valores obtidos da rede
                best_W = np.copy(self.W)
                best_bias = np.copy(self.bias)
                best_acc_val = cp(acc_val)
                best_mse_val = cp(self.mse_val[-1])
                patience = 1
            else:
                # Caso o erro tenha piorado, aumenta a paciência (rede estagnada)
                patience += 1
            
            # Salvando os erros de acurácia (acerto/erro) para conj. validação
            self.all_acc_val = np.append(self.all_acc_val, acc_val)
            self.all_error_val = np.append(self.all_error_val, 1.0 - acc_val)
            self.all_best_acc_val = np.append(self.all_best_acc_val, best_acc_val)
            self.all_best_error_val = np.append(self.all_best_error_val, 1.0 - best_acc_val)

            # Incrementando a época/iteração
            epoch += 1
        
        print(f"{'Treinamento finalizado!'}")
        print(f"{'-'*50}")

        # Salvando os melhores valores obtidos
        self.W = np.copy(best_W)
        self.bias = np.copy(best_bias)

        print(f"Melhores Pesos:\n{self.W}")
        print(f"Melhores Bias: {self.bias}")
        print(f"Taxa de Aprendizado: {self.learning_rate}")
        print(f"Taxa Média de Acerto (Validação): {np.mean(self.all_acc_val).astype(float) * 100 :.2f}%")
        print(f"Taxa Média de Erro (Validação): {np.mean(self.all_error_val).astype(float) * 100 :.2f}%")
        print(f"{'-'*50}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Para cada dado X[i] do conj., vai obter a respectiva classe esperada, com base no peso e bias treinados
        return self.activation_function(np.dot(X, self.W) + self.bias)
