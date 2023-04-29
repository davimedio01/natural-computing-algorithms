"""
Trabalho 2 - Redes Neurais
Data de Entrega: 11/05/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np               # Matrizes e Funções Matemáticas


#####################################################
#             Rede Neural: Perceptron               #
#####################################################
class Perceptron:
    """Perceptron Simples para Classificação de Padrões.

    Args:
        W : np.ndarray
            Vetor de pesos do Perceptron, que é atualizado durante a fase de treinamento (fit).
        bias : float
            Valor de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
        learning_rate : float
            Taxa de aprendizado para atualização dos pesos.
        max_it : int
            Número máximo de iterações como critério de parada.
    
    Attributes:
        W : np.ndarray
            Vetor de pesos do Perceptron, que é atualizado durante a fase de treinamento (fit).
        bias : float
            Valor de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
        learning_rate : float
            Taxa de aprendizado para atualização dos pesos.
        max_it : int
            Número máximo de iterações como critério de parada.
        activation_function : function
            Função de ativação utilizada pelo Perceptron. Padrão: função limiar de passo.
        mse_train : np.ndarray
            Vetor que armazena os valores do erro médio quadrático (MSE) para o conjunto de treinamento. 
            Cada elemento do vetor corresponde ao MSE calculado após uma iteração (ou época) do algoritmo.
            Para o Perceptron, quanto menor o valor do MSE, melhor é o ajuste da rede aos dados de treinamento.
        mse_val : np.ndarray
            Vetor que armazena os valores do erro médio quadrático (MSE) para o conjunto de validação. 
            Cada elemento do vetor corresponde ao MSE calculado após uma iteração (ou época) do algoritmo.
            Para o Perceptron, quanto menor o valor do MSE, melhor é o ajuste da rede aos dados de treinamento.
             
    Methods:
        fit (X: np.ndarray, d: np.ndarray)
            Realiza o treinamento do Perceptron, com base nos parâmetros de inicialização.
                X_train: matriz do subconjunto de treinamento
                y_train: vetor de saídas desejadas do subconjunto de treinamento
                X_val: matriz do subconjunto de validação
                y_val: vetor de saídas desejadas do subconjunto de validação
        
        predict (X: np.ndarray) 
            Recebe um ndarray e calcula o Perceptron equivalente, após a fase de treinamento 
                X: matriz com os valores de entrada
    """

    def __init__(self, W, bias, learning_rate, max_it) -> None:
        self.W = W
        self.bias = bias
        self.learning_rate = learning_rate
        self.max_it = max_it
        self.activation_function = self.__step_function__
        self.mse_train = None
        self.mse_val = None

    def __step_function__(self, x):
        return np.where(x >=0, 1, -1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        #! Inicialização de W (pesos) e bias ocorre antes da chamada da função. !#
        
        # Obtendo a quantidade de amostras e características do conj. treinamento
        n_samples_train, n_features_train = X_train.shape

        # Obtendo a quantidade de amostras e características do conj. validação
        n_samples_val, n_features_val = X_val.shape
        
        # Inicializando os vetores de MSE para treinamento e validação
        self.mse_train = []
        self.mse_val= []
        
        # Percorrendo as épocas/iterações
        t = 1 # Época atual
        E_train = 1  # Erro acumulado da época
        best_acc_val = -1 # Melhor acurácia obtida
        while t < self.max_it and E_train > 0:
            E_train = 0
            E_val = 0 # Erro acumulado da validação

            # Percorrendo o padrão de treinamento para a época/iteração 't'
            for i in range(n_samples_train):
                # Obtendo a saída da rede para X_train
                y_pred = self.activation_function(np.dot(X_train, self.W) + self.bias)

                # Determinando o erro para X_train[i]
                error_train = y_train[i] - y_pred[i]

                # Atualizando vetor de pesos
                self.W += self.learning_rate * error_train * X_train[i]

                # Atualizando o bias
                self.bias += self.learning_rate * error_train

                # Acumulando o erro quadrático para treinamento
                E_train += error_train ** 2

            # Salvando o erro médio quadrático do treinamento de uma época
            self.mse_train.append(E_train / n_samples_train)
            
            # Calculando o erro médio quadrático da validação
            for i in range(n_samples_val):
                # Obtendo a saída da rede para X_val
                y_val_pred = self.activation_function(np.dot(X_val, self.W) + self.bias)
                
                # Determinando o erro para X_val[i]
                error_val = y_val[i] - y_val_pred[i]
                
                # Acumulando o erro quadrático para treinamento
                E_val += error_val ** 2
                
            # Salvando o erro médio quadrático da validação de uma época
            self.mse_val.append(E_val / n_samples_val)

            # Modificando hiperparâmetros com base nos erros obtidos
            #!####################################################
            

            # Incrementando a época/iteração
            t += 1

        # Convertendo os erros salvos para NumPy
        self.mse_train = np.array(self.mse_train, dtype=np.float64)
        self.mse_val = np.array(self.mse_val, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Para predizer, basta aplicar os dados à função de ativação e o último bias obtido
        return self.activation_function(np.dot(X, self.W) + self.bias)
