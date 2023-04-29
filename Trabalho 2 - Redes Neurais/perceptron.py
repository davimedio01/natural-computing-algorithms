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
        W : np.ndarray (tam: X.shape[1] ou num_caracteristicas)
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
            Função de ativação utilizada pelo Perceptron. Padrão: função degrau unitário (step function).
        mse : np.ndarray
            Vetor que armazena os valores do erro médio quadrático (MSE) durante a fase de treinamento. 
            Cada elemento do vetor corresponde ao MSE calculado após uma iteração (ou época) do algoritmo.
            Para o Perceptron, quanto menor o valor do MSE, melhor é o ajuste da rede aos dados de treinamento.
            
    Methods:
        fit (X: np.ndarray, d: np.ndarray)
            Realiza o treinamento do Perceptron, com base nos parâmetros de inicialização.
                X: matriz com os valores de entrada
                d: vetor de saídas desejadas (para erro)
        
        predict (X: np.ndarray) 
            Recebe um ndarray e calcula o Perceptron equivalente, após a fase de treinamento 
                X: matriz com os valores de entrada
    """

    def __init__(self, W, bias, learning_rate, max_it) -> None:
        self.W = W
        self.bias = bias
        self.learning_rate = learning_rate
        self.max_it = max_it
        self.activation_function = self.__unit_step_func__
        self.mse = None

    def __unit_step_func__(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X: np.ndarray, d: np.ndarray):
        # Obtendo a quantidade de amostras e características
        n_samples, n_features = X.shape

        #! Inicializar W (pesos) e bias separadamente

        # Percorrendo as épocas/iterações
        t = 1
        E = 1
        self.mse = []
        while t < self.max_it and E > 0:
            E = 0

            # Percorrendo o padrão de treinamento para a época/iteração 't'
            for i in range(n_samples):
                # Obtendo a saída da rede para X[i]
                y = self.activation_function(np.dot(X[i], self.W) + self.bias)

                # Determinando o erro para X[i]
                error = d[i] - y

                # Atualizando vetor de pesos
                self.W += self.learning_rate * error * X[i]

                # Atualizando o bias
                self.bias += self.learning_rate * error

                # Acumulando o erro quadrático
                E += error ** 2

            # Salvando o erro médio quadrático de uma época
            self.mse.append(E / n_samples)

            # Incrementando a iteração
            t += 1

        # Convertendo os erros salvos para NumPy
        self.mse = np.array(self.mse, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Para predizer, basta aplicar os dados à função de ativação e o último bias obtido
        return self.activation_function(np.dot(X, self.W) + self.bias)
