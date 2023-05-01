"""
Trabalho 2 - Redes Neurais
Data de Entrega: 11/05/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np               # Matrizes e Funções Matemáticas
from sklearn.metrics import accuracy_score # Cálculo da Acurácia
from scipy.special import softmax   # Função de Ativação: Softmax

#####################################################
#             Rede Neural: Perceptron               #
#####################################################
class Perceptron:
    """Perceptron Simples para Classificação de Padrões.

    Args:
        W : np.ndarray (n_linhas: n_características, n_colunas: n_classes)
            Matriz de pesos do Perceptron para cada classe, que é atualizado durante a fase de treinamento (fit).
        bias : np.ndarray (n_colunas: n_classes)
            Vetor com valor de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
        learning_rate : float
            Taxa de aprendizado para atualização dos pesos.
        n_class : int
            Número de classes dos conjuntos de dados.
        max_epoch : int
            Número máximo de épocas/iterações como critério de parada.
        max_patience : int
            Número máximo de iterações em que não houve melhora (padrão: 10)
    
    Attributes:
        W : np.ndarray (n_linhas: n_características, n_colunas: n_classes)
            Matriz de pesos do Perceptron para cada classe, que é atualizado durante a fase de treinamento (fit).
        bias : np.ndarray (n_colunas: n_classes)
            Vetor com valor de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
        learning_rate : float
            Taxa de aprendizado para atualização dos pesos.
        n_class : int
            Número de classes dos conjuntos de dados.
        max_epoch : int
            Número máximo de épocas/iterações como critério de parada.
        max_patience : int
            Número máximo de iterações em que não houve melhora (padrão: 10)
        activation_function : function
            Função de ativação utilizada pelo Perceptron. Padrão: função limiar de passo.
        * mse_train : np.ndarray *
            Vetor que armazena os valores do erro médio quadrático (MSE) para o conjunto de treinamento. 
            Cada elemento do vetor corresponde ao MSE calculado após uma iteração (ou época) do algoritmo.
            Para o Perceptron, quanto menor o valor do MSE, melhor é o ajuste da rede aos dados de treinamento.
        all_acc_val : np.ndarray
            Vetor que armazena os valores das taxas de acerto para o conjunto de validação.
        all_error_val : np.ndarray
            Vetor que armazena os valores das taxas de erro para o conjunto de validação.
             
    Methods:
        fit (X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray)
            Realiza o treinamento do Perceptron, com base nos parâmetros de inicialização.
                X_train: matriz do subconjunto de treinamento
                y_train: vetor de saídas desejadas do subconjunto de treinamento
                X_val: matriz do subconjunto de validação
                y_val: vetor de saídas desejadas do subconjunto de validação
        
        predict (X: np.ndarray) 
            Recebe um ndarray e calcula o Perceptron equivalente, após a fase de treinamento 
                X: matriz com os valores de entrada
            Return:
                y_pred: vetor com valores das classes preditadas
    
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
        max_patience: int
    ) -> None:
        print(f"{'-'*50}")
        print(f"{'Perceptron':^50}")
        print(f"{'-'*50}")
    
        # Definição e inicialização dos atributos da classe
        self.W = W
        self.bias = bias
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_epoch = max_epoch
        self.max_patience = max_patience
        self.activation_function = self.__softmax__
        #self.mse_train = None
        self.all_acc_val = None
        self.all_error_val = None

    def __softmax__(self, x: np.ndarray):
        #return np.exp(x)/sum(np.exp(x))
        return softmax(x=x)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        #! Inicialização de W (pesos) e bias deve ocorrer antes da chamada da função. !#
        print(f"{'Treinando a rede'}", end='')
        print(f"{'.'*25}")
        
        # Inicializando variáveis dos melhores valores
        best_W = np.copy(self.W)
        best_bias = np.copy(self.bias)
        best_acc_val= -1.0  # Melhor taxa de acerto do conj. validação
        
        # # Inicializando os vetores de MSE para treinamento 
        # self.mse_train = []
        
        # Inicializando o vetor das taxas acerto/erro da validação 
        self.all_acc_val = []
        self.all_error_val = []
        
        # Percorrendo as épocas/iterações
        epoch = 1    # Época atual
        patience = 1 # Paciência atual
        
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
                error_train = y_train[i] - np.argmax(y_train_pred) 
                    # ((y_train[i] - y_train_pred) ** 2) / y_train_pred.shape[0] 
                    # # mean_squared_error(y_train[i], y_train_pred)

                # Atualizando vetor de pesos
                self.W += self.learning_rate * X_train[i].reshape(-1, 1) * error_train

                # Atualizando o bias
                self.bias += self.learning_rate * error_train

                # Acumulando o erro quadrático para treinamento
                E_train += np.sum(error_train)

            print(E_train)
            # # Salvando o erro médio quadrático do treinamento de uma época
            # self.mse_train.append(E_train)
            # #print(np.array(self.mse_train))
            
            ########################################
            #!          Fase de Validação         !#
            ########################################
            
            # Realizando a predição da validação
            y_val_pred = self.predict(X=X_val)
            
            # Encontrando a acurácia atual (acerto e erro)
            acc_val = accuracy_score(y_true=y_val, y_pred=y_val_pred)
            error_val = 1 - acc_val
            
            # # Erros acumulados da validação
            # E_val = 0.0
            
            # # Calculando o erro médio quadrático da validação
            # for i in range(X_val.shape[0]):
            #     # Obtendo a saída da rede para X_val[i]
            #     y_val_pred = self.activation_function(X_val[i].dot(self.W) + self.bias)
                
            #     # Determinando o erro para X_val[i]
            #     error_val = ((y_val[i] - y_val_pred) ** 2) / y_val_pred.shape[0]
                
            #     # Acumulando o erro quadrático para treinamento
            #     E_val += np.sum(error_val)
  
            # # Salvando o erro médio quadrático da validação de uma época
            # #self.mse_val.append(E_val / n_samples_val)

            # # Modificando hiperparâmetros (pesos e bias) com base no MSE acumulativo
            # cumulative_mse_val = E_val
            
            if acc_val > best_acc_val:
                # Caso o erro tenha melhorado, salva os valores obtidos da rede
                best_W = np.copy(self.W)
                best_bias = np.copy(self.bias)
                best_acc_val = acc_val
                self.all_acc_val.append(acc_val)
                self.all_error_val.append(error_val)
                patience = 1
            else:
                # Caso o erro tenha piorado, aumenta a paciência (rede estagnada)
                self.all_acc_val.append(self.all_acc_val[-1])
                self.all_error_val.append(self.all_error_val[-1])
                patience += 1

            # Incrementando a época/iteração
            epoch += 1

        # Convertendo os erros salvos para NumPy
        #self.mse_train = np.array(self.mse_train)
        self.all_acc_val = np.array(self.all_acc_val)
        self.all_error_val = np.array(self.all_error_val)
        
        print(f"{'Treinamento finalizado!'}")
        print(f"{'-'*25}")
        
        # Salvando os melhores valores obtidos
        self.W = np.copy(best_W)
        self.bias = np.copy(best_bias)
        
        print(f"Melhores Pesos:\n{self.W}")
        print(f"Melhores Bias: {self.bias}")
        #print(f"Melhor Saída: {0}")
        print(f"{'-'*25}\n")

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = []
        
        # Para predizer, basta aplicar os dados à função de ativação e o último bias obtido
        for i in range(X.shape[0]):
            # Calcula os dados com a função de ativação e pesos e bias de treinamento
            y = self.activation_function(np.dot(X[i], self.W) + self.bias)
            
            # Obtém a classe associada
            y_pred.append(np.argmax(y))
        
        return np.array(y_pred)
