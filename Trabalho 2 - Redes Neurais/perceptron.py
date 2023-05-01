"""
Trabalho 2 - Redes Neurais
Data de Entrega: 11/05/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np                               # Matrizes e Funções Matemáticas
from sklearn.metrics import accuracy_score       # Cálculo da Acurácia
from sklearn.preprocessing import OneHotEncoder  # Abordagem: "Um vs Todos" (OVR)
#from scipy.special import softmax                # Função de Ativação: Softmax

#####################################################
#             Rede Neural: Perceptron               #
#####################################################
class Perceptron:
    """Perceptron Simples para Classificação de Padrões
    por meio da abordagem "Um vs Resto" (One vs Rest - OVR).

    Args:
        W : np.ndarray (n_linhas: n_classes, n_colunas: n_características)
            Matriz de pesos do Perceptron para cada classe, que é atualizado durante a fase de treinamento (fit).
        bias : np.ndarray (n_linhas: 1, n_colunas: n_classes)
            Vetor de colunas com valor de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
        learning_rate : float
            Taxa de aprendizado para atualização dos pesos.
        n_class : int
            Número de classes dos conjuntos de dados.
        max_epoch : int
            Número máximo de épocas/iterações como critério de parada.
        max_patience : int
            Número máximo de iterações em que não houve melhora (padrão: 100)
    
    Attributes:
        W : np.ndarray (n_linhas: n_classes, n_colunas: n_características)
            Matriz de pesos do Perceptron para cada classe, que é atualizado durante a fase de treinamento (fit).
        bias : np.ndarray (n_linhas: 1, n_colunas: n_classes)
            Vetor de colunas com valor de bias do Perceptron, que é atualizado durante a fase de treinamento (fit).
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
        max_patience=100,
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
        self.activation_function = self.__step_function__
        self.mse_train = None
        self.mse_val = None
        self.all_acc_val = None
        self.all_error_val = None

    def __step_function__(self, x: np.ndarray):
        return np.where(x > 0, 1, 0)
    
    def __softmax__(self, x: np.ndarray):
        return np.exp(x)/sum(np.exp(x))
        # return softmax(x)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        #! Inicialização de W (pesos) e bias deve ocorrer antes da chamada da função. !#
        print(f"{'Treinando a rede'}", end='')
        print(f"{'.'*35}")
        
        # Aplicando "One Hot Encoder": abordagem "One vs All"
        enc = OneHotEncoder(sparse_output=False) #reshape(-1, 1)
        y_train = enc.fit_transform(y_train.reshape(-1, 1)).astype(int)
        y_val = enc.fit_transform(y_val.reshape(-1, 1)).astype(int)
        #print(y_train, ' ', y_val)

        # Inicializando variáveis dos melhores valores
        best_W = np.copy(self.W)
        best_bias = np.copy(self.bias)
        best_acc_val = np.array([-1.0] * y_val.shape[0])#.reshape((1, -1))  # Melhor taxa de acerto do conj. validação

        # # Inicializando os vetores de MSE para treinamento e validação
        self.mse_train = np.array([])
        self.mse_val = np.array([])

        # Inicializando o vetor das taxas acerto/erro da validação
        self.all_acc_val = np.array([])
        self.all_error_val = np.array([])
        print(self.all_acc_val, ' ', self.all_error_val)
        
        # Percorrendo as épocas/iterações
        epoch = 1    # Época atual
        patience = 1  # Paciência atual

        while epoch < self.max_epoch and patience < self.max_patience:
            ########################################
            #!         Fase de Treinamento        !#
            ########################################
            # Erros acumulados do treinamento
            E_train = np.zeros(self.n_class)

            # Percorrendo o padrão de treinamento para a época/iteração 't'
            for i in range(X_train.shape[0]):
                # Obtendo a saída da rede para X_train
                y_train_pred = self.activation_function(np.dot(self.W, X_train[i].T) + self.bias)

                # Determinando o erro para X_train[i]
                error_train = y_train[i] - y_train_pred
                #print(error_train)
                # Atualizando vetor de pesos
                self.W = self.W + (self.learning_rate * X_train[i].reshape((-1, 1)) * error_train).reshape(self.W.shape)

                # Atualizando o bias
                self.bias = self.bias + (self.learning_rate * error_train).reshape(self.bias.shape)

                # Acumulando o erro quadrático para treinamento
                E_train += np.sum(error_train ** 2)

            # Salvando o erro médio quadrático do treinamento de uma época
            #print(E_train, ' ', E_train.shape)
            #self.mse_train.append(np.mean(E_train))
            self.mse_train = np.append(self.mse_train, np.mean(E_train))
            print(self.mse_train)

            ########################################
            #!          Fase de Validação         !#
            ########################################

            # Erros acumulados da validação
            E_val = np.zeros(self.n_class)

            # Calculando o erro médio quadrático da validação
            for i in range(X_val.shape[0]):
                # Obtendo a saída da rede para X_val[i]
                y_val_pred = self.activation_function(np.dot(self.W, X_val[i].T) + self.bias)

                # Determinando o erro para X_val[i]
                error_val = y_val[i] - y_val_pred

                # Acumulando o erro quadrático para treinamento
                E_val += np.sum(error_val ** 2)

            # Salvando o erro médio quadrático da validação de uma época
            #self.mse_val.append(np.mean(E_val))
            self.mse_val = np.append(self.mse_val, np.mean(E_val))
            
            # Realizando a predição da validação
            y_val_pred = self.predict(X=X_val)

            #! ERRO PRINCIPAL ESTÁ AQUI
            # Encontrando a acurácia atual (acerto e erro) de cada classe
            acc_val = []
            error_val = []
            for i in range(y_val.shape[0]):
                print(y_val, ' ', y_val_pred)
                acc_val.append(accuracy_score(y_true=y_val[i], y_pred=y_val_pred[i]))
                print('acc_val: ', acc_val)

                # Modificando hiperparâmetros (pesos e bias) com base no MSE acumulativo
                if acc_val[i] > best_acc_val[i]:
                    # Caso o erro tenha melhorado, salva os valores obtidos da rede
                    best_W[i] = np.copy(self.W[i])
                    best_bias[0][i] = np.copy(self.bias[0][i])
                    best_acc_val[i] = np.copy(acc_val[i])
                    # self.mse_train.append(E_train)
                    # self.mse_val.append(np.mean(E_val))
                    patience = 1
                else:
                    # Caso o erro tenha piorado, aumenta a paciência (rede estagnada)
                    # self.mse_train.append(self.mse_train[-1])
                    # self.mse_val.append(self.mse_val[-1])
                    patience += 1
            
            # Salvando os erros de acurácia (acerto/erro) para conj. validação
            self.all_acc_val = np.append(self.all_acc_val, best_acc_val)
            self.all_error_val = np.append(self.all_error_val, 1 - best_acc_val)

            # Incrementando a época/iteração
            epoch += 1

        # Convertendo os erros salvos para NumPy
        #self.mse_train = np.array(self.mse_train)
        #self.mse_val = np.array(self.mse_val)
        #self.all_acc_val = np.array(self.all_acc_val)
        #self.all_error_val = np.array(self.all_error_val)

        # Convertendo os formatos de saída necessários
        self.all_acc_val = self.all_acc_val.reshape((-1, 1))
        self.all_error_val = self.all_error_val.reshape((-1, 1))
        
        print(f"{'Treinamento finalizado!'}")
        print(f"{'-'*25}")

        # Salvando os melhores valores obtidos
        self.W = np.copy(best_W)
        self.bias = np.copy(best_bias)

        print(f"Melhores Pesos:\n{self.W}")
        print(f"Melhores Bias: {self.bias}")
        # print(f"Melhor Saída: {0}")
        print(f"{'-'*35}\n")

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred_all = []

        # Para cada classe do problema, vai obter um conjunto de predições (One vs All para cada classe)
        for i in range(self.n_class):
            y_pred = np.zeros(X.shape[0])
            
            # Para predizer, basta aplicar os dados à função de ativação e o último bias obtido
            for j in range(X.shape[0]):
                # Calcula os dados com a função de ativação e pesos e bias de treinamento
                y = self.activation_function(np.dot(self.W, X[j].T) + self.bias)

                # Obtém a classe associada
                y_pred[j] = np.argmax(y)
            
            # Salva os valores do "One vs All" preditos
            #print(y_pred_all, ' ', y_pred)
            y_pred_all.append(y_pred)
        
        # Convertendo para NumPy
        y_pred_all = np.array(y_pred_all)
        return OneHotEncoder(sparse_output=False).fit_transform(y_pred_all.reshape((-1, 1))).astype(int)

