"""
Trabalho 3 - PSO e ACO
Data de Entrega: 22/06/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np          # Matrizes e Funções Matemáticas
from copy import copy as cp # Copiar objetos (não somente a referência)
from time import time       # Biblioteca para capturar o tempo de execução


#####################################################
#           Ant Colony Optimization (ACO)           #
#####################################################
class ACO_TSP:
    """Otimização por Colônia de Formigas (ACO) para otimizar o
    Problema do Caixeiro Viajante (TSP).

    Args:
        alpha : int
            Peso da trilha de feromônio 'tau' (padrão: 1) \n
        beta : int
            Peso do desejo heurístico 'eta' (padrão: 5) \n
        alpha : float
            Taxa de evaporação do feromônio (padrão: 0.5) \n
        Q : int
            Quantidade de feromônio depositado por uma formiga (padrão: 100) \n
        elite_ant : int
            Número de formigas elitistas (padrão: 100) \n
        
    Attributes:
        alpha : int
            Peso da trilha de feromônio 'tau' (padrão: 1) \n
        beta : int
            Peso do desejo heurístico 'eta' (padrão: 5) \n
        alpha : float
            Taxa de evaporação do feromônio (padrão: 0.5) \n
        Q : int
            Quantidade de feromônio depositado por uma formiga (padrão: 100) \n
        elite_ant : int
            Número de formigas elitistas (padrão: 100) \n
        itr : int
            Número da iteração final, após otimização \n
        best_path_distance : np.ndarray [float, ...]
            Lista as distâncias dos melhores caminhos encontrados por iteração. \n
        best_path_nodes : np.ndarray [[float, float], ...]
            Lista contendo os nós do melhor caminho encontrados do TSP. \n
        exec_time : float
            Tempo de execução do algoritmo, em segundos. \n
             
    Methods:
        optimize (node_coords: np.ndarray, tau_init: float, max_it: int, max_patience: int)
            Aplicação do ACO para TSP, a partir dos parâmetros descritos abaixo 
                node_coords: Lista contendo o conjunto de coordenadas das cidades do TSP.
                tau_init: Número inicial da trilha de feromônio 'tau' (padrão: 10^-6)
                max_it: número máximo de iterações possíveis (padrão: 1000)
                max_patience: número máximo de iterações em que não houve melhora (padrão: 50)
                
    Notes:
        
    """

    def __init__(
        self,
        alpha=1,
        beta=5,
        rho=0.5,
        Q=100,
        elite_ant=5,
    ) -> None:
        
        # Definição e inicialização dos atributos da classe
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.elite_ant = elite_ant
        self.itr = 0
        
        # Melhores valores encontrados: distância por iteração e coordenadas do melhor
        self.best_path_distance = []
        self.best_path_nodes = []
        
        # Tempo de execução
        self.exec_time = time()


    def __euclidian_distance__(self, point1: np.ndarray, point2: np.ndarray,):
        return np.sqrt(np.sum((point1 - point2)**2))


    def optimize(
        self,
        node_coords: np.ndarray,
        tau_init=1e-6,
        max_it=1000,
        max_patience=50,
    ):
        """Aplicação do ACO para a busca do caminho mínimo de um problema do tipo TSP. \n
        
        Args:
            node_coords : np.ndarray[[float, float], ...]
                Lista contendo o conjunto de coordenadas das cidades do TSP com (linhas: num_cidades, colunas: 2).
                Verifique os experimentos que mostram como realizar a conversão do objeto TSP necessário da biblioteca 'tsplib95' para NumPy. \n
            tau_init : float
                Número inicial da trilha de feromônio 'tau' (padrão: 10^-6) \n
            max_it : int
                Número máximo de iterações possíveis (padrão: 1000) \n
            max_patience : int
                Número máximo de iterações em que não houve melhora (padrão: 50) \n
        """

        print(f"\n{'-'*50}")
        print(f"{'ACO-TSP':^50}")
        print(f"{'-'*50}")
        
        # Definindo o número de formigas da colônia (número de cidades)
        num_ant = node_coords.shape[0]
        
        # Encontrando a matriz de distâncias a partir das coordenadas
        dist_cities = np.zeros((num_ant, num_ant), dtype=np.float32)
        for index, city in enumerate(node_coords):
            dist_cities[index] = np.sqrt(((node_coords - city) ** 2).sum(axis=1))

        # Desejo Heurístico: inverso das distâncias
        with np.errstate(all='ignore'):
            eta = (1 / dist_cities)
        eta[eta == np.inf] = 0
        
        # Feromônios
        tau = np.full((num_ant, num_ant), tau_init)      
        
        # Auxiliares para salvar os melhores valores da distância
        best_path_distance = np.inf
        
        # Percorrendo as iterações
        self.itr = 0
        patience = 1
        ant_city = np.random.choice(num_ant, size=num_ant, replace=False)
        
        while (self.itr < max_it and patience < max_patience):
            # Aumenta o número da iteração atual
            self.itr += 1
            
            # Para cada formiga, completar o caminho
            ant_paths = []
            ant_paths_distance = []

            #for ant in range(num_ant):
            for ant in ant_city:
                # Realizando configuração de cidades visitadas e atual
                visited_city = np.array([False] * num_ant)
                current_city = cp(ant)
                
                # Marcando a cidade atual como visitada
                visited_city[current_city] = True
                
                # Iniciando as variáveis relativas ao caminho
                path = [current_city]
                path_distance = 0.0
                prev_city = cp(current_city)
                
                # Percorrendo o grafo com a formiga atual
                while np.any(np.logical_not(visited_city)):
                    # Marcando as cidades não visitadas
                    unvisited_city = np.where(np.logical_not(visited_city))[0]

                    # Calculando a probabilidade da próxima cidade
                    prob_next_city = np.zeros(unvisited_city.shape[0], dtype=np.float32)
                    for index, next_city in enumerate(unvisited_city):
                        prob_next_city[index] = (tau[current_city, next_city] ** self.alpha) * (eta[current_city, next_city] ** self.beta)
                    if np.sum(prob_next_city) == 0:
                        prob_next_city = np.ones_like(prob_next_city) / prob_next_city.shape[0]
                    else:
                        prob_next_city /= np.sum(prob_next_city)
                    
                    # Obtendo a próxima cidade a ser visitada com base na maior probabilidade
                    next_city = np.random.choice(unvisited_city, p=prob_next_city)
                    
                    # Atualizando o grafo com a próxima cidade a ser visitada
                    path.append(next_city)
                    path_distance += dist_cities[current_city, next_city]
                    visited_city[next_city] = True
                    prev_city = cp(current_city)
                    current_city = cp(next_city)    
                
                # Obtendo a distância relativa ao último nó para o primeiro
                path_distance += dist_cities[prev_city, current_city]
                
                # Avaliando a rota construída por cada formiga
                if path_distance < best_path_distance:
                    best_path_distance = cp(path_distance)
                    self.best_path_nodes = np.copy(path)
                    patience = 1
                elif path_distance == best_path_distance:
                    # Aumentar a estagnação
                    patience += 1
                
                # Salvando os dados obtidos da formiga
                ant_paths.append(path)
                ant_paths_distance.append(path_distance)
            
            # Salvando os melhores valores das distâncias por iteração
            self.best_path_distance = np.append(self.best_path_distance, best_path_distance)

            # Obtendo a evaporação da trilha de feromônio
            delta_tau = np.zeros_like(tau)
            for path, path_distance in zip(ant_paths, ant_paths_distance):
                delta_tau[path[:-1], path[1:]] += self.Q / path_distance
                delta_tau[path[-1] , path[0]]  += self.Q / path_distance
            
            # Aplicação do elitismo
            elitism = self.elite_ant * (self.Q / np.min(ant_paths_distance))
            
            # Atualizando as trilhas de feromônio
            tau = (1 - self.rho) * tau + delta_tau + elitism
        
        # Atualizando o tempo de execução
        self.exec_time = time() - self.exec_time
        
        print(f"Iteração {self.itr}")
        print(f"Distância do Melhor Caminho Encontrado: {self.best_path_distance[-1]}")
        print(f"Tempo de Execução (s): {self.exec_time:.4f}")
        
        print(f"{'-'*50}")
        print(f"{'Fim do ACO':^50}")
        print(f"{'-'*50}")
