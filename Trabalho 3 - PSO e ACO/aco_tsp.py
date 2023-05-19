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
        
    
    Attributes:
        best_path_distances : np.ndarray [float, ...]
            Lista as distâncias dos melhores caminhos encontrados por iteração. \n
        best_path_coords : np.ndarray [[float, float], ...]
            Lista contendo as coordenadas (x, y) do melhor caminho encontrados do TSP. \n
        exec_time : float
            Tempo de execução do algoritmo, em segundos. \n
             
    Methods:
        
    
    Notes:
        
    """

    def __init__(
        self,
        alpha=1,
        beta=5,
        rha=0.5,
        Q=100,
        e=0.1,
        b=5
    ) -> None:
        
        # Definição e inicialização dos atributos da classe
        self.best_path_distances = None
        self.best_path_coords = None
        
        
        # Tempo de execução
        self.exec_time = time()
    
      
    def optimize(
        self,
        dist_coords: np.ndarray,
        tau_init=1e-6,
        max_it=10000,
        max_patience=100,
    ):
        """desc.
        
        
        Args:
            
        """

        print(f"\n{'-'*50}")
        print(f"{'ACO':^50}")
        print(f"{'-'*50}")

        # Inicialização de variáveis básicas do ACO
        N = dist_coords.shape[0]
        tau = np.full((N, N), tau_init)
        
        
        # Atualizando o tempo de execução
        self.exec_time = time() - self.exec_time
        
        print(f"Iteração {self.itr}")
        print(f"Distância do Melhor Caminho Encontrado: {self.best_path_distances[-1]}")
        print(f"Tempo de Execução (s): {self.exec_time:.4f}")
        
        print(f"{'-'*50}")
        print(f"{'Fim do ACO':^50}")
        print(f"{'-'*50}")
