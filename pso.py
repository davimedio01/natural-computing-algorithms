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
#        Particle Swarm Optimization (PSO)          #
#####################################################
class PSO:
    """Otimização por Enxame de Partículas (PSO) para minimização de 
    funções contínuas de uma ou mais variáveis.
    

    Args:
        VMIN : float
            Constante para limite velocidade mínima das partículas (recomendado: entre [1, 0.1N] e VMIN < VMAX) \n
        VMAX : float
            Constante para limite velocidade máxima das partículas (recomendado: entre [1, 0.1N] e VMIN < VMAX) \n
        W : float
            Constante da inércia da velocidade (padrão: 0.7) \n
        AC1 : float
            Constante de aceleração da partícula - Aceleração do Comportamento Cognitivo (padrão: 2.05) \n
        AC2 : float
            Constante de aceleração dos vizinhos da partícula - Aceleração do Comportamento Social (padrão: 2.05) \n
    
    Attributes:
        VMIN : float
            Constante para limite velocidade mínima das partículas (recomendado: entre [1, 0.1N] e VMIN < VMAX) \n
        VMAX : float
            Constante para limite velocidade máxima das partículas (recomendado: entre [1, 0.1N] e VMIN < VMAX) \n
        W : float
            Constante da inércia da velocidade (padrão: 0.7) \n
        AC1 : float
            Constante de aceleração da partícula - Aceleração do Comportamento Cognitivo \n
        AC2 : float
            Constante de aceleração dos vizinhos da partícula - Aceleração do Comportamento Social \n
        best_global_particle : np.ndarray [float, ...]
            Lista contendo a melhor partícula global encontrada durante as iterações \n
        best_global_fitness : np.ndarray [float, ...]
            Lista contendo os valores de aptidão da melhor partícula global encontrada durante as iterações \n
        best_local_particle : np.ndarray [float, ...]
            Lista contendo a melhor partícula local encontrada durante as iterações \n
        best_local_fitness : np.ndarray [float, ...]
            Lista contendo os valores de aptidão das melhores partículas locais encontradas durante as iterações \n
        all_mean_fitness : np.ndarray [float, ...]
            Lista contendo a média das aptidões locais durante as gerações \n
        exec_time : float
            Tempo de execução do algoritmo, em segundos. \n
           
    Methods:
        optimize (fitness_func: callable, is_min: bool, max_it: int, max_patience: int)
            Aplicação do PSO, a partir dos parâmetros descritos abaixo 
                fitness_func: função de avaliação de aptidão. Ver exemplo de uso na '__rosenbrock_func__' (padrão: __rosenbrock_func__)
                is_min: booleana para atribuir algoritmo para min/max (padrão: True)
                max_it: número máximo de iterações possíveis (padrão: 10000)
                max_patience: número máximo de iterações em que não houve melhora (padrão: 100)
                
    Notes:
        
    """

    def __init__(
        self,
        VMIN: float,
        VMAX: float,
        W: float = 0.7,
        AC1: float = 2.05,
        AC2: float = 2.05,
    ) -> None:
        # Definição e inicialização dos atributos da classe
        self.VMIN = VMIN
        self.VMAX = VMAX
        self.W = W
        self.AC1 = AC1
        self.AC2 = AC2
        self.itr = 1
        
        # Somente o melhor global
        self.best_global_particle = None
        self.best_global_fitness = None
        
        # Somente o melhor local (da iteração)  
        self.best_local_particle = None
        self.best_local_fitness = None
        
        # Média de todos os locais (por iteração)
        self.best_mean_fitness = None 
        
        # Tempo de execução
        self.exec_time = time()
        
    def optimize(
        self,
        fitness_func: callable = None,
        is_min=True,
        bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        num_particles: int = np.random.randint(10, 51),
        max_it=10000,
        max_patience=100,
    ):
        """Aplicação do PSO para a otimização de uma função mutlivariável.
        Número de partículas (N) definido como inteiro aleatório entre [10,50] por padrão.
        Número de dimensões (num_dimensions) é definido com base na quantidade de variáveis da função de aptidão, por meio do parâmetro "bounds".
        
        Args:
            fitness_func : callable
                Função de avaliação de aptidão (ver exemplo de uso na '__rosenbrock_func__ (x: list[float])') \n
            is_min : bool
                Booleana para atribuir algoritmo para otimização min/max (padrão: True)
                    -- Se 'True', irá verificar minimização \n
                    -- Se 'False', irá verificar maximização \n
            bounds : np.ndarray[[float, float], ...]
                Lista contendo os intervalos min/max das variáveis da função (padrão: [[-1.0, 1.0], [-1.0, 1.0]]) \n
            num_particles : int
                Número de partículas do PSO (padrão: inteiro aleatório entre [10,50]) \n
            max_it : int
                Número máximo de iterações possíveis (padrão: 10000) \n
            max_patience : int
                Número máximo de iterações em que não houve melhora (padrão: 100) \n
            
        """
        
        print(f"\n{'-'*50}")
        print(f"{'PSO':^50}")
        print(f"{'-'*50}")
        
        # Definindo a função de avaliação padrão, se necessário
        if fitness_func is None:
            fitness_func = self.__rosenbrock_func__
        
        # Geração dos valores iniciais do PSO
        num_dimensions = bounds.shape[0]
        cur_swarm_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, num_dimensions))
        cur_swarm_vel = np.random.uniform(self.VMIN, self.VMAX, (num_particles, num_dimensions))
        cur_swarm_fitness = np.array([fitness_func(individual) for individual in cur_swarm_pos])
        
        # Obtenção dos melhores valores (iniciais)
        self.best_global_fitness = [np.min(cur_swarm_fitness) if is_min else np.max(cur_swarm_fitness)]
        self.best_global_particle = np.array([cur_swarm_pos[np.argmin(cur_swarm_fitness)] if is_min 
                                               else cur_swarm_pos[np.argmax(cur_swarm_fitness)]])
        self.best_local_fitness = [self.best_global_fitness[-1]]
        self.best_local_particle = [self.best_global_particle[-1]]
        self.best_mean_fitness = [np.mean(cur_swarm_fitness)]
        
        # Percorrendo as iterações
        self.itr = 1
        patience = 1
        global_fitness = np.copy(self.best_global_fitness[-1])
        global_particle = np.copy(self.best_global_particle[-1])
        
        while (self.itr < max_it and patience != max_patience):
            # Aumenta o número da iteração atual
            self.itr += 1
            
            # Para cada partícula, realizar análise
            local_all_fitness = np.zeros_like(cur_swarm_fitness)
            local_all_particle = np.zeros_like(cur_swarm_pos)
            for i in range(cur_swarm_pos.shape[0]):
                # Atualizando a velocidade das particulas, dentro do intervalo
                r1 = np.random.uniform(0, self.AC1)
                r2 = np.random.uniform(0, self.AC2)
                v_cognitive = r1 * (self.best_local_particle[-1] - cur_swarm_pos[i])
                v_social = r2 * (self.best_global_particle[-1] - cur_swarm_pos[i])
                cur_swarm_vel[i] = (self.W * cur_swarm_vel[i]) + (v_cognitive + v_social)
                cur_swarm_vel[i] = np.clip(cur_swarm_vel[i], self.VMIN, self.VMAX)
        
                # Atualizando a posição das partículas, dentro do intervalo
                cur_swarm_pos[i] += cur_swarm_vel[i]
                cur_swarm_pos[i] = np.clip(cur_swarm_pos[i], bounds[:, 0], bounds[:, 1])
                
                # Avaliação individual: obtendo a aptidão da partícula local
                local_fitness = fitness_func(cur_swarm_pos[i])
                local_particle = cur_swarm_pos[i]
                
                # Minimização
                if is_min:
                    # Atualizando o melhor local, se possível
                    if local_fitness < cur_swarm_fitness[i]:
                        cur_swarm_fitness[i] = cp(local_fitness)
                        cur_swarm_pos[i] = cp(local_particle)
                    
                    # Avaliação global: obtendo a aptidão das partícula vizinhas
                    if local_fitness < global_fitness:
                        global_fitness = cp(local_fitness)
                        global_particle = cp(local_particle)

                # Maximização
                else:
                    # Atualizando o melhor local, se possível
                    if local_fitness > cur_swarm_fitness[i]:
                        cur_swarm_fitness[i] = cp(local_fitness)
                        cur_swarm_pos[i] = cp(local_particle)
                    
                    # Avaliação global: obtendo a aptidão das partículas vizinhas
                    if local_fitness > global_fitness:
                        global_fitness = cp(local_fitness)
                        global_particle = cp(local_particle)

                # Salvando os dados, independente de melhora
                local_all_fitness[i] = cp(local_fitness)
                local_all_particle[i] = cp(local_particle)

            # Verificação de estagnação (paciência)
            if self.best_global_fitness[-1] == global_fitness:
                patience += 1
            else:
                patience = 1
                
            # Salvando os melhores dados da iteração
            self.best_global_fitness = np.append(
                self.best_global_fitness, 
                [global_fitness],
                axis = 0,
            )
            self.best_global_particle = np.append(
                self.best_global_particle,
                [global_particle],
                axis = 0,
            )
            self.best_mean_fitness = np.append(
                self.best_mean_fitness,
                [np.mean(local_all_fitness)],
                axis=0,
            )
            self.best_local_fitness = np.append(
                self.best_local_fitness,
                [np.min(local_all_fitness) if is_min
                 else np.max(local_all_fitness)],
                axis=0,
            )
            self.best_local_particle = np.append(
                self.best_local_particle,
                [local_all_particle[np.argmin(local_all_fitness)] if is_min
                 else local_all_particle[np.argmax(local_all_fitness)]],
                axis=0,
            )
        
        # Atualizando o tempo de execução
        self.exec_time = time() - self.exec_time
        
        print(f"Iteração {self.itr}")
        print(f"Melhor Aptidão: {self.best_global_fitness[-1]}")
        print(f"Melhor Partícula: {self.best_global_particle[-1]}")
        print(f"Tempo de Execução (s): {self.exec_time:.4f}")
        
        print(f"{'-'*50}")
        print(f"{'Fim do PSO':^50}")
        print(f"{'-'*50}")
        
        
    #####################################################
    #          Exemplo de Função de Avaliação           #
    #####################################################

    # Função de Rosenbrock
    def __rosenbrock_func__(self, x: list[float]) -> float:
        return (1 - x[0]) ** 2 + 100 * (x[1] - (x[0] ** 2)) ** 2