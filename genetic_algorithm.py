"""
Trabalho 1 - Algoritmos Genéticos
Data de Entrega: 20/04/2023

Aluno: Davi Augusto Neves Leite
RA: CCO230111
"""

import numpy as np          # Matrizes e Funções Matemáticas
from copy import copy as cp # Copiar objetos (não somente a referência)
from time import time       # Biblioteca para capturar o tempo de execução


#####################################################
#              Algoritmo Genético (GA)              #
#####################################################
class GA:
    """Algoritmo Genético para minimização ou maximização de
    funções contínuas de uma ou mais variáveis. 
    Utilização de bitstrings e seleção por torneio (maximizar)
    ou roleta (minimizar).

    
    Attributes:
        bounds : np.ndarray [[float, ...], ...]
            Lista contendo os intervalos das variáveis da função contínua \n
        bitstring_size : np.ndarray [int, int]
            Lista do tipo [int, int] contendo a quantidade máx. de dígitos para parte inteira [0] e decimal [1] das bitstrings \n
            Não está incluído o bit de sinal, o qual é acrescido separadamente \n
        population_size : int
            Tamanho desejado da população inicial \n
        current_population : np.ndarray [[float, ...], ...]
            Lista contendo a população atual. Utilizada para geração da população inicial \n
        best_global_population : np.ndarray [[float, ...], ...]
            Lista contendo a melhor população global encontrada nas gerações \n
        best_global_fitness : np.ndarray [float, ...]
            Lista contendo os valores de aptidão da melhor população global encontrada nas gerações \n
        best_local_population : np.ndarray [[float, ...], ...]
            Lista contendo a melhor população local encontrada nas gerações \n
        best_local_fitness : np.ndarray [float, ...]
            Lista contendo os valores de aptidão das melhores populações locais encontradas nas gerações \n
        all_mean_fitness : np.ndarray [float, ...]
            Lista contendo a média das aptidões locais durante as gerações \n
        best_generation : int      
            Número de gerações percorridas até o encontro da melhor população \n
        exec_time : float
            Tempo de execução do algoritmo, em segundos. \n
         
    Methods:
        generate_population (bounds: np.ndarray, population_size: int, bitstring_size: np.ndarray)
            Realiza a geração inicial de população, com base nos parâmetros de inicialização.
            Deve ser chamada uma vez antes da execução da função principal.
                bounds: lista contendo os intervalos das variáveis da função (padrão: [[-1.0, 1.0], [-1.0, 1.0]])
                population_size: tamanho desejado da população inicial (padrão: 10)
                bitstring_size: lista do tipo [int, int] contendo a quantidade máx. de dígitos para parte inteira [0] e decimal [1] das bitstrings (padrão: [11, 20])
        
        optimize (fitness_func: callable, is_min: bool, max_gen: int, max_patience: int, size_tournament: int, elitism: bool, elite_size: int, crossover_rate: float, mutation_rate: float)
            Aplicação do Algoritmo Genético, a partir dos parâmetros e da população inicial gerada na 'generate_population' 
                fitness_func: função de avaliação de aptidão. Ver exemplo de uso na '__rosenbrock_func__' (padrão: __rosenbrock_func__)
                is_min: booleana para atribuir algoritmo para min/max (padrão: True)
                max_gen: número máximo de gerações possíveis (padrão: 10000)
                max_patience: número máximo de iterações em que não houve melhora (padrão: 100)
                size_tournament: número de indivíduos selecionados aleatóriamente para o torneio, no caso de minimização (padrão: 3)
                elitism: booleana para considerar, ou não, o elitismo na reprodução (padrão: False)
                elite_size: quantidade de indivíduos para elitismo, quando 'elitism=True' (padrão: 3)
                crossover_rate: taxa de crossover para reprodução (padrão: 0.8)
                mutation_rate: taxa de mutação para reprodução (padrão: 0.2)
    
    Notes:
        
    """
    
    def __init__(self) -> None:
        self.bounds = None
        self.bitstring_size = None
        self.population_size = None
        self.current_population = None
        self.best_global_population = None
        self.best_global_fitness = None
        self.best_local_population = None
        self.best_local_fitness = None
        self.all_mean_fitness = None
        self.best_generation = 0
        self.exec_time = time() # Tempo inicial de execução
        
    
    #####################################################
    #               Conversão Bitstring                 #
    #####################################################

    # Float para Binário
    def __float_to_binary__(
        self,
        values: np.ndarray,
        bitstring_size = np.array([11, 20]),
    ):
        """Função responsável por converter um valor,
        em float, para representação binária,
        com número máximo de casas inteiras e decimas.
        
        OBS: O primeiro bit da parte inteira é de sinal!
        (Considere como tamanho total = 1 + int_digits + dec_digits)

        Args:       
            values : np.ndarray[float, ...]
                Variáveis em número em float \n
            bitstring_size : np.ndarray[int, int]
                Lista a quantidade máx. de dígitos para parte inteira [0] e decimal [1] (padrão: [11, 20] = 31 bits + 1 bit sinal) \n
            
        Returns:
            binary_converted : list[bitstring, ...]
                Lista com número convertido em binário \n
        """

        # Inicia a lista de saída
        binary_converted = []

        # Percorrendo cada valor gerado de cada variável
        for value in values:
            # Separar a parte inteira da decimal
            signal = '1' if value < 0 else '0'
            int_part = abs(int(value))
            dec_part = abs(value) - int_part

            # Converter a parte inteira para binário, completando com zero à esquerda
            int_str = np.binary_repr(int_part).zfill(bitstring_size[0])

            # Conversão da parte decimal para binário, completando com zeros à direita
            dec_str = ""
            for _ in range(bitstring_size[1]):
                # Realiza a potência de 2
                dec_part *= 2

                # Caso tenha ficado parte inteira, acrescenta "1"
                if dec_part >= 1:
                    dec_str += "1"
                    dec_part -= 1
                # Caso não tenha ficado parte inteira, acrescenta "0"
                else:
                    dec_str += "0"

            # Salvando o resultado na lista
            binary = signal + int_str + dec_str
            binary_converted.append(list(binary))

        return binary_converted


    # Binário para Float
    def __binary_to_float__(
        self,
        values: np.ndarray,
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        bitstring_size = np.array([11, 20]),
    ):
        """Função responsável por converter um valor,
        em representação binária, para float,
        com número máximo de casas inteiras e decimas.
        
        OBS: O primeiro bit da parte inteira é de sinal!
        (Considere como tamanho total = 1 + int_digits + dec_digits)

        Args:       
            values : np.ndarray[bitstring, ...]
                Lista com as variáveis em representação binária \n
            bounds : np.ndarray[[float, float], ...]
                Lista contendo os intervalos min/max das variáveis da função (padrão: [[-1.0, 1.0], [-1.0, 1.0]]) \n
            bitstring_size : np.ndarray[int, int]
                Lista a quantidade máx. de dígitos para parte inteira [0] e decimal [1] (padrão: [11, 20] = 31 bits + 1 bit sinal) \n
            
        Returns:
            float_converted : list[float, ...]
                Lista de variáveis convertidas em float \n
        """

        # Obtém o maior valor possível do binário, pela quantidade inteira (exclui bit de sinal)
        largest_binary_num = (2 ** bitstring_size[0]) - 1

        # Inicia a lista de saída
        float_converted = []

        # Percorrendo a quantidade de variáveis da função
        for i in range(bounds.shape[0]):
            # Separa a parte inteira da decimal
            signal, int_str, dec_str = values[i][0], values[i][1:bitstring_size[0] +
                                                            1], values[i][bitstring_size[0] + 1:]

            # Convertendo sinal
            signal_value = (-1) if signal == '1' else (1)

            # Converter a parte inteira para número
            int_str = "".join(str(c) for c in int_str)
            int_num = int(int_str, 2)

            # Definir o intervalo de conversão (escala)
            scaled_int_num = bounds[i][0] + (int_num / largest_binary_num) * (bounds[i][1] - bounds[i][0])

            # Converter a parte decimal para número pela fórmula inversa da decimal
            dec_num = np.sum([int(x) * 2 ** (- (k + 1))
                            for k, x in enumerate(dec_str)])

            # Recuperando o número float por completo
            float_converted.append(signal_value * (scaled_int_num + dec_num))

        return float_converted


    ########################################
    #              1 - Geração             #
    ########################################
    
    def generate_population(
        self,
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        population_size = 10,
        bitstring_size = np.array([11, 20]),
    ):
        """Função responsável por gerar uma nova população,
        em bitstring (vetor de bits), advindos do float.

        OBS: O primeiro bit da parte inteira é de sinal!
        (Considere como tamanho total = 1 + int_digits + dec_digits)

        Args: 
            bounds : np.ndarray[[float, float], ...]
                Lista com valores do intervalo da função (padrão: [[-1.0, 1.0], [-1.0, 1.0]]) \n
            population_size : int
                Tamanho da população (padrão: 10) \n
            bitstring_size : np.ndarray[int, int]
                Lista a quantidade máx. de dígitos para parte inteira [0] e decimal [1] (padrão: [11, 20] = 31 bits + 1 bit sinal) \n
        """
        
        # Salvando valores de tamanho de população
        self.bounds = bounds
        self.population_size = population_size
        self.bitstring_size = bitstring_size
        
        # Geração de Float aleatório para cada variável
        individual_vals = np.random.uniform(
            bounds[:, 0], bounds[:, 1], (population_size, bounds.shape[0]))

        # Conversão de Float para bitstring e salva a população inicial
        self.current_population = np.array([self.__float_to_binary__(values=individual, bitstring_size=bitstring_size)
                            for individual in individual_vals])


    ########################################
    #              2 - Seleção             #
    ########################################

    # Seleção por Roleta: Problemas de MAXIMIZAÇÃO
    def __roulette_selection__(
        self,
        cur_population: np.ndarray,
        fitness: np.ndarray,
    ):
        """Seleção de indivíduo pelo método da Roleta.

        Args:
            cur_population : np.ndarray[[bitstring, ...], ...]
                Vetor contendo a população atual para seleção \n
            fitness : np.ndarray[float, ...]
                Vetor contendo todos os valores de aptidão da população \n
            
        Returns:
            new_population : np.ndarray[[bitstring, ...], ...]
                Vetor contendo a nova população selecionada \n
            selected_fitness : np.ndarray[float, ...]
                Vetor contendo todos os valores de aptidão da população selecionada \n
        """

        # Determinar a porção da roleta para cada indivíduo no intervalo [start; end]
        proportions = fitness / (np.sum(fitness))

        # Posição dos indivíduos selecionados aleatoriamente com base nas porções
        idx_selected = np.random.choice(
            cur_population.shape[0], size=cur_population.shape[0], p=proportions)

        return cur_population[idx_selected], fitness[idx_selected]


    # Seleção por Torneio: Problemas de MINIMIZAÇÃO
    def __tournament_selection__(
        self,
        cur_population: np.ndarray,
        fitness: np.ndarray,
        size = 3,
    ):
        """Seleção de indivíduo pelo método do Torneio.

        Args:
            cur_population : np.ndarray[[bitstring, ...], ...]
                Vetor contendo a população atual para seleção \n
            fitness : np.ndarray[float, ...]
                Vetor contendo todos os valores de aptidão da população \n
            size : int
                Número de indivíduos selecionados aleatóriamente (padrão: 3) \n
            
        Returns:
            new_population : np.ndarray[[bitstring, ...], ...]
                Vetor contendo a nova população selecionada \n
            selected_fitness : np.ndarray[float, ...]
                Vetor contendo todos os valores de aptidão da população selecionada \n
        """

        # Criação do vetor para nova população, com base no tamanho da população atual
        new_population = np.empty_like(cur_population)
        selected_fitness = np.empty_like(fitness)

        # Percorrendo o vetor da população atual (número de linhas)
        for i in range(cur_population.shape[0]):
            # Escolha aleatória dos indivíduos candidatos
            idx_candidates = np.random.choice(
                cur_population.shape[0], size, replace=False)

            # Escolha do vencedor com base no MENOR valor obtido
            idx_winner = np.argmin(fitness[idx_candidates])

            # Salvando o vencedor na nova população
            new_population[i] = cur_population[idx_candidates[idx_winner]]
            selected_fitness[i] = fitness[idx_candidates[idx_winner]]

        return new_population, selected_fitness

    
    ########################################
    #           3 - Reprodução             #
    ########################################

    # Crossover
    def __crossover__(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        crossover_rate = 0.8,
    ):
        """Aplicação de crossover entre dois indivíduos.

        Args:
            parent1 : np.ndarray[bitstring, ...]
                Vetor representando o primeiro indivíduo \n
            parent2 : np.ndarray[bitstring, ...]
                Vetor representando o segundo indivíduo \n
            crossover_rate : float
                Float que representa a taxa acontecimento de crossover (padrão: 0.8) \n
            
        Returns:
            child1 : np.ndarray[bitstring, ...]
                Vetor representando o primeiro filho \n
            child2 : np.ndarray[bitstring, ...]
                Vetor representando o segundo filho \n
        """

        # Cria os vetores iniciais para abrigar os filhos
        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        # Percorrendo a quantidade de variáveis
        for i in range(parent1.shape[0]):

            # Para ocorrer o crossover, um número aleatório deve ser menor ou igual a taxa
            if np.random.rand() <= crossover_rate:
                # Sorteia um ponto de corte
                crossover_point = np.random.randint(0, parent1[i].shape[0])

                # Realização do crossover
                child1[i] = np.concatenate(
                    (parent1[i][:crossover_point], parent2[i][crossover_point:]))
                child2[i] = np.concatenate(
                    (parent2[i][:crossover_point], parent1[i][crossover_point:]))
            else:
                # Não ocorrência de crossover, apenas mantém os pais
                child1[i] = parent1[i]
                child2[i] = parent2[i]

        return child1, child2


    # Mutação
    def __mutation__(
        self,
        individual: np.ndarray,
        mutation_rate = 0.2,
    ):
        """Aplicação de mutação em um indivíduo.

        Args:
            individual : np.ndarray[bitstring, ...]
                Vetor representando o indivíduo a sofrer mutação \n
            mutation_rate : float
                Float que representa a taxa de mutação (padrão: 0.2) \n
            
        Returns:
            mutant : np.ndarray[bitstring, ...]
                Vetor representando o indivíduo com mutação \n
        """

        # Cria o vetor inicial para abrigar o mutante
        mutant = np.copy(individual)

        # Percorrendo cada bitstring do indivíduo
        for i in range(mutant.shape[0]):
            # Percorrendo cada caractere na bitstring
            for j in range(mutant.shape[1]):
                # Verifica se o número aleatório é menor ou igual à taxa de mutação
                if np.random.uniform() <= mutation_rate:
                    # Troca o bit dependendo do valor
                    mutant[i, j] = '1' if mutant[i, j] == '0' else '0'

        return mutant


    # Reprodução
    def __reproduction__(
        self,
        selected_population: np.ndarray,
        selected_fitness: np.ndarray,
        elitism = False,
        elite_size: int = 3,
        crossover_rate=0.8,
        mutation_rate=0.2,
    ):
        """Reprodução de uma determinada população, em bitstring, 
        considerando crossover e mutação.

        Args:
            selected_population : np.ndarray[[bitstring, ...], ...]
                Vetor com a população selecionada \n
            selected_fitness : np.ndarray[float, ...]
                Vetor contendo todos os valores de aptidão da população \n
            elitism : bool
                Considerar ou não o elitismo (padrão: False) \n
            elite_size : int (opcional se 'elitism=False')
                Quantidade de indivíduos para elitismo (padrão: 3) \n
            crossover_rate : float
                Taxa de crossover (padrão: 0.8) \n
            mutation_rate : float
                Taxa de mutação (padrão: 0.2) \n
            
        Returns:
            new_population : np.ndarray[[bitstring, ...], ...]
                Vetor com a nova população \n
        """

        # Seleção de todos os pais para reprodução
        parents = selected_population

        # Elitismo
        if elitism:
            # Seleção dos indivíduos de elite (primeiros 'elite_size' da população previamente ordenada...)
            elite = selected_population[np.argsort(
                selected_fitness)][::-1][:elite_size]

            # Seleção dos indivíduos sem a elite, para geração de filhos
            parents = selected_population[np.argsort(
                selected_fitness)][::-1][elite_size:]

        # Criação de novos indivíduos com crossover e mutação
        children = []
        for i in range(0, parents.shape[0]-1, 2):
            # Percorre a população em dois a dois, selecionando pares contínuos
            parent1, parent2 = parents[i], parents[i + 1]

            # Fase de crossover
            child1, child2 = self.__crossover__(parent1, parent2, crossover_rate)

            # Fase de mutação
            child1 = self.__mutation__(child1, mutation_rate)
            child2 = self.__mutation__(child2, mutation_rate)

            # Adiciona os filhos na nova população
            children.append(child1)
            children.append(child2)

        if elitism:
            # Caso o número da população seja ímpar, adiciona o último indivíduo
            if ((parents.shape[0] - elite_size) % 2 == 1):
                children.append(parents[-1])
        else:
            # Caso o número da população seja ímpar, adiciona o último indivíduo
            if (parents.shape[0] % 2 == 1):
                children.append(parents[-1])

        # Adicionando a elite e os filhos gerados
        new_population = np.concatenate(
            (elite, children)) if elitism else np.array(children)

        return new_population


    ########################################
    #         3 - Função Principal         #
    ########################################

    def optimize(
        self,
        fitness_func: callable = None,
        is_min = True,
        max_gen = 10000,
        max_patience = 100,
        size_tournament = 3,
        elitism = False,
        elite_size = 3,
        crossover_rate = 0.8,
        mutation_rate = 0.2,
    ):
        """Aplicação do Algoritmo Genético, a partir
        de uma população de bitstring, para min/max de uma 
        função multivariável.

        Seleção por Torneio para Minimização. \n
            -> 'size_tournament': define o tamanho do torneio

        Seleção por Roleta para Maximização.

        Args:
            fitness_func : callable
                Função de avaliação de aptidão (ver exemplo de uso na '__rosenbrock_func__ (x: list[float])') \n
            is_min : bool
                Booleana para atribuir algoritmo para min/max (padrão: True)
                    -- Se 'True', irá verificar minimização \n
                    -- Se 'False', irá verificar maximização \n
            max_gen : int
                Número máximo de gerações possíveis (padrão: 10000) \n
            max_patience : int
                Número máximo de iterações em que não houve melhora (padrão: 100) \n
            size_tournament : int (opcional se 'is_min=False')
                Número de indivíduos selecionados aleatóriamente para o torneio (padrão: 3) \n
            elitism : bool
                Considerar ou não o elitismo (padrão: False) \n
            elite_size : int (opcional se 'elitism=False') 
                Quantidade de indivíduos para elitismo (padrão: 3) \n
            crossover_rate : float
                Taxa de crossover (padrão: 0.8) \n
            mutation_rate : float
                Taxa de mutação (padrão: 0.2) \n
        """
        print(f"{'-'*50}")
        print(f"{'Algoritmo Genético':^50}")
        print(f"{'-'*50}")
        
        # Definindo a função de avaliação padrão, se necessário
        if fitness_func is None:
            fitness_func = self.__rosenbrock_func__
        
        # Avaliação da população inicial
        current_fitness = np.array([fitness_func(self.__binary_to_float__(individual, bounds=self.bounds, bitstring_size=self.bitstring_size)) for individual in self.current_population])

        # Recuperando o melhor da população inicial e definindo valores iniciais
        self.best_global_population = np.array([self.__binary_to_float__(individual, bounds=self.bounds, bitstring_size=self.bitstring_size) for individual in [
                               self.current_population[np.argmin(current_fitness)] if is_min else self.current_population[np.argmax(current_fitness)]]])
        self.best_global_fitness = [np.min(current_fitness) if is_min else np.max(current_fitness)]
        self.best_local_population = [self.best_global_population[-1]]
        self.best_local_fitness = [self.best_global_fitness[-1]]
        self.all_mean_fitness = [np.mean(current_fitness)]

        # Percorrendo as gerações, a partir da população inicial
        self.best_generation = 1
        patience = 1
        best_population = [self.best_global_population[-1]]
        best_fitness = [self.best_global_fitness[-1]]
        
        while (self.best_generation < max_gen and patience != max_patience):
            # Aumenta o número da geração atual
            self.best_generation += 1
            
            #############################
            #!     Fase de Seleção     !#
            #############################
            if is_min:
                # Minimização: uso do Torneio
                self.current_population, current_fitness = self.__tournament_selection__(
                    cur_population=self.current_population,
                    fitness=current_fitness,
                    size=size_tournament,
                )
            else:
                # Maximização: uso da Roleta
                self.current_population, current_fitness = self.__roulette_selection__(
                    cur_population=self.current_population,
                    fitness=current_fitness,
                )
            
            ################################
            #!     Fase de Reprodução     !#
            ################################
            self.current_population = self.__reproduction__(
                selected_population=self.current_population,
                selected_fitness=current_fitness,
                elitism=elitism,
                elite_size=elite_size,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
            )
            
            ###############################
            #!     Fase de Avaliação     !#
            ###############################
            current_fitness = np.array(
                [fitness_func(self.__binary_to_float__(
                    individual,
                    bounds=self.bounds,
                    bitstring_size=self.bitstring_size))
                for individual in self.current_population]
            )
            
            local_population = None
            local_fitness = None
            if is_min:
                # Minimização: atualização dos valores globais
                if np.min(current_fitness) < self.best_global_fitness[-1]:
                    patience = 1
                    best_population = [self.__binary_to_float__(
                        individual,
                        bounds=self.bounds,
                        bitstring_size=self.bitstring_size)
                        for individual in
                        [self.current_population[np.argmin(current_fitness)]]]
                    best_fitness = np.array([np.min(current_fitness)])
                else:
                    best_population = [self.best_global_population[-1]]
                    best_fitness = [self.best_global_fitness[-1]]

                # Verificação da Paciência
                if np.min(current_fitness) == self.best_global_fitness[-1]:
                    patience += 1
                
                # Salvando valores locais
                local_population = [self.__binary_to_float__(
                                    individual,
                                    bounds=self.bounds,
                                    bitstring_size=self.bitstring_size)
                                    for individual in [self.current_population[np.argmin(current_fitness)]]]
                local_fitness = np.array([np.min(current_fitness)])
            else:
                # Maximização: atualização dos valores globais
                if np.max(current_fitness) > self.best_global_fitness[-1]:
                    patience = 1
                    best_population = [self.__binary_to_float__(
                        individual,
                        bounds=self.bounds,
                        bitstring_size=self.bitstring_size)
                        for individual in
                        [self.current_population[np.argmax(current_fitness)]]]
                    best_fitness = np.array([np.max(current_fitness)])
                else:
                    best_population = [self.best_global_population[-1]]
                    best_fitness = [self.best_global_fitness[-1]]

                # Verificação da Paciência
                if np.max(current_fitness) == self.best_global_fitness[-1]:
                    patience += 1
                    
                # Salvando valores locais
                local_population = [self.__binary_to_float__(
                                    individual,
                                    bounds=self.bounds,
                                    bitstring_size=self.bitstring_size)
                                    for individual in [self.current_population[np.argmax(current_fitness)]]]
                local_fitness = np.array([np.max(current_fitness)])
            
            # Salvando os melhores valores globais
            self.best_global_population = np.append(
                self.best_global_population,
                best_population,
                axis = 0
            )
            self.best_global_fitness = np.append(
                self.best_global_fitness,
                best_fitness,
                axis = 0
            )
            
            # Salvando os melhores valores locais
            self.best_local_population = np.append(
                self.best_local_population,
                local_population,
                axis = 0
            )
            self.best_local_fitness = np.append(
                self.best_local_fitness,
                local_fitness,
                axis = 0
            )
            
            # Salvando as médias das aptidões
            self.all_mean_fitness = np.append(self.all_mean_fitness, np.array([np.mean(current_fitness)]), axis=0)
        
        # Atualizando o tempo de execução
        self.exec_time = time() - self.exec_time
        
        print(f"Geração {self.best_generation}")
        print(f"Melhor Aptidão: {self.best_global_fitness[-1]}")
        print(f"Melhor Indivíduo: {self.best_global_population[-1]}")
        print(f"Tempo de Execução (s): {self.exec_time:.4f}")

        print(f"{'-'*50}")
        print(f"{'Fim do Algoritmo Genético':^50}")
        print(f"{'-'*50}")

        
    #####################################################
    #          Exemplo de Função de Avaliação           #
    #####################################################

    # Função de Rosenbrock
    def __rosenbrock_func__(self, x: list[float]) -> float:
        return (1 - x[0]) ** 2 + 100 * (x[1] - (x[0] ** 2)) ** 2

