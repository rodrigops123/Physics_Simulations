#-----------------------------------------------------------------------------#
#   Projeto 3                                                                 #
#-----------------------------------------------------------------------------#
#   Codigo usado para simular uma configuracao quadrada de spins              #
#   seguindo o modelo de Ising 2D, por meio do algoritmo de                   #
#   Metropolis.                                                               #
#                                                                             #
#   Disciplina: Metodos Computacionais em Fisica (4300331)                    #
#   Professor: Luis Gregorio Dias da Silva                                    #
#   Monitores: Joao Victor Ferreira Alves e Lauro Barreto Braz                #
#                                                                             #
#   Autor: Rodrigo Pereira Silva. NUSP: 11277128                              #
#   Contato: rodrigopereirasilva@usp.br                                       #
#-----------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from numba import jit                    #biblioteca de otimizacao para funcoes

@jit(nopython=True)
def E_flip(i, j, config, h, J):
    '''
    Essa funcao calcula o Eflip para posterior uso na funcao principal de
    simulacao.
    Input (nessa ordem):
    i       --> indice de linha
    j       --> indice de coluna
    config  --> matriz quadrada de spins que assume valor +1 ou -1
    h       --> intensidade do campo magnetico externo
    J       --> intensidade de interacao entre spins vizinhos
    
    Output:
    Eflip   --> diferenca de energia entre configuracoes apos flipar o spin na posicao (i,j)
    '''
    #A estrategia de tirar o resto da divisao de (i+1)%Nspins ou (j+1)%Nspins serve
    #para retornar para a coordenada 0 quando estamos na borda. Note que nao foi 
    #necessario fazer o mesmo para i-1 e j-1, pois o python reconhece indices negativos
    Nspins = config.shape[0]
    return 2*config[i,j]*( J*( config[(i+1)%Nspins, j] + config[i-1,j] 
                              + config[i, (j+1)%Nspins] + config[i, j-1] ) + h )


@jit(nopython=True)
def ising_2d(config, N_var, T, h, J):
    '''
    Essa funcao simula a configuracao quadrada de spins
    Input (nessa ordem):
    config  --> matriz quadrada contendo NxN spins assumindo valores +1 ou -1
    N_var   --> numero de varreduras completas a serem realizadas na grade
    T       --> temperatura na qual o sistema esta submetido
    h       --> intensidade do campo magnetico externo
    J       --> intensidade de interacao entre spins vizinhos
    '''
    Nspins = config.shape[0]
    m_var = np.zeros(N_var)  #magnetizacao ao fim de cada varredura
    for k in range(N_var):
        index = np.arange(0,Nspins*Nspins,1)    #define os indices a serem usados
        np.random.shuffle(index)         #embaralha os indices aleatoriamente
        for n in index:
            i = n//Nspins      #define a coordenada de linha a partir do indice
            j = n%Nspins       #define a coordenada de coluna a partir do indice
            Eflip = E_flip(i, j, config, h, J)  
            if Eflip<=0:
                config[i,j] *= -1
            elif Eflip>0 and (np.random.uniform(0,1) < np.exp(-Eflip/T) ):
                config[i,j] *= -1
        m_var[k] = np.mean(config)
    m_alpha = np.mean(config)   #calcula a configuracao apos o fim da simulacao
    return (m_alpha, config, m_var)

#Exemplo de uso da rotina para uma cadeia com 300x300 spins para a temperatura
#critica do sistema, assumindo ausencia de campo magnetico externo e forca de
#interacao entre spins vizinhos unitaria

config = np.random.choice([-1,1], size=(300,300))
config = ising_2d(config = config, N_var = 1000, T = 2.268, h = 0, J = 1)[1]

#Plot da configuracao:
plt.imshow(config, cmap='Greys')
plt.show()