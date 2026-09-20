## Bibliotecas
import pandas as pd
import numpy as np
from pyxparser.parser import AnaredeParser #Biblioteca para leitura de arquivos do anarede
import networkx as nx
from scipy.optimize import linprog
import copy

class FP_Linearizado:
    
    def __init__(self, caminho_sistema, Sbase, Pgmin, Pgmax, Tmax, Custo_ger):

        ## Globalização dos parâmetros:
        self.caminho_sistema = caminho_sistema #Local do arquivo PWF
        self.sbase = Sbase #Sbase do Sistema
        ## Vetores dos limites de geração [1 x NG]:
        self.Pgmin = Pgmin #Limite mínimo de geração de P ativa das barras de geração
        self.Pgmax = Pgmax #Limite máximo de geração de P ativa das barras de geração
        ## Vetores de limite de fluxo nos elementos dlin (linhas e transformadores) [1 x NLIN]:
        self.Tmax = Tmax #Fluxo no sentido barra de -> barra para
        self.Tmin = [-T for T in Tmax] #Fluxo no sentido barra para -> barra de
        ## Vetor de Custos para as unidadedes geradoras [1 X NG]:
        self.Custo_ger = Custo_ger

        ## Acesso à topologia do sistema:
        self.parser = AnaredeParser() #Criação do objeto
        self.sistema = self.parser.parse_file(self.caminho_sistema) #Acesso ao sistema
        # print(self.sistema.keys()) #Chaves que podem ter no sistema (barras, linhas, etc)

        ## Acesso aos dados do sistema:
        self.dbar = self.sistema.get('DBAR', []) #Todas as barras do sistema
        print("\nExemplo da forma como são apresentados os dados de barra:")
        print(self.dbar[1]) #Print de uma barra do sistema
        self.dlin = self.sistema.get('DLIN', []) #Todas as linhas do sistema
        print("\nExemplo da forma como são apresentados os dados de linha:")
        print(self.dlin[7]) #Print da uma linha do sistema

        ## Número de barras do sistema:
        self.nbus = len(self.dbar)
        print(f"\nO número de barramentos do sistema é de {self.nbus}\n")

        ## Número de elementos (linhas + transformadores) do sistema:
        self.nlin = len(self.dlin)
        print(f"Há {len(self.dlin)} elementos (Linhas de Transmissão ou transformadores) no sistema original\n")

        ## Loop para obter a numeração (identificação) das barras do Dbar:
        self.buses_id = [] #Lista para armazenar as identificações dos barramentos do sistema
        for bus in self.dbar: #Loop em função dos barramentos do sistema
            self.buses_id.append(int(bus['number']))#Identificação de cada barramento
        print(f"Os barramentos do sistema são dados por:\n{self.buses_id}\n")

        ## Identificação da barra de referência: 
        for i, bus in enumerate(self.dbar):#Loop em função do total de barramentos do sistema
            if bus["type"] == 2: #Barra VTeta
                self.Theta_ref = np.deg2rad(float(bus["angle"])) #Ângulo theta especificado (conversão para radianos -> o ANAREDE fornece em graus)
                self.Idx_VTheta = i #Índice do barramento VTheta
        print(f"O Barramento VTheta é o {self.buses_id[self.Idx_VTheta]} com ângulo de {self.Theta_ref} radianos\n")

        ## Obtenção do total de barras PV:
        self.npv = 0 #Variável para representar o total de barras PV
        for i, bus in enumerate(self.dbar): #Loop em função do total de barramentos do sistema
            if bus["type"] == 1: #Se for PV
                self.npv += 1 #Atualização do total de barras PV
        print(f"O total de barras PV é de {self.npv}\n")

    def B_Matrix(self, dlin=None):

        ## Se não for passado um dlin alterado (para eventual análise de contingências), considera-se o dlin original do sistema:
        if dlin is None:
            dlin = self.dlin

        ## Inicialização da matriz B completa [NBUS x NBUS]:
        self.B = np.zeros((self.nbus, self.nbus))

        ## Loop de incorporação das linhas de transmissão e transformadores em fase na matriz Ybus:
        for cont, elemento in enumerate(dlin, start=1): #start=1 faz com que o contador cont comece em 1

            # print(f"Incorporação do elemento {cont} na matriz Y\n")

            ## Índices linha/coluna da Matriz:
            i = self.buses_id.index(int(elemento["from_bus"])) #Índice da "Barra de" do elemento
            k = self.buses_id.index(int(elemento["to_bus"])) #Índice da "Barra para" do elemento
            # print(f"O elemento conecta a barra de {int(elemento["from_bus"])} à barra {int(elemento["to_bus"])}\n")

            ## Incorporação da Linha de Transmissão:
            if not elemento["tap"]: #Se a informação do tap estiver vazia

                # print(f"O elemento {cont} é uma Linha de Transmissão\n")

                ## Para o FPL temos que o fluxo de potência na LT é dado por Pkm = xkm^−1∙ θkm:

                ## Obtenção dos parâmetros da linha (divisão por 100 pois no PWF os valores estão em %):
                Xl = float(elemento["reactance"])/100 #Reatância
                ## Cálculo da Susceptância série da linha:
                Bl = 1/Xl #Susceptância

                ## Incorporação na matriz B:
                self.B[i, i] += Bl 
                self.B[k, k] += Bl 
                self.B[i, k] -= Bl
                self.B[k, i] -= Bl

            ## Incorporação do Transformador em Fase:
            elif elemento["tap"] and not elemento["phase_shift"]: #Se a informação do tap não estiver vazia e a informação da defasagem estiver

                # print(f"O elemento {cont} é um Transformador em Fase\n")

                ## Obtenção dos parâmetros do Transformador em Fase:
                X_trafo = float(elemento["reactance"])/100 #Reatância

                ## Para o modelo do transformador em fase a equação do FP é dada por: (Pkm = a ∙ xkm^−1 ∙ θkm)
                ## Ou: Pkm = θkm / Xeq , Xeq = (a / xkm)
                # a = 1/float(elemento["tap"]) #Tap (O fator "a" é dado por 1/t e o ANAREDE fornece o t
                ## Para a solução do FPL o ANAREDE considera a = 1:
                a = 1 #Valor do tap
                B_trafo = a/X_trafo #Susceptância

                ## Incorporação na matriz B:
                self.B[i, i] += B_trafo
                self.B[k, k] += B_trafo
                self.B[i, k] -= B_trafo
                self.B[k, i] -= B_trafo

        # print(self.B)
        return self.B

    def Modelagem(self, B, dlin=None):

        ## Se não for passado um dlin alterado (para eventual análise de contingências), considera-se o dlin original do sistema:
        if dlin is None:
            dlin = self.dlin #dlin igual ao do sistema orignal
            nlin = self.nlin #Número de elementos do dlin igual ao do sistema original
        else: #Se for passado um dlin com contingência
            nlin = (self.nlin - 1) #O número de elementos do dlin é uma unidade menor em função da contingência

        ## RESTRIÇÃO DE IGUALDADE: Ag . Pg – Pd = Bred . θ’
        print("Modelagem da equação: Ag . Pg – Pd = Bred . θ’\n")

        ## Cosntrução da Matriz Ag de incidência [NBUS X NPG]:
        Ag = np.zeros((self.nbus, self.npv+1)) #Inicialização de zeros #ROZI
        aux = 0 #Auxiliar para preenchimento da matriz
        for i, bus in enumerate(self.dbar): #Loop em função do total de barramentos do sistema
            if bus["type"] == 1 or bus["type"] == 2: #Se for PV ou VTETA (OU SEJA, GERADOR)
                Ag[i,aux] = 1 #Indicação da presença do gerador aux na barra i
                aux +=1 #Atualização da variável auxiliar (próximo gerador)
        print(f"A matriz de incidência Ag é de proporção {Ag.shape[0]} x {Ag.shape[1]} dada por:")    
        print(Ag)

        ## Vetor das potências geradas [NG x 1] (Variável do problema) NG = NPV + 1 (BARRA VTETA):
        Pger = np.zeros((self.npv+1, 1)) #Inicialização de zeros 
        print(f"A matriz Pger é de proporção {Pger.shape[0]} x {Pger.shape[1]}")
        # print(Pger)

        ## Vetor [NBUS x 1] das potências demandadas:
        Pdem = np.zeros((self.nbus, 1)) #Inicialização de zeros
        for i, bus in enumerate(self.dbar): #Loop em função do total de barramentos do sistema (INCLUINDO VTheta)
            Pdem[i,0] = float(bus["active_load"])/self.sbase #Obtenção de Pdem para a barra i
        print(f"A matriz Pdem é de proporção {Pdem.shape[0]} x {Pdem.shape[1]}")
        # print(Pdem)

        ## Obtenção da matriz Breduzida [NBUS X (NBUS - 1)]:
        print(f"A matriz B é de proporção {B.shape[0]} x {B.shape[1]}")
        # print(B)
        Bred = np.delete(B, self.Idx_VTheta, axis=1) #Matriz Bred, pela remoção da coluna referente à barra de referência
        print(f"A matriz B reduzida é de proporção {Bred.shape[0]} x {Bred.shape[1]}")
        # print(Bred)

        ## Vetor de ângulos [(NBUS-1) X 1] (SEM A BARRA VTHETA) (Variável do problema):
        Theta_linha = np.zeros(((self.nbus-1), 1)) #Inicialização de zeros
        print(f"O vetor de ângulos é de proporção: {Theta_linha.shape[0]} x {Theta_linha.shape[1]}")

        ## RESTRIÇÃO DE IGUALDADE: T = X^-1 . A' . θ'
        print("\nModelagem da equação: T = X^-1 . At . θ'\n")

        ## O FP NAS LINHAS PELO MODELO LINEARIZADO É DADO POR:
        ## tij = (θi – θj)/ xij -> LTs
        ## tij = a * (θi – θj)/ xij -> Trafos em fase

        ## Vetor de fluxo nas linhas [NLIN x 1] (Variável do problema):
        T = np.zeros((nlin, 1)) #Inicializaçõ de zeros
        print(f"O vetor de fluxos é de proporção: {T.shape[0]} x {T.shape[1]}")

        ## Matriz diagonal X com reatância xij [NLIN x NLIN] X=diag{xl1​,xl2​,⋯} Γ = 1/X:
        X = np.zeros((nlin, nlin)) #Inicialização de zeros
        for i, elemento in enumerate(dlin):
            if not elemento["tap"]: #Se for uma LT
                X[i,i] = float(elemento["reactance"])/100 #Reatância
            elif elemento["tap"] and not elemento["phase_shift"]: #Se for trafo em fase
                ## Obtenção do tap:
                # a = 1/float(elemento["tap"]) #Tap (O fator "a" é dado por 1/t e o ANAREDE fornece o t)
                a = 1 #ANAREDE USA O TAP EM 1 PARA O FPL
                X[i,i] = (float(elemento["reactance"])/100)/a #Reatância equivalente
        print(f"A Matriz diagonal X é de proporção: {X.shape[0]} x {X.shape[1]}")
        # print(X)

        ## Matriz A de Incidência Barra-Ramo [NBUS x NLIN]:
        A = np.zeros((self.nbus, nlin)) #Inicialização de zeros
        for lin, elemento in enumerate(dlin): #Loop em função das linhas para peenchimento da matriz
            i = self.buses_id.index(int(elemento["from_bus"])) #Índice da "Barra de" do elemento
            k = self.buses_id.index(int(elemento["to_bus"])) #Índice da "Barra para" do elemento
            A[i,lin] = 1
            A[k, lin] = -1
        print(f"A matriz de incidência A é de proporção {A.shape[0]} x {A.shape[1]}")
        # print(A)
        ## Matriz A_linha(Remoção da linha correspondente à barra de referência):
        A_linha = np.delete(A, self.Idx_VTheta, axis=0) #Matriz A_linha ([NBUS-1 x NLIN])
        print(f"A matriz de incidência A_linha é de proporção {A_linha.shape[0]} x {A_linha.shape[1]}") 
        # print(A_linha)

        return Ag, Bred, Pdem, X, A_linha, Theta_linha, Pger, T, nlin

    def Otimização(self, Ag, Bred, Pdem, X, A_linha, Theta_linha, Pger, T, nlin, Idx_elemento_removido = None):

        ## Vetor de variáveis do problema:
        Var = np.concatenate((Theta_linha, Pger, T))
        # print(Var)

        ## FOB (Minimização)
        ## Como os ângulos e os fluxos não afetam o custo da fob é necessário incorporá-los com peso zero na FOB
        ## Os Vetores de ângulos e Fluxos foram criados zerados, logo:
        Fob = np.concatenate([np.transpose(Theta_linha).flatten(), self.Custo_ger, np.transpose(T).flatten()])
        print("O vetor de custos da FOB é dado por:")
        print(Fob)
        # print(np.transpose(Theta_linha))
        # print(self.Custo_ger)
        # print(np.transpose(T))

        ## Coeficientes das equações de igualdade (Aeq):
        Nulo_T = np.zeros((self.nbus, nlin))
        Identidade_T = np.eye(nlin)
        Nulo_Pger = np.zeros((nlin, self.npv+1))
        Aeq = np.block([
              [Bred, -Ag, Nulo_T],
              [(- (np.linalg.inv(X)) @ (np.transpose(A_linha)) ), Nulo_Pger, Identidade_T]  
        ])

        ## Termos independentes das equações de igualdade (Beq):
        Beq = np.concatenate([-Pdem.flatten(), np.zeros(nlin)]) ##Arrumar tamanho

        ## Limites das variáveis:

        ## Limites dos ângulos:
        Theta_linha_min = [None] * (self.nbus-1)
        Theta_linha_max = [None] * (self.nbus-1)

        ## Alteração do vetor de limites de fluxo para as contingências:
        if Idx_elemento_removido is None: #Se não for uma caso de contingência
            Tmin = self.Tmin
            Tmax = self.Tmax
        else: #Para a contingência
            Tmin = copy.deepcopy(self.Tmin)
            Tmin.pop(Idx_elemento_removido)
            Tmax = copy.deepcopy(self.Tmax)
            Tmax.pop(Idx_elemento_removido)

        ## Concatenação dos limites mínimo e máximos:
        limites_min = np.concatenate((Theta_linha_min, self.Pgmin, Tmin))
        limites_max = np.concatenate((Theta_linha_max, self.Pgmax, Tmax))
        # print(limites_min)
        # print(limites_max)

        ## Formatação dos limites:
        bounds = []
        for i in range(len(limites_min)):
            bounds.append((limites_min[i], limites_max[i]))
        print("Os limites das variáveis são dados por:")
        print(bounds)

        ## Uso da função de otimização:
        res = linprog(c=Fob,A_eq=Aeq, b_eq=Beq, bounds=bounds, method="simplex")
        # print(res) 

        ## Organização dos resultados:
        if res.success == True: #Se houve solução

            print("O problema de otimização convergiu!")
            ## Ângulos das barras em graus (EXCETO BARRA SLACK):
            Theta_linha = res.x[:(self.nbus-1)] #Obtenção das variáveis de ângulo
            Theta_linha_deg = [float(np.round(float(np.rad2deg(ang)),1)) for ang in Theta_linha] #Conversão de radianos para graus
            print(f"Os ângulos (em graus) nos barramentos do sistema (Exceto barra slack) são dados por:")
            print(Theta_linha_deg)

            ## Potências Ativas geradas (em pu):
            Pger = res.x[(self.nbus-1):(self.nbus-1)+self.npv+1]
            print(f"As potências ativas das barras de geração (em pu) são dadas por:")
            print(Pger)

            ## Fluxo de potência nas linhas (em PU):
            T = res.x[(self.nbus-1+self.npv+1):] 
            print(f"Os fluxos de potência ativa (em pu) nas linhas do sistema são dadas por:")
            print(T)
            print(len(T))

            ## Obtenção do valor da FOB:
            print(f"O Valor da FOB é de: {res.fun} $")

        else: #Se divergiu

            print("O problema de otimização não apresentou solução ótima!")

        return res.success, res.fun, Theta_linha, Pger, T

    def Analise_Contingencias(self):

        print(f"Ínicio do processo de análise de contingências (N-1)")

        ## Obtenção das conexões dos elementos dlin do sistema:
        Linhas = [] #Lista de tuplas para armazenar as barras de e para dos elementos
        for linha in self.dlin: ##Loop em função dos elementos de dlin
            ## Conjuntos barra de e barra para dos elementos de dlin:
            Bus_de = self.buses_id.index(int(linha["from_bus"])) #Índice do barramento correspondente à barra de
            Bus_para = self.buses_id.index(int(linha["to_bus"])) #índice do barramento correspondente à barra para
            Linhas.append((Bus_de, Bus_para))
        print(f"Os pares de barramentos de/para dos elementos do sistema (total de {len(Linhas)}) é dado por:\n{Linhas}\n")

        ## Criação do Grafo:
        G = nx.Graph()
        G.add_edges_from(Linhas)

        ## Determinação dos elementos que se removidos provocam ilhamento
        Ilhas = list(nx.bridges(G)) #Função para identificar linhas que quebram o grafo em duas partes (ilhamento)
        print(f"Os elementos que provocam ilhamento (na forma barra de e barra para) são: {Ilhas}")

        ## Determinação dos índices dos elementos de dlin que podem ser removidos (sem causar ilhamento):
        Idx_linhas_contingencias = [] #Lista para armazenar os índices dos elementos que podem ser removidos na análise de contingências
        for i, linha in enumerate(Linhas): #loop em função de todos os elementos dlin
            if linha not in Ilhas: #Se não for um caso de ilhamento
                Idx_linhas_contingencias.append(i) #Adição do índice do elemento que não causa ilhamento

        print(f"Os elementos que podem ser removidas são dados pelos índices (total de {len(Idx_linhas_contingencias)}):\n{Idx_linhas_contingencias}\n")

        ## Loop para o FPO Linearizado para cada caso n-1:
        dict_contingencias = {} #Dicionário para armazenar os resultados do FPO de cada contingência
        for Idx_elemento in Idx_linhas_contingencias: #Loop em função do total de elementos que podem ser removidos
        # for Idx_elemento in range(1): #Teste -> Para visualizar o resultado de um elemento específico
        #     Idx_elemento = 1 #Teste -> Para visualizar o resultado de um elemento específico
            print(f"Análise n-1 para o elemento de índice {Idx_elemento}\n")
            dlin_novo = copy.deepcopy(self.dlin) #Cópia do sistema original
            # print(dlin_novo)
            dlin_novo.pop(Idx_elemento) #Remoção do elemento em contingência
            # print(dlin_novo)
            ## Resolução do FPO Linear para o caso com contingência:
            B_novo = self.B_Matrix(dlin_novo) #Cáculo da nova matriz B
            Ag, Bred, Pdem, X, A_linha, Theta_linha, Pger, T, nlin = self.Modelagem(B_novo, dlin_novo) #Modelagem para o caso
            convergiu, fob, Theta_linha, Pger, T = self.Otimização(Ag, Bred, Pdem, X, A_linha, Theta_linha, Pger, T, nlin, Idx_elemento) #Otimização
            Theta_linha_deg = [float(np.round(float(np.rad2deg(ang)),1)) for ang in Theta_linha] #Conversão de radianos para graus
            dict_contingencias[Idx_elemento]= [Idx_elemento, self.dlin[Idx_elemento]["from_bus"], self.dlin[Idx_elemento]["to_bus"],  convergiu, fob, Theta_linha_deg, Pger, T] #Preenchimento do dicionário
            print(dict_contingencias[Idx_elemento])

        ## Exportação dos resultados:
        caminho_planilha = r"C:\Users\mathe\OneDrive\Área de Trabalho\Mestrado\Métodos de Otimização\Trabalho_1\Resultados\Analise_de_Contingencias.xlsx" #Local do arquivo
        colunas = ["Idx_elemento_removido"] + ["Barra De"] + ["Barra Para"] + ["Convergência"] + ["FOB"] + ["Ângulos_graus"] + ["Pgerado_pu"] + ["Fluxo_pu"] #Nome das colunas

        ## Criação de um dataframe para exportação dos dados via planilha em excel:
        df = pd.DataFrame(dict_contingencias.values(), columns=colunas)
        ## Exportação:
        df.to_excel(caminho_planilha, index=False)
        print("A planilha de resultados da análise de contingências foi gerada com sucesso!\n")
        print(f"Uma amostra dos dados apresentados na planilha ({df.shape[0]} casos) é dada por:\n{df.head(10)}\n")
        print(f"O total de contingências analisadas foi de {len(dict_contingencias)}")


