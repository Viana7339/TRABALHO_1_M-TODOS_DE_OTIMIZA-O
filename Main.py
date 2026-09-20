import FP_Linearizado as FPL
import numpy as np

## Parâmetros Globais:
Sbase = 100
system = 39
## Sistema IEEE 39 BARRAS:
if system == 39:
        caminho_sistema = r"C:\Cepel\Anarede\V120001\Exemplos\NewEngland.PWF"
        ## Limites de Pgmin e Pgmax, em pu, das usinas do sistema, na mesma ordem dos barramentos com geração (PV e VTHETA) do ANAREDE
        Pgmin = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        Pgmax = [10.40, 6.46, 7.25, 6.52, 5.08, 6.87, 5.80, 5.64, 8.65, 11.0]
        ## Limites de fluxo de P Ativa nas linhas e transformadores do IEEE39BUS (EM PU), na mesma ordem do ANAREDE:
        Tmax = [6, 10, 5, 5, 5, 5, 5, 6, 5,
                12, 9, 9, 4.8, 18, 9, 9, 9, 6,
                6, 9, 5, 5, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 9, 9, 9, 9, 6, 9, 6,
                9, 6, 9, 6, 6, 6, 6, 12]
        # print(len(Tmax))
        ## Custos de Geração das unidades, na mesma ordem dos barramentos com geração (PV e VTHETA) do ANAREDE:
        Custo_ger = [15.37, 11.29, 8.80, 8.00, 11.40, 10.45, 10.03, 10.15, 7.98, 8.0] #$/MW
        Custo_ger = [c*Sbase for c in Custo_ger] #$/pu
else:
        raise ValueError(f"Os sistemas cadastrados são: 39. Cadastre outro sistema ou tente um disponível")

## Criação do objeto:
Teste = FPL.FP_Linearizado(caminho_sistema, Sbase, Pgmin, Pgmax, Tmax, Custo_ger)

## Otimização para o caso base:
B = Teste.B_Matrix()
Ag, Bred, Pdem, X, A_linha, Theta_linha, Pger, T, nlin = Teste.Modelagem(B)
Teste.Otimização(Ag, Bred, Pdem, X, A_linha, Theta_linha, Pger, T, nlin)

# Análise de contingências:
Teste.Analise_Contingencias()