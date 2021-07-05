###############################################################
#              EXERCÍCIO PROGRAMA 1 - MAP3121                 #
#            Autovalores e Autovetores de Matrizes            #
#           Tridiagonais Simétricas - O Algoritmo QR          #
#                                                             #
#                   INTEGRANTES:                              #
#          Alessandro Jiã Iong Li - 10291791                  #
#       Pedro Paulo Teles Alves da Silva - 11234332           #
#                                                             #
#             Professor: Fabio Armando Tal                    #
#                    Julho de 2021                            #
###############################################################

import numpy as np
import matplotlib.pyplot as plt

# Obtenção de matriz:
  # Pede entradas ao usúario;
  # Retorna uma matriz com os valores de entrada. 
def obter_matriz():
  dimensao = int(input("Escolha a dimensao da matriz: "))

  matriz = np.zeros((dimensao, dimensao), dtype=np.float64);

  i = 0
  j = 0
  while (i < dimensao):
    while (j < dimensao):
      matriz[i,j] = float(input(f"Entre com o elemento da linha {i+1} coluna {j+1} da matriz (a{i+1}{j+1}): "))
      j = j + 1
    i = i + 1
    j=0
    
  return matriz

#__________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# Rotação de Givens - Qk(A, k)
  # Recebe uma matriz 'A' quadrada e um índice 'k';
  # Retorna uma matriz de rotação de Givens de modo a anular o beta_K.
def Qk(A, k):
  n = A.shape[0] # Dimensão da matriz

  k = k - 1 # No enuncidado os betas são possuem sub-índice de 1 a (n-1).
            # Adota-se o sistema de índices do enunciado. Assim, nessa linha
            # muda-se para o sistema de índices do python (1)

  # Obtenção dos cossenos e senos
  c = A[k,k] / np.sqrt((A[k,k]**2) + (A[k+1, k]**2))        
  s = (-A[k+1,k]) / np.sqrt((A[k,k]**2) + (A[k+1, k]**2))  

  # Construção da matriz de Givens
  Q = np.identity(n)
  Q[k,k] = c
  Q[k+1, k+1] = c
  Q[k+1, k] = s
  Q[k, k+1] = -s

  return Q

#__________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# Fatoração QR - fatoracao_QR(A)
  # Recebe uma matriz 'A';
  # Retorna um Q e um R da fatoração QR da matriz A.
def fatoracao_QR(A):
  n = A.shape[0]

  Q = np.identity(n)
  R = A

  for indice_beta in range(1, n):
    Q_temp = Qk(R, indice_beta)
    Q = Q @ np.transpose(Q_temp) # Q = (Q_1)' * (Q_2)' * ... * (Q_n-1)'
    R = Q_temp @ R               # R = (Q_n-1) * ... * (Q_2) * (Q_1) * A

  return Q, R

#__________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#  Heurística de Wilkinson - wilkinson(A)
  # Recebe uma matriz A
  # Retorna o Mu pela heurística de Wilkinson.

def wilkinson(A):
  n = A.shape[0] - 1
  dk = (A[n-1,n-1] - A[n,n])/2

  mu = 0

  if dk >= 0:
    mu = A[n,n] + dk - np.sqrt((dk**2) + (A[n,n-1])**2)
  elif dk < 0:                                         # apenas p/ verificar se d_k eh valido
    mu = A[n,n] + dk + np.sqrt((dk**2) + (A[n,n-1])**2)

  return mu

#__________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# Método QR para diagonalizar matriz - algoritmo_QR(A, eps, deslocado)
  # Rebebe uma matriz tridiagonal simétrica 'A'
  # Recebe uma tolerância 'eps'
  # Recebe um parâmetro que indica o uso de deslocamento espectral ou não
  # Retorna uma matriz diagonal de autovalores de 'A' 
  # Retorna uma matriz cujas colunas são autovetores de  'A'

def algoritmo_QR(A, eps, deslocado=True):
  n = A.shape[0]

  V = np.identity(n)
  
  mu = 0
  
  k = 0
  m = n
  
  autovalores = np.array([])
  autovetores = np.identity(n)

  while m >= 2:
    while abs(A[m-1, m-2]) > eps:
      
      #Caso nao seja a primeira iteracao obtenha o fator de Wilkinson
      if k>0 and deslocado == True:
        mu = wilkinson(A[0:m, 0:m])
      else: mu = 0

      #Iteracao do autovetor
      A_temporario = np.identity(n)                           #Crie uma matriz de formato original papel em branco!
                                                              #O enunciado deixa explicito "consideramos beta_(n-1) = 0", portanto posso zerar
                                                              #os betas e considerar que a matriz ja se encaminha para uma diagonal

      for i in range(0, autovalores.shape[0]):                #Adicione os autovalores já encontrados caso os tenha
        A_temporario[i,i] = autovalores[i]
      A_temporario[0:m, 0:m] = A[0:m, 0:m]                    #Para o restante da matriz papel em branco, coloque a submatriz em iteracao
      A_temporario = A_temporario - mu*np.identity(n)         #Faca iteracao QR para encontrar autovetor
      Q_temporario, R_temporario = fatoracao_QR(A_temporario)
      V = V @ Q_temporario            
          
      #Iteracao do autovalor                                  #Itere a submatriz cujas entradas nao englobam os autovalores ja encontrados
      A = A - mu*np.identity(m)                        
      Q, R = fatoracao_QR(A)
      A = (R @ Q) + (mu*np.identity(m))

      k = k + 1
    
    #Caso encontrou um autovalor, tire-o da matriz para que nao mais seja alterado. Faca o mesmo com o autovetor
    autovalores = np.append(autovalores, A[m-1, m-1]) # adicione, a direita, o novo autovalor encontrado
    autovetores[:, n-m] = V[:, m-1]                   # guarde, a direita, o novo autovetor correspondente ao autovalor encontrado

    m = m - 1
    A = A[0:m, 0:m] #Obtenha a nova submatriz na qual sera aplicada a iteracao QR
    
  autovalores = np.append(autovalores, A[m-1, m-1]) # Adicione o ultimo autovalor econtrado
  autovetores[:, n-m] = V[:, m-1]                   # Adicione o ultimo autovetor encontrado

  return autovalores, autovetores, k

#__________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# Teste do item A
def gerar_matriz_teste(dimensao):
  n = dimensao
  A = np.zeros((n,n))
  
  for i in range(0, n):
    A[i, i] = 2
    
  
  for i in range(1, n):
    A[i, i-1] = -1
    A[i-1, i] = -1
  
  return A

#_______________________________________________________________________________

def gerar_autovalores_esperados(dimensao):
  n = dimensao

  autovalores = np.zeros(n)

  for j in range(0, n):
    autovalores[j] = 2*(1-np.cos(((j+1)*(np.pi))/(n + 1)))

  return autovalores

#_______________________________________________________________________________

def gerar_autovetores_esperados(dimensao):
  n = dimensao
  autovetores = np.identity(n)

  for j in range(0, n):
    vetor_j = np.array([])
    for i in range(0, n):
      coordenada = np.sin(((j+1)*(i+1)*np.pi)/(n+1))
      vetor_j = np.append(vetor_j, coordenada)
    
    autovetores[:, j] = np.transpose(vetor_j)

  return autovetores

#_______________________________________________________________________________

def organizar_resposta(autovalores, autovetores):
  autovalores_organizado = np.sort(autovalores)
  autovetores_organizado = np.identity(autovetores.shape[0])

  for i in range(0, autovalores.shape[0]):
    for j in range(0, autovalores.shape[0]):
      if autovalores_organizado[i] == autovalores[j]:
        autovetores_organizado[:, i] = autovetores[:, j]
  
  return autovalores_organizado, autovetores_organizado

#_______________________________________________________________________________
def teste_algoritmo_QR(dimensao, eps=1e-6, deslocado=True): 
  n = dimensao
  A = gerar_matriz_teste(n)

  autovalores_esperados = gerar_autovalores_esperados(n)
  autovetores_esperados = gerar_autovetores_esperados(n)
  
  autovalores_obtidos, autovetores_obtidos, k = algoritmo_QR(A, eps, deslocado)
  autovalores_obtidos, autovetores_obtidos = organizar_resposta(autovalores_obtidos, autovetores_obtidos)

  deslocado_texto = ""
  if deslocado == True:
    deslocado_texto = "Deslocado"
  elif deslocado == False:
    deslocado_texto = "Não Deslocado" 
  print("\nTeste do algoritmo QR - ", deslocado_texto, "( n=", n, "; eps=", eps, ")")
  print("Numero de iteraçÕes: ", k, "\n")

  #Autovetores
  #Faca eles terem o mesmo formato somente multiplicando o vetor obtido por numero real
  for i in range(0, n):
    k = (autovetores_esperados[0, i] / autovetores_obtidos[0, i])
    autovetores_obtidos[:, i] = autovetores_obtidos[:, i] * k

  #Autovalores
  for i in range(0, n):
    print("Autovalor ", i+1)
    print("Autovalor Esperado: ", autovalores_esperados[i]) 
    print("Autovalor Obtido:   ", autovalores_obtidos[i])
    print("Autovetor Esperado: ", autovetores_esperados[:, i])
    print("Autovetor Obtido:   ", autovetores_obtidos[:, i], '\n')

#_______________________________________________________________________________

# Obtenção da matriz de coeficientes elasticos: matriz_A(n)
   # Recebe um numero 'n' de massas
   # Recebe um parametro que informa se as constantes k são crescentes ou não
   # Retorna a matriz tridiagonal simetrica positiva A com os k's correspondentes ao problema  

def matriz_A(n, k_crescente = True):
  m = 2
  kelasticas = []
  i = 1
  kaux = 0
  A = np.zeros((n,n),dtype=np.int)

  if k_crescente:
    while i < (n+2):
      kaux = 40 + 2*i
      kelasticas.append(kaux)
      i = i + 1
    for i in range(0, n):
      A[i, i] = (kelasticas[i] + kelasticas[i+1])/m
    for i in range(1, n):
      A[i-1, i] = (-kelasticas[i])/m
      A[i, i-1] = (-kelasticas[i])/m  

  else:                 
    while i < (n+2):
      kaux = 40 + (2*(-1)**i)
      kelasticas.append(kaux)
      i = i + 1
    for i in range(0, n):
      A[i, i] = (kelasticas[i] + kelasticas[i+1])/m
    for i in range(1, n):
      A[i-1, i] = (-kelasticas[i])/m
      A[i, i-1] = (-kelasticas[i])/m   

  return A

#_______________________________________________________________________________

# Resolução do sistema - resolver_sistema()
  # Recebe a matriz de coeficientes eláticos
  # Recebe vetor de posiçoes iniciais
  # Retorna um vetor de frequencias
  # Retorna Amplitude das cossenoides

def resolver_sistema(A, Posicoes, eps=1e-6, deslocado=True):
  Lambda, Q, k = algoritmo_QR(A, eps, deslocado)
  Qt = np.transpose(Q)
  n = Lambda.shape[0]
  Frequencias = np.zeros(n)
  Y = np.zeros(n)
  for i in range(0,n):
    Frequencias[i] = np.sqrt(Lambda[i])
  Y = Qt @ np.transpose(Posicoes)
  Amplitudes = Q * Y

  Modos_naturais = Q

  return Amplitudes, Frequencias, k, Modos_naturais

#_______________________________________________________________________________

# Funcao que grafica posicoes - plotar_posicoes()
# Recebe uma vetor de frequencias naturais.
# Recebe uma matriz de amplitudes de cossenos cujas linhas correspondem a cada x(t),
# isto, é a11 é a amplitude que multiplica um cosseno que é parcela de x1(t) e possui frequencia omega1;
#         a12 é a amplitude que multiplica um cosseno que é parcela de x1(t) e possui frequencia omega2;
#         aij é a amplitude que multiplica um cosseno que é parcela de xi(t) e possui frequencia omegaj.
# Recebe de forma opcional os parâmetros duracao da janela e periodo de amostragem. 
# Plota a posicao de cada massa em funcao do tempo, sendo, por exemplo, x1(t) a posicao da massa m1.

def plotar_posicoes(Frequencia, Amplitude, duracao = 10.0, espacamento = 0.025):
  n = Frequencia.shape[0]
  t = np.arange(0.0, duracao, espacamento)

  for i in range(0, n):
    s = np.zeros(t.shape[0])
    for j in range(0, n):
      s = s + Amplitude[i, j]*np.cos(Frequencia[j] * t)
    plt.plot(t,s)  
    plt.grid(color = 'b', linestyle = 'solid', linewidth = 0.25)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Posição [m]")
    plt.title('Posição em relação ao equilíbrio da massa ' + str(i+1))
    plt.show()

def plotar_junto(Frequencia, Amplitude, duracao=10.0, espacamento=0.025):
  n = Frequencia.shape[0]
  t = np.arange(0.0, duracao, espacamento)

  for i in range(0, n):
    s = np.zeros(t.shape[0])
    for j in range(0, n):
      s = s + Amplitude[i, j]*np.cos(Frequencia[j] * t)
    plt.plot(t,s)  
    plt.grid(color = 'b', linestyle = 'solid', linewidth = 0.25)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Posição [m]")
  plt.title('Posição em relação ao equilíbrio das massas')
  plt.show()

#_______________________________________________________________________________

#Main
def main():
  print("\nESCOLHA A OPÇÃO DIGITANDO O NÚMERO CORRESPONDENTE.\n")

  print("Escolha uma opção:")
  print("1. Visualizar as respostas do enunciado. \n2. Realizar teste personalizado.")
  enunciado_personalizado = input("Escolha: ")

  #1. Respostas ao enunciado.
  if enunciado_personalizado == '1':
    print("\nEscolha um item do enunciado:")
    print("1. Item A \n2. Item B. \n3. Item C.")
    a_b_c = input("Escolha: ")
    
    ##1. Item A. (OK)
    if a_b_c == '1':
      N = [4, 8, 16, 32]
      for n in N:
        gerar_matriz_teste(n)
        teste_algoritmo_QR(n, deslocado = True)
        teste_algoritmo_QR(n, deslocado=False)

    ##2. Item B.
    elif a_b_c == '2':

      #Perguntar se quer ver deslocamento espectral
      print("\nEscolha uma opção:")
      print("1. Com deslocamento espectral. \n2. Sem deslocamento espectral.")
      deslocado = int(input("Escolha: "))

      if deslocado == 1:
        deslocado = True
      elif deslocado == 2:
        deslocado = False

      #Perguntar o tipo de arranjo
      print("\nEscolha o arranjo de X0:")
      print("1. Primeiro arranjo de X0 \n2. Segunda arranjo de X0 \n3. Terceiro arranjo de X0:")
      arranjo = int(input("Escolha: "))
      X0 = np.array([])

      #Arranjo 1
      if arranjo == 1:
        X0 = np.array([-2, -3, -1, -3, -1])
        print("\nSistema de 5 massas de 2kg e 6 molas de K's crescentes: ")
        print("X0: ", X0)

        A = matriz_A(5, k_crescente=True)
        Amplitudes, Frequencias, k, Modos_naturais = resolver_sistema(A, Posicoes=X0, eps=1e-6, deslocado=deslocado)

        print("Número de iterações:", k, "\neps considerado: 1e-6\n")


        for i in range(0, Frequencias.shape[0]):
          print(f"Frequência   {i+1}:   {Frequencias[i]}" )
          print(f"Modo natural {i+1}: {Modos_naturais[:, i]}\n")

        plotar_posicoes(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)
        plotar_junto(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)
       

      #Arranjo 2
      elif arranjo == 2:
        X0 = np.array([1, 10, -4, 3, -2])
        print("\nSistema de 5 massas de 2kg e 6 molas de K's crescentes: ")
        print("X0: ", X0)

        A = matriz_A(5, k_crescente=True)
        Amplitudes, Frequencias, k, Modos_naturais = resolver_sistema(A, Posicoes=X0, eps=1e-6, deslocado=deslocado)

        print("Número de iterações:", k, "\neps considerado: 1e-6\n")

        for i in range(0, Frequencias.shape[0]):
          print(f"Frequência   {i+1}:   {Frequencias[i]}" )
          print(f"Modo natural {i+1}: {Modos_naturais[:, i]}\n")

        plotar_posicoes(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)
        plotar_junto(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)

      #Arranjo 3
      elif arranjo == 3:
        print("\nSistema de 5 massas de 2kg e 6 molas de K's crescentes: ")

        A = matriz_A(5, k_crescente=True)

        #Obter modo de maior frequencias
        autovalores, autovetores, k = algoritmo_QR(A, eps=1e-6, deslocado=deslocado)
        indice_maior_lamb = 0
        maior = autovalores[0]
        for i in range(0, autovalores.shape[0]):
          if autovalores[i] > maior:
            maior = autovalores[i]
            indice_maior_lamb = i
        
        X0 = autovetores[:, indice_maior_lamb]
      
        print("X0 correspondente ao modo de maior frequencias: ", X0)

        Amplitudes, Frequencias, k, Modos_naturais = resolver_sistema(A, Posicoes=X0, eps=1e-6, deslocado=deslocado)

        print("Número de iterações:", k, "\neps considerado: 1e-6\n")

        for i in range(0, Frequencias.shape[0]):
          print(f"Frequência   {i+1}:   {Frequencias[i]}" )
          print(f"Modo natural {i+1}: {Modos_naturais[:, i]}\n")

        plotar_posicoes(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)
        plotar_junto(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)
      
    ##3. Item C.
    elif a_b_c == '3':
      #Perguntar se quer ver deslocamento espectral
      print("\nEscolha uma opção:")
      print("1. Com deslocamento espectral. \n2. Sem deslocamento espectral.")
      deslocado = int(input("Escolha: "))

      if deslocado == 1:
        deslocado = True
      elif deslocado == 2:
        deslocado = False

      #Perguntar o tipo de arranjo de X0
      print("\nEscolha o arranjo de X0:")
      print("1. Primeiro arranjo de X0 \n2. Segunda arranjo de X0 \n3. Terceiro arranjo de X0:")
      arranjo = int(input("Escolha: "))

      #Apresentação
      print("\nSistema de 10 massas de 2kg e 11 molas de K's alternadas e velocidades iniciais nulas:\n")

      #Gerar matriz do sistema
      A = matriz_A(10, k_crescente=False)

      if arranjo == 1: #1. Primeiro arranjo de X0.
        X0 = np.array([-2, -3, -1, -3, -1, -2, -3, -1, -3, -1])
        print("X0: ", X0)

        #Resolução do sistema
        Amplitudes, Frequencias, k, Modos_naturais = resolver_sistema(A, X0, eps=1e-6, deslocado=deslocado)
        print("Número de iterações:", k, "\neps considerado: 1e-6\n")

        #Frequencias e modos naturais
        for i in range(0, Frequencias.shape[0]):
          print(f"Frequência   {i+1}:   {Frequencias[i]}" )
          print(f"Modo natural {i+1}: {Modos_naturais[:, i]}\n")

        #Plotar Graficos
        plotar_posicoes(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)
        plotar_junto(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)

      if arranjo == 2: #2. Segundo arranjo de X0.
        X0 = np.array([1, 10, -4, 3, -2, 1, 10, -4, 3, -2])
        print("X0: ", X0)

        #Resolução do sistema
        Amplitudes, Frequencias, k, Modos_naturais = resolver_sistema(A, X0, eps=1e-6, deslocado=deslocado)
        print("Número de iterações:", k, "\neps considerado: 1e-6\n")

        #Frequencias e modos naturais
        for i in range(0, Frequencias.shape[0]):
          print(f"Frequência   {i+1}:   {Frequencias[i]}" )
          print(f"Modo natural {i+1}: {Modos_naturais[:, i]}\n")

        #Plotar Graficos
        plotar_posicoes(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)
        plotar_junto(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)

      if arranjo == 3: #3. Terceiro arranjo de X0.

        #Obter modo de maior frequencias
        autovalores, autovetores, k = algoritmo_QR(A, eps=1e-6, deslocado=deslocado)
        indice_maior_lamb = 0
        maior = autovalores[0]
        for i in range(0, autovalores.shape[0]):
          if autovalores[i] > maior:
            maior = autovalores[i]
            indice_maior_lamb = i
        
        #X0 do modo de maior frequencia
        X0 = autovetores[:, indice_maior_lamb]
        print("X0 correspondente ao modo de maior frequencia: ", X0)

        #Resolucao o sistema
        Amplitudes, Frequencias, k, Modos_naturais = resolver_sistema(A, Posicoes=X0, eps=1e-6, deslocado=deslocado)
        print("Número de iterações:", k, "\neps considerado: 1e-6\n")

        #Frequencia e modos naturais
        for i in range(0, Frequencias.shape[0]):
          print(f"Frequência   {i+1}:   {Frequencias[i]}" )
          print(f"Modo natural {i+1}: {Modos_naturais[:, i]}\n")

        #Plotar graficos
        plotar_posicoes(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)      
        plotar_junto(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)

  #2. Teste personalizado.
  elif enunciado_personalizado == '2':
    print("\nEscolha uma opção de teste: ")
    print("1. Testar algoritmo QR.")
    print("2. Testar sistema massa-mola.")
    algoritmo_sistema = input("Escolha: ")

    #1. Teste do algoritmo QR.
    if algoritmo_sistema == '1':

      # Escolha se deve ocorrer deslocamento espectral
      print("\nEscolha uma opção: ")
      print("1. Com deslocamento espectral.")
      print("2. Sem deslocamento espectral.")
      deslocado = int(input("Escolha: "))
      deslocado_texto = ""
      if deslocado == 1:
        deslocado = True
        deslocado_texto = "com Deslocamento Espectral"
      elif deslocado == 2:
        deslocado = False
        deslocado_texto = "sem Deslocamento Espectral"

      print("\n")
      A = obter_matriz() # Definicao da matriz
      eps = float(input("Defina a tolerância 'eps': ")) #Definicao do eps
      autovalores, autovetores, k = algoritmo_QR(A, eps=eps, deslocado=deslocado) # Iteracao QR

      # Respostas
      print(f"\nTeste do algoritmo QR {deslocado_texto}")
      print(f"Numero de iterações: {k}\n")
      print(f"Matriz A:\n", A, "\n")

      for i in range(0, autovalores.shape[0]):
        print(f"Autovalor {i+1}: ", autovalores[i])
        print(f"Autovetor {i+1}: ", autovetores[:, i], "\n")
           
    #2. Testar sistema massa-mola.
    elif algoritmo_sistema == '2':

      #Parâmetros do sistema
      num_massas = int(input("\nDefina o número de massas: ")) #Número de massas

      print("\nEscolha uma opção: ") # Natureza das constantes k's
      print("1. Contantes elásticas crescentes.")
      print("2. Constantes elásticas alternadas.")
      crescente = int(input("Escolha: "))
      print("\n")
      crescente_texto = ""
      if crescente == 1:
        crescente = True
        crescente_texto = "crescentes"
      elif crescente == 2:
        crescente = False
        crescente_texto = "alternadas"
      
      if True == False:
        print("Não aguento mais fazer isso!!!")

      pos_iniciais = np.array([]) # Condições de contorno
      for i in range(0, num_massas):
        pos_iniciais = np.append(pos_iniciais, float(input(f"Posição inicial da massa {i+1}: ")))

      print("\nEscolha uma opção: ") # Aplicar deslocamento espectral ou não
      print("1. Aplicar deslocamento espectral na solução.")
      print("2. Não aplicar deslocamento espectral.")
      desloc = int(input("Escolha: "))
      desloc_texto = ""
      if desloc == 1:
        desloc = True
        desloc_texto = "com Deslocamento Espectral"
      elif desloc == 2:
        desloc = False
        desloc_texto = "sem Deslocamento Espectral"

      erro = float(input("\nDefina a tolerância 'eps': "))

      # Solução do sistema
      A = matriz_A(n=num_massas, k_crescente=crescente)
      Amplitudes, Frequencias, k, Modos_naturais = resolver_sistema(A, Posicoes=pos_iniciais, eps=erro, deslocado=desloc)
      
      # Resposta
      print(f"\nSISTEMA DE {num_massas} MASSAS E {num_massas+1} MOLAS INICIALMENTE EM REPOUSO.\n")
      print(f"Conjunto de molas de constantes K {crescente_texto}.")
      print(f"X0: {pos_iniciais}\n")
      print(f"Solução {desloc_texto}.")

      print(f"Número de iterações: {k}.\neps considerado: {erro}.\n")

      #Frequencias e modos naturais
      for i in range(0, Frequencias.shape[0]):
        print(f"Frequência   {i+1}:   {Frequencias[i]}" )
        print(f"Modo natural {i+1}: {Modos_naturais[:, i]}\n")

      #Plotar graficos
      plotar_posicoes(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)      
      plotar_junto(Frequencias, Amplitudes, duracao=10.0, espacamento=0.025)

#_______________________________________________________________________________

continuar = True
while continuar:
  main()
  continuar = int(input("\nContinuar? "))