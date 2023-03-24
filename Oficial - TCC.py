#Discente: Iuri Leno Pereira da Silva
#Curso: Engenharia Elétrica - Universidade Federal do Amapá
#Gmail: iurileno.silva@gmail.com

# encoding: utf-8

import cv2 #Biblioteca de visão computacional - open source
import numpy as np #Bbilioteca de arranjos e matrizes
import datetime #Biblioteca de manipulação de data e hora
import time #Biblioteca utilizada como temporizador
import pyautogui #Biblioteca para fazer o registro em imagem (Print da tela - não utilizada)
from matplotlib import pyplot as plt #Biblioteca de plotagem gráfica

#Captura do video
video = cv2.VideoCapture('v2.MOV') #Entrada de vídeo

#Visualização do primeiro frame do vídeo em gráfico para obtenção dos pontos dos pixels
_, frame = video.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
plt.show()

#Inserção dos pontos (x,y) no terminal para elaboração da área delimitada
#Bastante manual esse processo, dar de melhorar
print("========================================================")
print("\033[1mDigite os valores de coordenadas no sentido anti-horário")
print("========================================================")

print("\033[1mPrimeiro quadrante")
x1 = int(input("\033[1mDigite o valor de X1: "))
y1 = int(input("\033[1mDigite o valor de Y1: "))
print("\033[1mSegundo quadrante")
x2 = int(input("\033[1mDigite o valor de X2: "))
y2 = int(input("\033[1mDigite o valor de Y2: "))
print("\033[1mTerceiro quadrante")
x3 = int(input("\033[1mDigite o valor de X3: "))
y3 = int(input("\033[1mDigite o valor de Y3: "))
print("\033[1mQuarto quadrante")
x4 = int(input("\033[1mDigite o valor de X4: "))
y4 = int(input("\033[1mDigite o valor de Y4: "))

#A área delimitada em banco e o resto da área do frame em preto
subtrator_de_fundo = cv2.bgsegm.createBackgroundSubtractorMOG() # Detectar objetos em movimento

#Para melhorar a imagem binária
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#Cores das classes
cores_classe = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

#Carregar as classes
nomes_classes = []
with open('tcc.names', 'r') as f:
    nomes_classes = [cname.strip() for cname in f.readlines()]

#Carregando os pesos e as configurações da rede neural
rna = cv2.dnn.readNet('tcc.weights', 'tcc.cfg')

#Setando os parâmetros da rede neural
modelo = cv2.dnn_DetectionModel(rna)
modelo.setInputParams(size=(416, 416), scale=1 / 255)

#Analise dos frames
while True:
    #Captura do frame
    _, frame = video.read()

    if not _:
        break

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Desenhar um retângulo em cada frame, para analisar o estado da área delimitada
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (255, 255, 255), -1)
    cor_aviso = (255, 0, 0)
    texto_estado = 'Sem deteccao'

    #Especificando os pontos delimitadores da área em que será analisada
    #Não alterar essas condições
    if x1 > x2:
        area = np.array([[x2, y2], [x1, y1], [x3, y3], [x4, y4]])

    if y3 > y2:
        area = np.array([[x2, y2], [x1, y1], [x4, y4], [x3, y3]])

    if x2 > x1:
        area = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    #Imagem auxiliar que atuará como o detector de movimentos
    imagem_auxiliar = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imagem_auxiliar = cv2.drawContours(imagem_auxiliar, [area], -1, 255, -1)  #Deixar a área delimitada em branco e o resto preto
    imagem_area = cv2.bitwise_and(cinza, cinza, mask=imagem_auxiliar)  #Area delimitada em cinza

    #Aplicar limiarização e dilatação na imagem binária
    limiarização = subtrator_de_fundo.apply(imagem_area)
    limiarização = cv2.morphologyEx(limiarização, cv2.MORPH_OPEN, kernel) #Abertura Morfológica - Eliminar os ruídos
    limiarização = cv2.dilate(limiarização, None, iterations=2) #Dilatação - Dilatar os elementos Significativos

    #Encontrar os contornos da limiarização, para filtrar somente os objetos em movimento
    objeto_limiarizado_em_movimento = cv2.findContours(limiarização, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for contornos in objeto_limiarizado_em_movimento:
        if cv2.contourArea(contornos) > 500: #Filtro de contornos - tamanho do filtro
            # etecção
            classes, scores, boxes = modelo.detect(frame, 0.8, 0.9) #Thresholding - limiar (isso tem que ser de acordo com a rede neural)

            #Percorrer todas as detecções
            for (classid, score, box) in zip(classes, scores, boxes):
                #Gerando uma cor para a classe
                cor = cores_classe[int(classid) % len(cores_classe)]

                #Inserindo o nome da classe pelo ID e o seu score de acuracia
                identificador = f"{nomes_classes[classid]}: {score:.3f}"

                #Desenhando o box da detecção
                cv2.rectangle(frame, box, cor, 2)

                #Escrevendo o nome da classe em cima da box do objeto
                cv2.putText(frame, identificador, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

                texto_estado = f'{nomes_classes[classid]} detectado'
                cor_aviso = (0, 0, 255)

                #ABAIXO TESTE DA CAPTURA DE TELA, RECOMENDA-SE UTILIZAR UM COMPUTADOR COM SISTEMA OPERACIONAL ELEVADO, SENÃO IRÁ TRAVAR
                '''for imagem in range(1):
                    foto = pyautogui.screenshot()
                    time.sleep(1)
                    foto.save("foto1%d.png" % imagem)'''

                #Escrever relatório do objeto detectado e o horário
                tempo = datetime.datetime.now()
                with open('Relatorio.txt', 'a') as arquivo:
                    arquivo.write('\nTipo de Veículo: {}\nData e Hora: {}\n'.format(identificador, tempo))

    #Visualizando o redor da área
    cv2.drawContours(frame, [area], -1, cor_aviso, 2)

    #Visualizando o estado de detecção de movimento
    cv2.putText(frame, texto_estado, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor_aviso, 2)

    #Resultados
    cv2.imshow('Video', frame) #Vídeo original
    cv2.imshow('Limiarizacao', limiarização) #Vídeo Limiarizado
    cv2.imshow('Imagem da Area delimitada em escala de cinza', imagem_area)

    #Espera da resposta
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

#Liberação da camera e destrói todas as janelas - menor custo de memória
video.release()
cv2.destroyAllWindows()