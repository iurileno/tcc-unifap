import cv2
import numpy as np

cap = cv2.VideoCapture("faixa.mp4")

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
while True:
    ret, frame = cap.read()
    if ret == False: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Desenhar um retângulo em cada frame, para analisar o estado da área delimitada
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    color = (0, 255, 0)
    texto_estado = 'Estado: sem deteccao de movimento'

    #especificando os pontos delimitadores da área
    area_pts = np.array([[280, 210], [380, 210], [570, 400], [40, 400]])

    #Imagem auxiliar que atuará como o detector de movimentos
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1) #deixar a área delimitada em branco e o resto preto
    imagem_area = cv2.bitwise_and(gray, gray, mask=imAux) #area delimitada em cinza

    #Aplicar subtração de fundo
    fgmsk = fgbg.apply(imagem_area)
    fgmsk = cv2.morphologyEx(fgmsk, cv2.MORPH_OPEN, kernel)
    fgmsk = cv2.dilate(fgmsk, None, iterations=2)

    #encontrar os contornos de fgmask, para filtrar somente os objetos em movimento
    cnts = cv2.findContours(fgmsk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            x, y, h, w = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            texto_estado = 'Estado: Movimento detectado'
            color = (0, 0, 255)

    #visualizando o redor da área
    cv2.drawContours(frame, [area_pts], -1, color, 2)

    #Visualizando o estado de detecção de movimento
    cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmsk)

    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows