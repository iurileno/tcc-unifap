import cv2
import numpy as np
from matplotlib import pyplot as plt

imagem = cv2.imread("imagem.jpg")

filtro = np.ones((9, 9), np.uint8)

imagem_escala_de_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

limiarizacao = cv2.adaptiveThreshold(imagem_escala_de_cinza, 255,
               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

erosao = cv2.erode(limiarizacao, filtro, iterations=1)
dilatacao = cv2.dilate(limiarizacao, filtro, iterations=1)
abertura_morfologica = cv2.morphologyEx(limiarizacao, cv2.MORPH_OPEN, filtro)

#histograma = cv2.calcHist([imagem], [0], None, [256], [0, 256])
histograma_cinza = cv2.calcHist([imagem_escala_de_cinza], [0], None, [256], [0, 256])
#histograma_limiarizacao = cv2.calcHist([limiarizacao], [0], None, [256], [0, 256])
#plt.plot(histograma)
plt.plot(histograma_cinza)
#plt.plot(histograma_limiarizacao)
plt.show()

cv2.imshow("Imagem original", imagem)
cv2.imshow("Imagem em escala de cinza", imagem_escala_de_cinza)
cv2.imshow("Imagem com erosao", erosao)
cv2.imshow("Imagem com dilatacao", dilatacao)
cv2.imshow("Gaussiano adaptativo", limiarizacao)
cv2.imshow("Abertura Morfologica", abertura_morfologica)


if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

