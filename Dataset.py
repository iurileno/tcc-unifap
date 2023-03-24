import sys
import os.path

imagens = os.listdir()
for numero, nome in enumerate(imagens):
    os.rename(nome, f'nome objeto {numero}.jpg')

