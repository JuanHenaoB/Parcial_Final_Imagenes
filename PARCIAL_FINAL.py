# Pontificia Universidad Javeriana. Departamento de Electrónica
# Author: Juan Henao, Estudiante de Ing. Electrónica.
# Procesamiento de Imagenes y video

# Importar librerias
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from math import sqrt
from hough import *
from orientation_estimate import *
from Clase_Bandera import *
import os

if __name__ == '__main__':
    print('Hello user, please enter the name of the flag you want to process')
    flag = input('You got 5 options flag1.png, flag2.png, ... flag5.png : ')
    path = 'C:/Users/ACER/Desktop/Semestre10/Imagenes/Presentaciones/Img_PFinal'
    path_name = os.path.join(path, flag)

    flag_obj = Clase_Bandera(path_name)
    n_colores = flag_obj.Colores()
    P_Colores = flag_obj.Porcentaje()
    line = flag_obj.orientacion()

    print('1. The number of colors in the flag you selected is: ', n_colores)
    print('2. Color percentage in the flag you selected are :', P_Colores)
    print('The lines present in the flag you selected are :', line)






