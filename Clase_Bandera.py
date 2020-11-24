# Pontificia Universidad Javeriana. Departamento de Electrónica
# Author: Juan Henao, Estudiante de Ing. Electrónica.
# Procesamiento de Imagenes y video

# Importar Librerias
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from math import sqrt
from hough import *
from orientation_estimate import *

# Esta función calcula las distancias entre pixeles de un cluster a su centro y las acumula
# La utilizo en el metodo colors para ayuddarme a decidir el N de colores de la bandera.
def CalcDistance(centers, labels, rows, cols, image, n):
    distSum = np.zeros(n, dtype=np.float64) # aux accumulative variable
    label_idx = int(0) # index aux variable
    for i in range(rows): #For every pixel in a given image
        for j in range(cols):
            aux = labels[label_idx] # closest cluster to the pixel
            x1 = centers[aux,0] # take every x,y,z (RGB component) of the center of the cluster
            y1 = centers[aux,1]
            z1 = centers[aux,2]
            x2 = image[i,j,0] #take every x,y,z (again in RGB space) of the given pixel.
            y2 = image[i,j,1]
            z2 = image[i,j,2]
            # Calculate distance between pixel and cluster center
            dist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            distSum[aux] = distSum[aux] + dist # Accumulate (sum) every distance for each of n clusters
            label_idx = label_idx + 1 # next pixel

    TotalSum = np.sum(distSum) # sum all of the n cluster distances
    return TotalSum

class Clase_Bandera:

    def __init__(self, path):
        Img = cv2.imread(path)
        self.Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

    def Colores(self):
        # Cambiar Img a arreglo np flotante
        image = self.Img.copy()
        image = np.array(image, dtype=np.float64) / 255
        rows, cols, ch = image.shape
        image_array = np.reshape(image, (rows * cols, ch))

        # take a sample of the image pixels
        image_array_sample = shuffle(image_array, random_state=0)[:int(rows * cols * 0.05)]

        n = int(0) #n clusters
        distance = np.zeros(4)
        for i in range (4):
            n = n + 1
            model = KMeans(n_clusters=n, random_state=0).fit(image_array_sample)  # train model
            labels = model.predict(image_array)  # get model labels
            centers = model.cluster_centers_  # get model center(s)
            distance[i] = CalcDistance(centers, labels, rows, cols, image, n)  # calculate distances

        n_colors = int(distance.argmin()) # Halla el ARGUMENTO del valor min
        n_colors = n_colors + 1

        return n_colors

    def Porcentaje(self):
        image = self.Img.copy()
        rows, cols, ch = image.shape
        Red = np.zeros(4)
        Green = np.zeros(4)
        Blue = np.zeros(4)

        for i in range (3):
            hist = cv2.calcHist([image], [i], None, [4], [0, 256], accumulate=False)
            if i == 0:
                Red[0] = hist[0]
                Red[1] = hist[1]
                Red[2] = hist[2]
                Red[3] = hist[3]
            elif i == 1:
                Green[0] = hist[0]
                Green[1] = hist[1]
                Green[2] = hist[2]
                Green[3] = hist[3]
            else :
                Blue[0] = hist[0]
                Blue[1] = hist[1]
                Blue[2] = hist[2]
                Blue[3] = hist[3]


        C1 = Red[0] + Green[0] + Blue[0]
        C2 = Red[1] + Green[1] + Blue[1]
        C3 = Red[2] + Green[2] + Blue[2]
        C4 = Red[3] + Green[3] + Blue[3]
        Total = C1 + C2 + C3 + C4

        C1 = (C1 * 100) / Total
        C2 = (C2 * 100) / Total
        C3 = (C3 * 100) / Total
        C4 = (C4 * 100) / Total

        CT = [C1, C2, C3, C4]
        # Para las dos banderas que tienen negro los porcentajes van a salir mal porque
        # no identifica el negro como color, pero creo que no me alcanza el tiempo para solucionarlo. :(
        return CT

    def orientacion(self):
        image = self.Img.copy()

        high_thresh = 300
        bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)

        hh = hough(bw_edges)

        accumulator = hh.standard_HT()

        acc_thresh = 50
        N_peaks = 11
        nhood = [25, 9]
        peaks = hh.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        [_, cols] = image.shape[:2]
        vertical = False
        vertical1 = False
        horizontal = False
        horizontal1 = False
        V = False
        H = False
        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = hh.theta[peaks[i][1]]

            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hh.center_x
            y0 = b * rho + hh.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

            if np.abs(theta_) < 80:
                vertical = True
            elif np.abs(theta_) > 100:
                vertical1 = True
            else:
                if theta_ > 0:
                    horizontal = True
                else:
                    horizontal1 = True

        if vertical or vertical1 == True:
            V = True
        if horizontal or horizontal1 == True:
            H = True
        if V and H == True:
            lineas = 'mixta'
        elif V == True:
            lineas = 'Vertical'
        elif H == True:
            lineas = 'Horizontal'

        return lineas























