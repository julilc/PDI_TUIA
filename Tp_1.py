### PDI : Trabajo Práctico N°1 ###
### Tec. Universitaria en Inteligencia Artificial ###
### 2024 ###

### Integrantes ###
# López Ceratto, Julieta : L-3311/1
# 
#
###

#Librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt

############ PROBLEMA 1: ECUALIZACIÓN DEL HISTOGRAMA #######

# Carga de imagen
img_path = './src/imagen_con_detalles_escondidos.tif'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

img.shape

def ecualizacion_hist(img, M, N):
    h, w = img.shape

    # Definir bordes
    top = bottom = M // 2
    left = right = N // 2

    # Agregar bordes
    img_border = cv2.copyMakeBorder(img, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_REPLICATE)

    # Crear imagen en blanco para almacenar resultados
    img_result = img.copy()

    # Iteración por cada píxel de la imagen original
    for i in range(h):
        for j in range(w):
            # Definir los límites de la ventana
            max_i = i + M
            max_j = j + N

            # Extraer la ventana
            ventana = img_border[i:max_i-1, j:max_j-1]

            # Calcular histograma
            hist = cv2.calcHist([ventana], [0], None, [256], [0, 256])

            # Calcular la distribución acumulada (CDF)
            cdf = hist.cumsum()
            cdf_norm = (cdf - cdf.min()) * 255 // (cdf.max() - cdf.min())
            cdf_norm = cdf_norm.astype('uint8')  # Asegurarse de que sea tipo uint8

            # Obtener el valor del pixel para la cdf normalizada
            pix_norm = cdf_norm[img_border[i+top, j+right]]
            img_result[i, j] = pix_norm
        #img_result = cv2.medianBlur(img_result)

    return img_result
    
img_ec = ecualizacion_hist(img, M = 20, N = 20)
img_ec.shape
plt.figure(), plt.imshow(img, cmap = 'gray'), plt.show(block = True)

plt.figure(), plt.imshow(img_ec, cmap = 'gray'), plt.show(block = True)

# Con M y N = 3 se presenta mucho ruido
# Con M y N = 10 se presenta menos ruido
# No hay mejora en aspecto de ruido a valores mas grandes
# No se presentan cambios muy distintos en cuanto a la
# Visualizacion de los valores dentrode los cuadrados



######## PROBLEMA 2: CORRECCIÓN DE MULTIPLE CHOICE ########

#Carga de paths
paths_img = ['imagen_1.png', 'imagen_2.png', 'imagen_3.png', 'imagen_4.png', 'imagen_5.png']
paths_img = ['./src/'+i for i in paths_img]




