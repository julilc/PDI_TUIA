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

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

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
    
img_ec = ecualizacion_hist(img, M = 50, N = 50)
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
paths_img = ['examen_1.png', 'examen_2.png', 'examen_3.png', 'examen_4.png', 'examen_5.png']
paths_img = ['./src/'+i for i in paths_img]
paths_img[0]
examen = cv2.imread(paths_img[1], cv2.IMREAD_GRAYSCALE)
plt.figure() , plt.imshow(examen, cmap = 'gray'), plt.show()

### Corrección de pregunta
x1 = 20
x2 = 250
y1 = 55
y2 = 163
pregunta = examen[y1:y2,x1:x2]
plt.figure() , plt.imshow(pregunta, cmap = 'gray'), plt.show()

def encontrar_lineas(img, bordes, r, t, tr):
    ####
    # Img : imagen 
    # bordes : canny de la imagen
    # r : rhoa
    # t: theta
    # tr : treshold
    ####

    lineas = cv2.HoughLines(bordes, r,t,tr)

    img_color = img.copy()
    img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)

    if lineas is not None:
        for linea in lineas:
            rho, theta = linea[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img_color, (x1,y1), (x2,y2), (0,0,255), 1)

    imshow(img_color)
    return lineas

def filtra_bordes(bordes, umbral = 10):
    ###
    # bordes: lineas de la imagen
    # umbral : valor minimo de píxeles que tienen que
    # tener de distancia una linea de otra.
    ###

    bordes_filt = []

    bordes = sorted(bordes, key = lambda l:l[0][0])

    for linea in bordes:
        rho, theta = linea[0]

        if len(bordes_filt) == 0:
            bordes_filt.append(linea)
        else:
            ult_rho = bordes_filt[-1][0][0]

            if abs(rho - ult_rho) > umbral:
                bordes_filt.append(linea)
    
    return bordes_filt

def dividir_pregunta(pregunta, lineas):
    h , w = pregunta.shape
    y_max = int(lineas[0][0][0])
    sub_pregunta = pregunta[0:y_max, 0:w]
    return sub_pregunta

def componentes_conectadas_pregunta(s_preg, conectivity = 900):
    # Aplicar umbral binario para binarizar la imagen
    _, binarizada = cv2.threshold(s_preg, 150, 200, cv2.THRESH_BINARY)
    # Obtener las componentes conectadas y estadísticas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarizada, conectivity , cv2.CV_32S)

    # Crear una imagen en color para dibujar las componentes conectadas
    output_image = cv2.cvtColor(binarizada, cv2.COLOR_GRAY2BGR)
    
    # Recorremos todas las componentes conectadas (ignorando el fondo)
    for i in range(1, num_labels):
        # Obtener las estadísticas para la componente actual
        x, y, w, h, area = stats[i]
        
        # Dibujar un rectángulo alrededor de la componente conectada
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Dibujar el centroide de la componente conectada
        cx, cy = centroids[i]
        cv2.circle(output_image, (int(cx), int(cy)), 1, (255, 0, 0), -1)
        
        # Escribir el número de la componente conectada cerca del centroide
        #cv2.putText(output_image, f'Componente {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Retornar la imagen con las componentes conectadas dibujadas
    imshow(output_image)



pregunta_canny = cv2.Canny(pregunta, threshold1=50, threshold2=100)
lineas_h = encontrar_lineas(pregunta, pregunta_canny, 1, np.pi / 2, tr = 110)
lineas_h = filtra_bordes(lineas_h, umbral = 10)
sub_preg = dividir_pregunta(pregunta, lineas_h)
componentes_conectadas_pregunta(sub_preg,  1)
pregunta_canny.shape
imshow(pregunta)
_, binarizada = cv2.threshold(sub_preg, 150, 200, cv2.THRESH_BINARY)
imshow(binarizada)

