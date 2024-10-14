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
imshow(examen)
x1 = [20,20,20,20,20,326,326,326,326,326 ]
x2 = [255,255,255,255,255, 560,560,560,560,560]
y1 = [59,183,312,440, 564,59,183,312,440, 564 ]
y2 = [172,296, 421 ,550,671, 172,296, 421 , 550,671 ]

def recortar_preguntas():
    preguntas = []
    for i in range(len(x1)):
        pregunta = examen[y1[i]:y2[i], x1[i]:x2[i]]
        preguntas.append(pregunta)
    return preguntas

preguntas = recortar_preguntas()

for pregunta in preguntas:
    imshow(pregunta)

def binarize(img):
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return img_bin

def lineas_horizontales(img_bin):
    # Convertir la imagen en color
    img_s = cv2.cvtColor(img_bin.copy(), cv2.COLOR_GRAY2BGR)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar variables para el rectángulo más ancho
    max_width = 0
    rect_mas_ancho = None

    for i in range(len(contours)):
        contour = contours[i]
        # Obtener el rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contour)
        
        # Considerar solo contornos que tengan un cierto alto para evitar ruido
        if h > 3:  # Ajusta el umbral según sea necesario
            continue
        
        # Comparamos el ancho
        if w > max_width:  
            max_width = w
            rect_mas_ancho = (x, y, w, h)  # Guardamos el rectángulo más ancho

    # Dibujar el rectángulo más ancho si existe
    if rect_mas_ancho is not None:
        x, y, w, h = rect_mas_ancho
        cv2.rectangle(img_s, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Rectángulo en azul
    
    # Dibujar los contornos en la imagen
    cv2.drawContours(img_s, contours, -1, (0, 255, 0), 1)  # Contornos en verde

    # Mostrar la imagen con contornos y el rectángulo más ancho
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
    plt.title("Contornos Encontrados y Rectángulo Más Ancho")
    plt.axis('off')  # Opcional: Oculta los ejes
    plt.show()

    return rect_mas_ancho

def recortar_pregunta(pregunta, linea_preg):
    x1,y1,w,h = linea_preg
    h, w = pregunta.shape
    sub_preg = pregunta[y1-14:y1-2, x1: w]
    imshow(sub_preg)
    return sub_preg

def detectar_letra(sub_preg):
    # Binarizar la imagen
    sub_bin = binarize(sub_preg)
    
    # Encontrar contornos y jerarquía
    contornos, jerarquia = cv2.findContours(sub_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear una imagen en blanco para dibujar los contornos
    img_contornos = cv2.cvtColor(sub_bin, cv2.COLOR_GRAY2BGR)
    
    # Inicializar el diccionario para guardar contornos padres e hijos
    dict_contornos = {}
    padres = 0
    hijos = 0
    # Dibujar contornos con diferentes colores
    for i in range(len(contornos)):
        # Obtener información de jerarquía
        _, _, first_child, parent = jerarquia[0][i]
        
        if parent == -1:  # Contorno padre
            padres += 1
            color = (0, 0, 255)  # Rojo para contornos padres
            dict_contornos[i] = []  # Inicializar la lista de hijos para este padre
        else:  # Contorno hijo
            hijos += 1
            color = (0, 255, 0)  # Verde para contornos hijos
            
            # Agregar este contorno hijo a la lista del contorno padre
            if parent in dict_contornos:
                dict_contornos[parent].append(i)

        cv2.drawContours(img_contornos, contornos, i, color, 2)  # Dibujar contornos con el color apropiado

    # Mostrar la imagen con contornos dibujados
    print(f' padres: {padres}, hijos : {hijos}')
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
    plt.title("Contornos de Letras Detectados")
    plt.axis('off')
    plt.show()

    return dict_contornos, contornos

def clasificar_letra(dict_contornos, contornos):
    respuestas = []
    if len(dict_contornos) == 0:
        return respuestas
    else:
        for key, value in dict_contornos.items():
            if cv2.contourArea(contornos[key]) > 2:
                if len(value) == 0:
                    respuestas.append('C')
                elif len(value) == 2:
                    respuestas.append('B')
                elif  len(value) == 1:
                    area_hijo = cv2.contourArea(contornos[value[0]])
                    area_padre = cv2.contourArea(contornos[key])
                    proporcion = area_hijo / area_padre
                    if proporcion < 0.8:
                        respuestas.append('A')
                    else:
                        respuestas.append('D')
    return respuestas
                
def corregir_examen(respuesta, i, respuestas_correctas):
    if len(respuesta) == 0:
        return 'No hay respuesta'
    elif len(respuesta) > 1:
        return 'Contesto mas de una letra'
    else:
        if respuesta[0] == respuestas_correctas[i]:
            return 'Bien'
        else:
            return 'Mal'

respuestas_corrctas = ['B', 'B', 'D', 'B', 'B', 'A', 'B', 'D', 'D']

for i in range(len(preguntas)):
    pregunta = preguntas[i]
    preg_bin = binarize(pregunta)
    linea_preg = lineas_horizontales(preg_bin)
    sub_preg = recortar_pregunta(pregunta,linea_preg)
    dic_letras, cont = detectar_letra(sub_preg)
    respuestas = clasificar_letra(dic_letras, cont)
    print(respuestas)
    corregir_examen(respuestas, i, respuestas_corrctas)


