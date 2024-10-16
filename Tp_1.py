### PDI : Trabajo Práctico N°1 ###
### Tec. Universitaria en Inteligencia Artificial ###
### 2024 ###

### Integrantes ###
# López Ceratto, Julieta : L-3311/1
# Dimenna, Valentin      : D-43366/4  
# Onega, Miranda PIlar   : O-1779/5
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
    #plt.title(title)
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



############################################################################################
######################### PROBLEMA 2: CORRECCIÓN DE MULTIPLE CHOICE ########################
############################################################################################

#Carga de paths
paths_img = ['examen_1.png', 'examen_2.png', 'examen_3.png', 'examen_4.png', 'examen_5.png']
paths_img = ['./src/'+i for i in paths_img]
paths_img[0]
examen = cv2.imread(paths_img[1], cv2.IMREAD_GRAYSCALE)
#imshow(examen)


################################# Encabezado #########################################

def obtener_renglon_de_datos(examen):
    """
    Devuelve la imagen del renglón de los campos a analizar
    """
    img = cv2.imread(examen, cv2.IMREAD_GRAYSCALE)
    umbral, umbralizada = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    img_neg = umbralizada==0  #True -> blanco, False --> negro

    img_row_zeros = img_neg.any(axis=1)
    x = np.diff(img_row_zeros)
    renglones_indxs = np.argwhere(x)  # me devuelve donde empieza y termina el renglon, me interesa la pos 2 y 3
    renglon_de_datos = [renglones_indxs[2], renglones_indxs[3]]
    # Genero imagen para pasar como argumento a la otra que analiza el texto
    recorte_renglon = img[renglon_de_datos[0][0]:renglon_de_datos[1][0], :]
    return recorte_renglon

def obtener_datos_de_campos(imagen):
    """ 
    Funcion que devuelve una lista con las imagenes de los campos completados
    """
    campos = imagen
    
    _, umbral = cv2.threshold(campos, 220, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    campos = []

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c) 
        
        if w > 77:
          # chequeo que la posiscion de x no sea 569 porque coincide con el ancho del campo de codigo
          if x == 569:
            continue

          campos.append((x, y, w, h))

    # Genero imagenes para pasar como argumento a la otra que analiza los caracteres:
    indv_datos_del_examen=[]
    campos_a_retornar=imagen.copy()
    for x, y, w, h in campos:
      indv_datos_del_examen.append(campos_a_retornar[y+3:y+h-3, x+3:x+w-3]) # Agrego los recortes de los campos, el +3, -3 para descartar los borde
    
    return indv_datos_del_examen

def contar_componentes(campos):
    """
    Función que cuenta los caracteres de mi imagen
    """
    componentes={}
    con = 0
    
    for imagen in campos:
      ret, thresh = cv2.threshold(imagen, 127, 255, 0)

      #cv2 Componets detecta los blancos como porciones de componentes --> hay que invertir los bits 
      img = cv2.bitwise_not(thresh)     
      output = cv2.connectedComponentsWithStats(img)
      caracteres = output[0]-1
        
      stats = output[2]
      sort_index = np.argsort(stats[:, 0])
      stats = stats[sort_index]
      
      # Descartar las componentes de ancho pequeño
      for i in range(len(stats)):
        if i >= 1:
          anchura = stats[i][2]
          if anchura <= 2:
             caracteres = caracteres -1

      espacios =  []
      for i in range(len(stats)):
        if i > 1: # para calcular la diferencia con el anterior
          val_espacio = stats[i][0]-(stats[i-1][0]) # calculo la diferencia entre la cordenada x de mi componente siguiente y la anterior
          if val_espacio > 9 and  i > 2: # > 2 Es para descartar el vector de mi primer componente. Porque las masyusculas tienden a ser mas anchas y no corresponden a espacios
            espacios.append(val_espacio)  
       
      clave = f"campo_{con}"
      componentes[clave] = (caracteres, len(espacios))
      con = con + 1

    return componentes

def validar_caracteres(componentes):

  for val, keys in componentes.items():
    n_caracteres = keys[0]
    espacios = keys[1]

    if val == "campo_1":
       if n_caracteres == 1:
          print("CODE:OK")
       else:
          print("CODE: MAL")  
       
    if val == "campo_2" or val == "campo_0": 
       if n_caracteres == 8:
          if val == "campo_0": 
            print("DATE:OK")
          else:
            print("ID:OK")
       else:
          if val == "campo_0": 
            print("DATE:MAL")
          else: 
            print("ID: MAL")  

    if val == "campo_3":
       if n_caracteres > 1 and  n_caracteres <= 25 and espacios == 1:
          print("NAME:OK")
       else:
          print("NAME: MAL")       
   
def obtener_campo_nombre(examen):
    '''Función que evuelve los crop de los campos name'''
    renglon = obtener_renglon_de_datos(examen)
    # Como sé que el ultimo campo es el nombre, me quedo con ese
    campos_datos = obtener_datos_de_campos(renglon)
    name = campos_datos[3]
    #plt.figure(), plt.imshow(renglon, cmap='gray'),  plt.show(block=True)
    return name

################################# Recorte de respuestas ##############################



############### Detección de Respuestas y corrección de preguntas ####################
#Recorto manual respuestas para hacer este punto
x1 = [20,20,20,20,20,324,324,324,324,324 ]
x2 = [258,258,258,258,258, 562,562,562,562,562]
y1 = [56,182,312,435, 561,56,182,312,435, 561 ]
y2 = [172,298, 421 ,555,675, 172,298, 421 , 555,675 ]

def recortar_preguntas():
    preguntas = []
    for i in range(len(x1)):
        pregunta = examen[y1[i]:y2[i], x1[i]:x2[i]]
        preguntas.append(pregunta)
    return preguntas

preguntas = recortar_preguntas()

# for pregunta in preguntas:
    #imshow(pregunta)

def binarize(img: np.array) -> np.array:
    ###
    #img: imagen en escala de grises
    #img_bin : imagen binarizada
    ###
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return img_bin

def lineas_horizontales(img_bin: np.array ) -> tuple:
    ###
    # img_bin : imagen binarizada.
    # rect_mas_ancho : linea de respuesta del examen.
    ###

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
    #plt.figure(figsize=(10, 6))
    #plt.imshow(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
    #plt.title("Contornos Encontrados y Rectángulo Más Ancho")
    # plt.axis('off')  # Opcional: Oculta los ejes
    # plt.show()

    return rect_mas_ancho

def recortar_pregunta(pregunta: np.array, linea_preg: tuple) -> np.array:
    ###
    # pregunta: imagen escala de grises de pregunta del examen.
    # linea_preg : linea horizontal de la pregunta del exament.
    # sub_preguna: área de respuesta de la pregunta.
    ###

    #obtengo coordenadas de la linea
    x1,y1,w,h = linea_preg

    #obtengo forma de la imagen de pregunta
    h, w = pregunta.shape
    
    #recorto la imagen obteniendo solo el área de respuesta
    sub_preg = pregunta[y1-14:y1-2, x1: w]

    #imshow(sub_preg)
    return sub_preg


def detectar_letra(sub_preg: np.array) -> tuple[dict, tuple]:
    ###
    # sub_pregunta : área de respuesta de la pregunta.
    #dict_contornos: diccionario clave contorno padre, valor
    # contornos hijos.
    # contornos: total de contornos encontrados.
    ###

    # Binarizo la imagen
    sub_bin = binarize(sub_preg)
    
    # Encuentro contornos y jerarquía
    contornos, jerarquia = cv2.findContours(sub_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    
    # Crea una imagen para dibujar los contornos
    img_contornos = cv2.cvtColor(sub_bin, cv2.COLOR_GRAY2BGR)
    
    # Inicializo el diccionario para guardar contornos padres e hijos
    dict_contornos = {}
    padres = 0
    hijos = 0

    # Dibujo contornos con diferentes colores
    for i in range(len(contornos)):

        # Obtener información de jerarquía
        _, _, first_child, parent = jerarquia[0][i]
        
        if parent == -1:  # Contorno padre
            padres += 1
            color = (0, 0, 255)  # Rojo para contornos padres
            dict_contornos[i] = []  # Inicializo la lista de hijos para este padre
        else:  
            # Contorno hijo
            hijos += 1
            color = (0, 255, 0)  # Verde para contornos hijos
            
            # Agregar este contorno hijo a la lista del contorno padre
            if parent in dict_contornos:
                dict_contornos[parent].append(i)

        cv2.drawContours(img_contornos, contornos, i, color, 2)  # Dibujar contornos con el color apropiado

    # Mostrar la imagen con contornos dibujados
    # print(f' padres: {padres}, hijos : {hijos}')
    #plt.figure(figsize=(10, 6))
    # plt.imsow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
    #plt.title("Contornos de Letras Detectados")
    # plt.axis('off')
    # plt.show()

    return dict_contornos, contornos

def clasificar_letra(dict_contornos: dict, contornos: tuple) -> list:
    ###
    #dict_contornos: diccionario clave contorno padre, valor
    # contornos hijos.
    # contornos: total de contornos encontrados.
    # respuestas: letras encontradas; si no encuentra letras devuelve
    # lista vacía.
    ###

    respuestas = []
    if len(dict_contornos) == 0: ### si no se detecto una letra se devuelve la lista vacia
        return respuestas
    else:
        for key, value in dict_contornos.items():
            if cv2.contourArea(contornos[key]) > 2: ## Filtro áreas menores a 2
                
                if len(value) == 0:
                    #Si el no tiene hijos entonces la letra es C
                    respuestas.append('C')
                
                elif len(value) == 2:
                    #Si tiene 2 hijos, la letra es B
                    respuestas.append('B')


                elif  len(value) == 1:
                    #Si tiene 1 hijo, hay que obtener la relación de área hijo/padre
                    # para definir si es letra A o D

                    area_hijo = cv2.contourArea(contornos[value[0]])
                    area_padre = cv2.contourArea(contornos[key])
                    proporcion = area_hijo / area_padre

                    if proporcion < 0.8:
                        #Si el hijo no abarca más del 80% que el padre
                        # es la letra A
                        respuestas.append('A')

                    else:
                        #De no ser así, es la letra D
                        respuestas.append('D')

    return respuestas
                
def corregir_pregunta(respuesta: list, i: int, respuestas_correctas: list) -> str:
    ###
    # respuesta: letras encontradas en la pregunta.
    # i : número de pregunta.
    # respuestas_correctas: lista con las respuestas correctas
    # del examen.
    ###

    if len(respuesta) == 0:    #Si no hay ninguna letra
        return 'No hay respuesta'

    elif len(respuesta) > 1: # si hay más de una letra
        return 'Contesto mas de una letra'
    
    else: #Si hay una letra

        if respuesta[0] == respuestas_correctas[i]:  #Si la letra es correcta

            return 'OK'

        else:               #Si la letra es incorrecta
            return 'Mal'

respuestas_correctas = ['B', 'B', 'D', 'B', 'B', 'A', 'B', 'D', 'D', 'D']

def corregir_examen(examen: np.array)-> None:
    ###
    # examen : imagen escala de grises del examen
    # completo
    ###
    nota = 0
    for i in range(len(preguntas)):
        pregunta = preguntas[i]
        preg_bin = binarize(pregunta)                                       #Binarizo pregunta
        linea_preg = lineas_horizontales(preg_bin)                          #Encuentro linea de respuesta.
        sub_preg = recortar_pregunta(pregunta,linea_preg)                   #Recorto área de respuesta
        dic_letras, cont = detectar_letra(sub_preg)                         #Encuento letra o vacío
        respuestas = clasificar_letra(dic_letras, cont)                     #Identifico qué letra es
        correccion = corregir_pregunta(respuestas, i, respuestas_correctas) #Corrección con respuestas correctas.
        
        #Devuelvo correción de la pregunta
        if correccion == 'OK':
            nota += 1
            print(f'Pregunta {i+1}: {correccion}')
        else:
            print(f'Pregunta {i+1}: MAL')
    
    #Imprimo nota del examen
    print(f'La nota es {nota}')
    

corregir_examen(examen)



img = cv2.imread('TP/examen_1.png',cv2.IMREAD_GRAYSCALE) 
cv2.imshow('GrayScale Image', img)

_, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Thresholded Image', img_bin)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
sub_imagenes=[]
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    if w > 50 and h > 50: 
        sub_image = img[y:y+h, x:x+w]
        
        # Guardar en archivo las subimagenes
        # cv2.imwrite(f'question_{i+1}.png', sub_image)
        sub_imagenes.append(sub_image)
        cv2.imshow(f'Question {i+1}', sub_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

subimage = sub_imagenes[1]

_, binary = cv2.threshold(subimage, 150, 255, cv2.THRESH_BINARY_INV)

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (75, 1))


horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)


contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

# Muestra las Lineas horizontales detectadas
# subimage_with_lines = cv2.cvtColor(subimage, cv2.COLOR_GRAY2BGR)
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(subimage_with_lines, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow('Detected Horizontal Lines', subimage_with_lines)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

prev_y = 0
question_images = []

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    if i > 0:
        question_img = subimage[prev_y:y, :]
        # question_images.append(question_img)
        
        # Guardar las subimagenes en archivos 
        # cv2.imwrite(f'question_{i}.png', question_img)
        question_images.append(question_img)
        cv2.imshow(f'Question {i}', question_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    prev_y = y
