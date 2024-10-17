### PDI : Trabajo Práctico N°1 ###
### Tec. Universitaria en Inteligencia Artificial ###
### 2024 ###

### Integrantes ###
# López Ceratto, Julieta : L-3311/1
# Dimenna, Valentin      : D-43366/4  
# Onega, Miranda PIlar   : O-1779/5
###
############################################################################################
######################### PROBLEMA 2: CORRECCIÓN DE MULTIPLE CHOICE ########################
############################################################################################

#Importar .rqst
from rqst import *

#Carga de paths
paths_img = ['examen_1.png', 'examen_2.png', 'examen_3.png', 'examen_4.png', 'examen_5.png']
paths_img = ['./src/'+i for i in paths_img]
paths_img[0]
examen = cv2.imread(paths_img[1], cv2.IMREAD_GRAYSCALE)
imshow(examen)


################################# Encabezado #########################################
def binarize(img: np.array) -> np.array:
    ###
    #img: imagen en escala de grises
    #img_bin : imagen binarizada
    ###
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return img_bin


def obtener_campos(examen: np.array):
    '''
    Devuelve 3 imágenes de los campos a analizar.
    examen : imagen en escala de grises del examen;
    nombre: imagen con el nombre;
    fecha: imagen con la fecha;
    clase : imagen con la clase;
    '''
    min_ancho = 70

    # Binarizo el examen
    w, h = examen.shape
    encabezado = examen[0:46, 0:w]
    #imshow(encabezado)
    # Binarización
    encabezado_bin = binarize(encabezado)

    # Convertir la imagen a color para visualizar contornos
    encabezado_sub = cv2.cvtColor(encabezado_bin.copy(), cv2.COLOR_GRAY2BGR)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(encabezado_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lineas_campos = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Solo agregar si el ancho es mayor al umbral `min_ancho`
        if w > min_ancho:
            lineas_campos.append((x, y, w, h))
            # Dibujar rectángulo alrededor del contorno
            cv2.rectangle(encabezado_sub, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Dibujar todos los contornos
    cv2.drawContours(encabezado_sub, contours, -1, (0, 255, 0), 1)

    # Mostrar la imagen con contornos y rectángulos
    # imshow(encabezado_sub)
    lineas_ordenadas = sorted(lineas_campos, key=lambda campo: campo[0])
    # print(lineas_campos)

    linea_nombre = lineas_ordenadas[0]
    x,y,w,h = linea_nombre
    nombre = encabezado[y-20:y+h-2,x:x+w]
    linea_fecha = lineas_ordenadas[1]
    x,y,w,h = linea_fecha
    fecha = encabezado[y-20:y+h-2,x:x+w]
    linea_clase = lineas_ordenadas[2]
    x,y,w,h = linea_clase
    clase = encabezado[y-20:y+h-2,x:x+w]

    # imshow(nombre)
    # imshow(fecha)
    # imshow(clase)

    return nombre, fecha, clase


# def obtener_datos_de_campos(campo: np.array, tipo: str):
#     ###
#     #Corrige los campos segun el tipo de campo
#     #campo: imagen del campo.
#     #tipo: tipo de campo del que se trata (nombre, fecha, clase)
#     ###
#     if tipo == 'nombre':
#         ### Encontrar si hay 2 palabras
#         if palabras == 2:
#             ### Contar cantidad de letras
#             if letras =< 25:
#                 return f'Nombre: OK'
#             else:
#                 return f'Nombre : Mal'
#         else:
#             return f'Nombre: Mal'
    
#     elif tipo == 'fecha':
#         ### Contar si la cantidad de caracteres son 8
#         if caracteres == 8:
#             return f'Fecha: OK'
#         else:
#             return f'Fecha : Mal'
    
#     elif tipo == 'clase':
#         ### Contar si tieen 1 solo caracter
#         if caracter == 1:
#             return f'Clase : OK'
#         else:
#             return f'Clase : Mal'
        
# obtener_datos_de_campos(nombre, 'a')


# def contar_componentes(campos):
#     """
#     Función que cuenta los caracteres de mi imagen
#     """
#     componentes={}
#     con = 0
    
#     for imagen in campos:
#       ret, thresh = cv2.threshold(imagen, 127, 255, 0)

#       #cv2 Componets detecta los blancos como porciones de componentes --> hay que invertir los bits 
#       img = cv2.bitwise_not(thresh)     
#       output = cv2.connectedComponentsWithStats(img)
#       caracteres = output[0]-1
        
#       stats = output[2]
#       sort_index = np.argsort(stats[:, 0])
#       stats = stats[sort_index]
      
#       # Descartar las componentes de ancho pequeño
#       for i in range(len(stats)):
#         if i >= 1:
#           anchura = stats[i][2]
#           if anchura <= 2:
#              caracteres = caracteres -1

#       espacios =  []
#       for i in range(len(stats)):
#         if i > 1: # para calcular la diferencia con el anterior
#           val_espacio = stats[i][0]-(stats[i-1][0]) # calculo la diferencia entre la cordenada x de mi componente siguiente y la anterior
#           if val_espacio > 9 and  i > 2: # > 2 Es para descartar el vector de mi primer componente. Porque las masyusculas tienden a ser mas anchas y no corresponden a espacios
#             espacios.append(val_espacio)  
       
#       clave = f"campo_{con}"
#       componentes[clave] = (caracteres, len(espacios))
#       con = con + 1

#     return componentes

# def validar_caracteres(componentes):

#   for val, keys in componentes.items():
#     n_caracteres = keys[0]
#     espacios = keys[1]

#     if val == "campo_1":
#        if n_caracteres == 1:
#           print("CODE:OK")
#        else:
#           print("CODE: MAL")  
       
#     if val == "campo_2" or val == "campo_0": 
#        if n_caracteres == 8:
#           if val == "campo_0": 
#             print("DATE:OK")
#           else:
#             print("ID:OK")
#        else:
#           if val == "campo_0": 
#             print("DATE:MAL")
#           else: 
#             print("ID: MAL")  

#     if val == "campo_3":
#        if n_caracteres > 1 and  n_caracteres <= 25 and espacios == 1:
#           print("NAME:OK")
#        else:
#           print("NAME: MAL")       
   
# def obtener_campo_nombre(examen):
#     '''Función que evuelve los crop de los campos name'''
#     renglon = obtener_renglon_de_datos(examen)
#     # Como sé que el ultimo campo es el nombre, me quedo con ese
#     campos_datos = obtener_datos_de_campos(renglon)
#     name = campos_datos[3]
#     #plt.figure(), plt.imshow(renglon, cmap='gray'),  plt.show(block=True)
#     return name

################################# Recorte de respuestas ##############################

def recortar_preguntas(img: np.array) -> list:
    '''
    Recorta pregunta dada una imagen del examen, devuelve una lista de imágenes de preguntas.
    img: imagen del examen;
    list: lista con una imagen por pregunta en el examen.
    '''

    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('Thresholded Image', img_bin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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
            # cv2.imshow(f'Question {i+1}', sub_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    question_images = []
    sub_imagenes = sub_imagenes[::-1]
    print(sub_imagenes)
    for i in range(len(sub_imagenes)):
        subimage = sub_imagenes[i]

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
        

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if i > 0:
                question_img = subimage[prev_y:y, :]
                # question_images.append(question_img)
                
                # Guardar las subimagenes en archivos 
                # cv2.imwrite(f'question_{i}.png', question_img)
                question_images.append(question_img)
                #cv2.imshow(f'Question {i}', question_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            
            prev_y = y
    return question_images


############### Detección de Respuestas y corrección de preguntas ####################

def lineas_horizontales(img_bin: np.array ) -> tuple:
    '''
    img_bin : imagen binarizada;
    rect_mas_ancho : linea de respuesta del examen.
    '''
    #imshow(img_bin)
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
    # plt.figure(figsize=(10, 6))
    # plt.imshow(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
    # plt.title("Contornos Encontrados y Rectángulo Más Ancho")
    # plt.axis('off')  # Opcional: Oculta los ejes
    # plt.show()

    return rect_mas_ancho

def recortar_pregunta(pregunta: np.array, linea_preg: tuple) -> np.array:
    '''
    pregunta: imagen escala de grises de pregunta del examen;
    linea_preg : linea horizontal de la pregunta del examen;
    sub_preguna: área de respuesta de la pregunta;
    '''

    #obtengo coordenadas de la linea
    x1,y1,w,h = linea_preg

    #obtengo forma de la imagen de pregunta
    h, w = pregunta.shape
    
    #recorto la imagen obteniendo solo el área de respuesta
    sub_preg = pregunta[y1-14:y1-2, x1: w]

    # imshow(sub_preg)
    return sub_preg


def detectar_letra(sub_preg: np.array) -> tuple[dict, tuple]:
    '''
    sub_pregunta : área de respuesta de la pregunta;
    dict_contornos: diccionario clave contorno padre, valor
    contornos hijos;
    contornos: total de contornos encontrados.
    '''

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
    # plt.figure(figsize=(10, 6))
    # plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
    # plt.title("Contornos de Letras Detectados")
    # plt.axis('off')
    # plt.show()

    return dict_contornos, contornos

def clasificar_letra(dict_contornos: dict, contornos: tuple) -> list:
    '''
    dict_contornos: diccionario clave contorno padre, valor
    contornos hijos;
    contornos: total de contornos encontrados;
    respuestas: letras encontradas; si no encuentra letras devuelve
    lista vacía.
    '''

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
    '''
    respuesta: letras encontradas en la pregunta;
    i : número de pregunta;
    respuestas_correctas: lista con las respuestas correcta
    del examen;
    '''

    if len(respuesta) == 0:    #Si no hay ninguna letra
        return 'No hay respuesta'

    elif len(respuesta) > 1: # si hay más de una letra
        return 'Contesto mas de una letra'
    
    else: #Si hay una letra

        if respuesta[0] == respuestas_correctas[i]:  #Si la letra es correcta

            return 'OK'

        else:               #Si la letra es incorrecta
            return 'Mal'


def corregir_examen(preguntas: list, respuestas_correctas : list)-> None:
    '''
    examen : imagen escala de grises del examen
    completo;
    respuestas_corresctas: lista de respuestas correctas
    '''
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
    return nota

#################################### Armar Imagen Resultados ##################################

def img_resultado(nombre_alumno: np.array, nota: int) -> np.array:
    '''
    Esta función arma una imagen de círculo rojo o verde
    con el mismo alto que la imagen del nombre del alumno
    según la nota recibida
    nombre_alumno : imagen nombre del alumno,
    nota : nota obtenida en el examen
    '''
    h = nombre_alumno.shape[0]
    w = 50
    if nota<5:
        color = (255,0,0)
    
    else:
        color = (0,255,0)
    # Crear una imagen en blanco (fondo blanco)
    resultado = np.ones((h, w, 3), dtype=np.uint8) * 255  # Imagen blanca (RGB 255,255,255)

    # Definir el centro y el radio del círculo
    center = (w // 2, h // 2)  # Centro del círculo
    radius = min(h, w) // 2 - 5  # Radio del círculo, con un pequeño margen

    # Dibujar el círculo verde
    color_circulo = color  # Color verde en formato BGR
    cv2.circle(resultado, center, radius, color_circulo, -1)  # -1 para llenar el círculo

    return resultado


def resultados_examenes(list_path):
    respuestas_correctas = ['C', 'B', 'D', 'B', 'B', 'A', 'B', 'D', 'D', 'D']
    cross = cv2.imread('./src/cross.jpg', cv2.IMREAD_ANYCOLOR)
    cross = cv2.cvtColor(cross, cv2.COLOR_BGR2RGB)
    tick = cv2.imread('./src/tick.png', cv2.IMREAD_ANYCOLOR)
    tick = cv2.cvtColor(tick, cv2.COLOR_BGR2RGB)
    dict_nombre_resultado = {}
    h_img_salida = 0
    w_max_nombre = 0
    
    #Corregir examen de cada alumno
    for i in range(len(list_path)):
        
        path_examen = list_path[i]
        examen = cv2.imread(path_examen,cv2.IMREAD_GRAYSCALE) 
        nombre_alumno, _, _ = obtener_campos(examen)
        h, w = nombre_alumno.shape
        if w > w_max_nombre:
            w_max_nombre = w
        h_img_salida += h+2
        preguntas = recortar_preguntas(examen)
        nota = corregir_examen(preguntas, respuestas_correctas)
        resultado = img_resultado(nombre_alumno, nota)
        nombre_alumno = cv2.cvtColor(nombre_alumno, cv2.COLOR_GRAY2BGR)

        dict_nombre_resultado[i] = (nombre_alumno,resultado)
        
    img_salida = np.ones((h_img_salida, w_max_nombre + 53, 3), dtype=np.uint8) * 255  # Imagen blanca en escala de grises

    
    # Dibujar la línea vertical en la posición deseada
    img_salida[0:h_img_salida, w_max_nombre + 2] = 0  # Línea vertical en negro
    y_inicial = 0
    l = 0
    # Colocar nombres y resultados
    for nombre_alumno, resultado in dict_nombre_resultado.values():
        h_alumno, w_alumno, _ = nombre_alumno.shape
        # Colocar la imagen del nombre
        img_salida[y_inicial:y_inicial + h_alumno, 0:w_alumno] = nombre_alumno
        # Colocar la imagen del resultado
        img_salida[y_inicial:y_inicial + h_alumno, w_max_nombre + 3:w_max_nombre + 3 + resultado.shape[1]] = resultado
        if l != len(dict_nombre_resultado)-1:
            # Dibujar la línea horizontal
            img_salida[y_inicial + h_alumno:y_inicial + h_alumno + 1, :] = 0
            y_inicial += h_alumno + 2  # Ajustar la posición para el siguiente alumno
        l += 1

    img_salida = cv2.copyMakeBorder(img_salida,1,1,1,1, cv2.BORDER_CONSTANT)
    # Mostrar la imagen resultante
    plt.figure(figsize=(10, 6))
    plt.imshow(img_salida, cmap='gray', vmin=0, vmax=255)  # Usar 'gray' para visualizar correctamente en escala de grises
    plt.axis('off')  # Ocultar los ejes
    plt.show()


resultados_examenes(paths_img)


