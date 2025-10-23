##########################
# Esté código esta basado en tutoriales y manuales de
# sus librerías.
#
# El objetivo de este código es ser una plantilla 
# para experimentar diferentes operaciones vectoriales, 
# con el fin de identificar algunos gestos de las manos
# que permitan controlar una aplicación estilo "Paint" 
#########################

# La rutina `gestos` requiere más conocimientos de álgebra
# lineal que de Python.
def calcula_gestos(puntos):
    # `puntos` es una matriz de 21 x 3
    #          cada columna corresponde a la posición
    #          de un 'landmark' en 3 dimensiones. 
    #          Los 'landmarks' se definen en
    #          https://google.github.io/mediapipe/solutions/hands#hand-landmark-model 
    gestos = dict() # se define loa variable `gestos` como un diccionario
                    # que se irá llenando con cada gesto encontrado

    # Antes de encontrar los gestos 
    # encontramos el promedio de algunos puntos claves para algunos gestos.
    # `puntos[:,i]` significa el vector columna `i` de la matriz `puntos`
    prom_muneca = (puntos[:,0] + puntos[:,1] + puntos[:,2])/3
    prom_MCP = (puntos[:,5] + puntos[:,9] + puntos[:,13] + puntos[:,17])/4
    prom_TIP = (puntos[:,8] + puntos[:,12] + puntos[:,16] + puntos[:,20])/4

    # encontramos la distancia entre algunos promedios
    dist_mun_TIP = np.linalg.norm(prom_muneca - prom_TIP)
    #dist_MCP_TIP = np.linalg.norm(prom_MCP - prom_TIP)
    dist_mun_MCP = np.linalg.norm(prom_muneca - prom_MCP)

    # Usando los promedios se definen algunos gestos.
    # Cada gesto tiene un número real aproximadamente entre 0 y 1.
    # A mayor valor, mejor es el gesto.
    gestos['mano_abierta'] = dist_mun_TIP / (2 * dist_mun_MCP)
    gestos['mano_cerrada'] = 1 - dist_mun_TIP / (2 * dist_mun_MCP)

    return gestos



# El código que sigue requiere conocimientos básicos de Python 

# Lo primero es importar las librerías a usar
import time
import numpy as np # Librería para el manejo de arreglos y matrices.
import cv2  # Libreria para procesamiento de imágenes.
import mediapipe as mp # Librería para procesar imágenes de personas.
import pygame as pg # Está librería se usará para desarrollar
                    # una aplicación estilo 'Paint'.


# Atajos a la librería `mediapipe`
mpHands = mp.solutions.hands
mp_dedos = mpHands.HandLandmark
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#cap = cv2.VideoCapture(0) # Captura un 'stream' con la cámara del computador.
cap = cv2.VideoCapture('./videos/hand.mp4')  # Captura un 'stream' a partir de un archivo de video.
cap_s= int(cap.get(4)), int(cap.get(3)) # Obtiene las dimensiones del 'stream'.
print('Dimensiones del "stream"',cap_s)
pTime=time.time() # Variable utilizada para medir los fps.
screen = pg.display.set_mode(cap_s,pg.FULLSCREEN) # Pantalla de nuestro 'Paint'.
pg_w, pg_h = pg.display.get_surface().get_size() # Obtiene las dimensiones de la pantalla
lienzo = pg.Surface((pg_w, pg_h)) # Lienzo para nuestro 'Paint'.

# variable que almacena los puntos de la mano (landmark) 
varios_pun=[]

# el siguiente ciclo itera sobre los 'frames'
repetir=True
while repetir:
    gesto={} # se inicializa la variable que almacena gestos
    gesto['mano_abierta'] = 0
    gesto['mano_cerrada'] = 0

    screen.fill((0, 0, 0)) # Color de fondo del 'Paint'

    success, img = cap.read() # Lee una imagen del 'stream'
                              # en un arreglo numpy

    img =cv2.flip(img, 1) # Voltea la imagen como espejo

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cambia el orden de los colores 
    
    results =hands.process(imgRGB) # realiza el procesamiento con mediapipe
                                   # para ubicar las manos en la imagen

    if results.multi_hand_landmarks: # si se encontraron manos ...
        for handLms in results.multi_hand_landmarks: # para cada mano ...
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # dibuja la mano
            pos=[] # variable usada para almacenar los puntos de la mano
            for punto in handLms.landmark: # itera sobre cada punto (landmark)
                pos.append([punto.x,punto.y,punto.z]) # almacena  cada punto de la mano

            # almacena hasta 10  manos en el tiempo para reducir ruido
            # y obtiene la mediana de los datos
            varios_pun.append(pos) 
            if len(varios_pun)>10: 
                varios_pun=varios_pun[-10:]
            varios_pun_np=np.median(np.array(varios_pun),axis=0).T

            gesto = calcula_gestos(varios_pun_np) 

            # calcula la posición del punto para pygame (pg) y para OpenCv (cv) del dedo índice
            pg_pos_punto=(int(varios_pun_np[0,8]*pg_w),int(varios_pun_np[1,8]*pg_h))
            cv_pos_punto=(int(varios_pun_np[0,8]*cap_s[1]),int(varios_pun_np[1,8]*cap_s[0]))
            pg.draw.circle(screen, ( 0, 255, 0), pg_pos_punto, 3)
            cv2.circle(img,cv_pos_punto, 5, (0,255,255), -1)
            
            # Dibuja círculos si la mano está abierta 
            if gesto['mano_abierta'] > 0.9:
                    cv2.circle(img,cv_pos_punto, 5, (0,0,255), -1)
                    pg_w, pg_h = pg.display.get_surface().get_size()
                    pg.draw.circle(lienzo, ( 0, 255, 255), pg_pos_punto, 2) 

            # Borra si la mano está cerrada
            if gesto['mano_cerrada'] > 0.6:
                   cv2.circle(img,cv_pos_punto, 5, (0,255,0), -1)
                   pg_pos_punto=(int(varios_pun_np[0,8]*pg_w),int(varios_pun_np[1,8]*pg_h))
                   pg.draw.circle(lienzo, (  0,   0, 0), pg_pos_punto, 20)
    
    # Calcula los fps y los escribe en la pantalla.
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(gesto['mano_abierta']),(100,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),2) 
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),2)

    # Evalúa concidiones para finalizar el programa
    if cv2.waitKey(5) & 0xFF == 27: #esc
      break
    for event in pg.event.get():
        if event.type == pg.QUIT:
            print('quit')
            repetir=False

    # Actualiza las ventanas de pygame y OpenCv
    screen.blit(lienzo,(0,0),special_flags=pg.BLEND_MAX)
    pg.display.flip()
    cv2.imshow("Image", img)
    cv2.waitKey(1)