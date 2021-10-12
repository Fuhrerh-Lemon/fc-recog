import argparse
import sys
import os

from utils import *

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='Dir modelo')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='Dir pesos del modelo')
parser.add_argument('--image', type=str, default='',
                    help='Imagen Dir')
parser.add_argument('--video', type=str, default='',
                    help='Video Dir')
parser.add_argument('--src', type=int, default=0,
                    help='Camara')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='Dir salida')
args = parser.parse_args()

#####################################################################
# Imprimiendo argumentos
print('----- info -----')
print('[i] Configuracion archivo: ', args.model_cfg)
print('[i] Peso: ', args.model_weights)
print('[i] Dir Imagen: ', args.image)
print('[i] Dir video: ', args.video)
print('###########################################################\n')

# cVerificando dir
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

# Proporcione los archivos de configuración y peso para el modelo y cargue la red
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def _main():
    wind_name = 'face_detection usando YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    output_file = ''

    if args.image:
        if not os.path.isfile(args.image):
            print("[!] ==> Archivo imagen {} no existe".format(args.image))
            sys.exit(1)
        cap = cv2.VideoCapture(args.image)
        output_file = args.image[:-4].rsplit('/')[-1] + '_yoloface.jpg'
    elif args.video:
        if not os.path.isfile(args.video):
            print("[!] ==> Archivo video {} no existe".format(args.video))
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4].rsplit('/')[-1] + '_yoloface.avi'
    else:
        # Datos de la camara
        cap = cv2.VideoCapture(args.src)

    # Video
    if not args.image:
        video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                       cap.get(cv2.CAP_PROP_FPS), (
                                           round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:

        has_frame, frame = cap.read()

        # Detener el video
        if not has_frame:
            print('[i] ==> FIN!!!')
            print('[i] ==> Output', os.path.join(args.output_dir, output_file))
            cv2.waitKey(1000)
            break

        # 4D frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Configura la entrada a la red
        net.setInput(blob)

        # Ejecuta el pase hacia adelante para obtener la salida de las capas
        outs = net.forward(get_outputs_names(net))

        # Eliminar los cuadros delimitadores con poca confianza
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # Rostros detectados: {}'.format(len(faces)))
        print('#' * 60)

        # inicializar el conjunto de información que mostraremos en el marco
        info = [
            ('Numero de rostros encontrados', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        # Salvando direccion salida
        if args.image:
            cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
        else:
            video_writer.write(frame.astype(np.uint8))

        cv2.imshow(wind_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrumpido!')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> FIN!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
