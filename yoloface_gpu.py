import argparse

from yolo.yolo import YOLO, detect_video, detect_img


#####################################################################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model-weights/YOLO_Face.h5',
                        help='Dir Modelos')
    parser.add_argument('--anchors', type=str, default='cfg/yolo_anchors.txt',
                        help='Definimos el ancho')
    parser.add_argument('--classes', type=str, default='cfg/face_classes.txt',
                        help='Definicion clases')
    parser.add_argument('--score', type=float, default=0.5,
                        help='Umbral de puntuacion')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='Umbral iou')
    parser.add_argument('--img-size', type=list, action='store',
                        default=(416, 416), help='Tamaño de la imagen')
    parser.add_argument('--image', default=False, action="store_true",
                        help='Deteccionde imagen')
    parser.add_argument('--video', type=str, default='samples/subway.mp4',
                        help='Dir video')
    parser.add_argument('--output', type=str, default='outputs/',
                        help='image/video dir salida')
    args = parser.parse_args()
    return args


def _main():
    # Obtenemos argumentos
    args = get_args()

    if args.image:
        # Detectamos imagenes
        print('[i] ==> Image deteccion\n')
        detect_img(YOLO(args))
    else:
        print('[i] ==> Video deteccion\n')
        # Detectamos videos
        detect_video(YOLO(args), args.video, args.output)

    print('FIN!!!')


if __name__ == "__main__":
    _main()
