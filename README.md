# fc-recog
reconocimiento de rostros con yolov3

# Pre-requisitos
- Tensorflow
- opencv-python
- opencv-contrib-python
- Numpy
- Keras
- Matplotlib
- Pillow

# Instalación
Primero crear un env con `virtualenv -p python3.6 faceRecog`
siguiente instalamos los requerimientos:  
**Con pip**  

``` bash
pip install -r requirements.txt
```
**Con poetry**
``` bash
poetry install
```
Usando anaconda:
``` bash
conda create -n faceRecog python=3.6
# Poetry manejador de paquetes en python
conda install poetry   
# Instalamos
poetry install
## Puede usar pip, como en la parte superior MD
```
# Uso
Con python:  
**Imagen**
```bash
python yoloface.py --image samples/img.jpg --output-dir outputs/
```
**Video**
```bash
python yoloface.py --video samples/video.mp4 --output-dir outputs/
```
Con bash:
```bash
sh ./run.sh
```
Puede modificar el archivo .sh
```bash
#!/usr/bin/env bash

python yoloface.py \
    --model-cfg './cfg/yolov3-face.cfg' \
    --model-weights './model-weights/yolov3-wider_16000.weights' \
    --image './samples/001.mp4' \
    --output-dir './outputs'
```
