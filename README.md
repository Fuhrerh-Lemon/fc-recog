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
