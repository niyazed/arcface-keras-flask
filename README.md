# Face Recognition using ArcFace
ArcFace Implementation using keras and Flask deploy 

## Custom Model
I haven't use any pretrained model. So, full customized model with Arc loss. Anyone can easily customize their own model.

## Model Architecture
[Image]

## Usage
- `$ python main.py` - Start the Flask app in the browser.
- `$ python face_extract.py` - Extract faces via webcam
- `$ python train.py` - Train the model
- `$ python mtcnn_detect.py` - Start face recognizer using webcam on OpenCV window

## Notebook
Check [ArcFace.ipynb](https://github.com/niyazed/arcface-keras-flask/blob/master/ArcFace.ipynb) for inference understanding.
