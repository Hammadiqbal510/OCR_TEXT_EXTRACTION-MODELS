# OCR_TEXT_EXTRACTION-MODELS
TEXT RECOGNATION AND EXTRACTION WITH KERAS,TERNSORFLOW,CRNN,EASYOCR,pytesseract



# Optical Character Recognition models

In these models images will be provided and model will recognise the text and try to predict the exact output.
we have two different models in our repositry
## Model 1
 In this model we used the python library of pytesseract and it helps to predict the text from different id card images.
## Model 2
in this model we have used EasyOcr technique to recognise text in the Nationa identity card image.

## Model 3
 In this model we used the python library of CRNN and it helps to predict the text from different CAPTHA IMAGES AND TRAIN MODELS.


## Model 4
 In this model we used the python library of TENSORFLOW and it helps to predict the text from CAPTHA IMAGES.


## Model 5

 In this model we used the python library of KERAS and it helps to predict the text from CAPTHA IMAGES.




## Authors

- [HAMMAD IQBAL](https://github.com/Hammadiqbal510/OCR_TEXT_EXTRACTION-MODELS)



## 🛠 Skills
- Deep learning
- Python
- OCR



## Installation

You have to install Python first then run the command mentioned bellow.

```bash
  git clone -----
  cd my-project
```
Now you should install requirements.txt file to install dependencies

```bash  
  pip install -r requirements.txt
```

1.     OCR_CRNN :

import glob


import os 

import string


from pathlib import Path

from PIL import Image


import numpy as np


import torch


import torch.nn as nn

import torch.nn.functional as F


from torch.utils.data import DataLoader, Dataset


from torchvision import transforms


from torchvision.models import resnet18


import matplotlib.pyplot as plt


import collections


from IPython.display import clear_output





2.      OCR_TENSORFLOW:


import os


import cv2


import numpy as np 


import pandas as pd


import matplotlib.pyplot as plt


from pathlib import Path


from collections import Counter


from sklearn.model_selection import train_test_split


import tensorflow as tf


from tensorflow import keras


from tensorflow.keras import layers



3.   OCR_WITH_EASY:

!pip install easyocr


!npx degit JaidedAI/EasyOCR/examples -f


import PIL


from PIL import ImageDraw


import easyocr





4.   OCR_WITH_KERAS:


import keras_ocr




!pip install -q keras-ocr




5.   OCR_WITH_PYTESSERACT:


import os


import cv2


import shutil


import random


import pytesseract


import numpy as np


import matplotlib.pyplot as plt


!apt install tesseract-ocr


!pip install pytesseract




