# OCR_TEXT_EXTRACTION-MODELS
TEXT RECOGNATION AND EXTRACTION WITH KERAS,TERNSORFLOW,CRNN,EASYOCR,pytesseract


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




