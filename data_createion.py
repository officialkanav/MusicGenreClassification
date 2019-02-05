#import dependencies
!pip install image_slicer
import supportlib.gettingdata as getdata
import librosa as lib
import cv2
import numpy as np
import os
import pandas as pd
import image_slicer

#downloading data
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
def spec_create(in_path,out_path):
    x,sr = lib.load(in_path,sr=44100,mono=True)
    b = librosa.power_to_db(librosa.feature.melspectrogram(x,sr,n_fft=1024))
    b = 255*(b-b.min())/np.ptp(b)
    cv2.imwrite(out_path,b)

import os
main_dir = os.listdir('./fma_small')

for i in tqdm(main_dir):
    i = os.path.join('./fma_small',i)
    sub_dir = os.listdir(i)
    for j in sub_dir:
        path = os.path.join(i,j)
        j = list(j)
        j = j[:-4]
        j = "".join(j)
        j = int(j)
        j = str(j)
        j = j+'.jpg'
        output = os.path.join('./fma_mel',j)
        try:
            spec_create(path,output)
        except:
            print('Corrupt file {}'.format(path))
def split_spectograms(image,n_splits):
    length_spectogram = len(image[0,:])
    n_outputs = int(length_spectogram/n_splits)
    image1 = list()
    for i in range(n_splits):
        image2 = image[:,0:n_outputs]
        image = np.delete(image,np.s_[:n_outputs],axis = 1)
        image1.append(image2)
    return image1
