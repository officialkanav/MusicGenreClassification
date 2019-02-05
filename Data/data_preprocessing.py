import librosa as lib
import cv2
import numpy as np
import os
from tqdm import tqdm
import librosa.display

class Dataset:
    def __init__(self,song_folder,new_folder):
        self.song_folder = song_folder
        self.new_folder = new_folder
        
    def chop_image(self, img, out_path):
      height, width, channels = img.shape
      w = width // 10
      for i in range(0,10):
        temp = img[:, (i*w):((i+1)*w), :]
        temp = cv2.resize(temp,(128,128))
        #cv2.imwrite(os.path.join(self.new_folder,('test_{}.png'.format(i),temp)))
        cv2.imwrite((out_path+'test_{}.png'.format(i)), temp)
    
    def spec_create(self,in_path,out_path):
        x,sr = lib.load(in_path,sr=44100,mono=True)
        img = librosa.power_to_db(librosa.feature.melspectrogram(x,sr,n_fft=1024))
        img = 255*(img-img.min())/np.ptp(img)
        self.chop_image(img, out_path)
    
    def create_data(self):
        for i in tqdm(self.song_folder):
            folder = os.path.join(self.song_folder,i)
            sub_folder = os.listdir(folder)
            for track in sub_folder:
                path = os.path.join(folder,track)
                track = list(track)
                track = track[:-4]
                track = "".join(track)
                track = int(track)
                track = str(track)
                #track = track + '.jpg'
                output = os.path.join(self.new_folder,track)
                try:
                    self.spec_create(path,output)
                except:
                    print('Corrupt file {}'.format(path))
