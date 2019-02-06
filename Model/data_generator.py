from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from utils import Utils

class datagen(Dataset):
  
    def __init__(self,label,image_dir):
      self.img_dir = image_dir
      self.label = label
      self.directory = os.listdir(self.img_dir)
      utils = Utils()
      self.encoder,_ = utils.create_encoder_decoder()
    def __len__(self):
      return len(os.listdir(self.img_dir))
      
    def __getitem__(self,idx):
      file_name = self.directory[idx]
      path = os.path.join(self.img_dir,file_name)
      image = cv2.imread(path)
      image = image/255.0
      file_name = file_name[:-6]
      file_name = "".join(file_name)
      genre = self.label.get(file_name,'notfound')
      if genre == 'notfound':
        print("not found wtf")
      genre = self.encoder[genre]
      genre = np.asarray(genre)
      return  np.asarray(image),genre