from data_generator import datagen
from utils import Utils
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

class DL:
    def __init__(self, csv_path, data_path):
        self.csv_path = csv_path
        self.data_path = data_path
        utils = Utils()
        self.label = utils.create_label(self.csv_path)
        
    def create_dataloader(self):
        datagen_object = datagen(self.label,self.data_path)
        test_size = 0.2
        valid_size = 0.1
        data_len = len(datagen_object)
        indices = list(range(data_len))

        split1 = int(np.floor(valid_size * data_len))
        split2 = int(np.floor(test_size * data_len))
        
        valid_idx , test_idx, train_idx = indices[:split1], indices[split1:split2] , indices[split2:] 
        
        train_sampler = SubsetRandomSampler(train_idx)   
        valid_sampler = SubsetRandomSampler(valid_idx)   
        test_sampler = SubsetRandomSampler(test_idx)
        
        train_loader = DataLoader(datagen_object, batch_size=64 , sampler=train_sampler)   
        valid_loader = DataLoader(datagen_object, batch_size=64, sampler=valid_sampler)   
        test_loader = DataLoader(datagen_object, batch_size=64, sampler=test_sampler)
        
        dataloader = {'train':train_loader,'val':valid_loader}
        
        return dataloader, test_loader