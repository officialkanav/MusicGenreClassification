import pandas as pd

class Utils:
    def __init__(self):        
        pass        
    
    def create_label(self, csv_path):
        self.track = pd.read_csv(csv_path)
        self.label = {}
        y = self.track.iloc[:,1:2].values
        ID = self.track.iloc[:,0:1].values
        y = y[2:,:]
        ID = ID[2:,:]
    
        for i,j in zip(ID,y):
          self.label[str(i[0])] = j[0]
        
        return self.label
          
    def create_encoder_decoder(self):
        self.encoder = {}
        self.decoder = {}
        
        self.encoder['Hip-Hop'] = 0
        self.encoder['Pop'] = 1
        self.encoder['Rock'] = 2
        self.encoder['Experimental'] = 3
        self.encoder['Folk'] = 4
        self.encoder['Instrumental'] = 5
        self.encoder['Electronic'] = 6
        self.encoder['International'] = 7
    
        self.decoder[0] ='Hip-Hop' 
        self.decoder[1] = 'Pop'
        self.decoder[2] = 'Rock'
        self.decoder[3] = 'Experimental'
        self.decoder[4] = 'Folk'
        self.decoder[5] = 'Instrumental'
        self.decoder[6] = 'Electronic'
        self.decoder[7] = 'International'
        
        return self.encoder, self.decoder