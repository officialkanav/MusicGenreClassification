from __future__ import print_function, division
import torch
import torch.nn as nn
import time
from torchvision.models import resnet50

class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model(self):
        res50 = resnet50()
        num_ftrs = res50.fc.in_features
        res50.fc = nn.Sequential(nn.Linear(num_ftrs,256),
                                   nn.ReLU(),
                                   nn.Linear(256,8),
                                   nn.LogSoftmax(dim=1)
                                   )
        return res50
    
    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=10, dataloaders):
        since = time.time()
        model.to(self.device)    
        best_acc = 0.0

        for epoch in range(num_epochs):
          print('Epoch:',epoch)
          
          for phase in ['train', 'val']:
            if phase == ' train':
                scheduler.step()
                model.train()  
            else:
                model.eval()   
                
            running_loss = 0.0
            running_corrects = 0
            total = 0
            
            for inputs, labels in dataloaders[phase]:    
                labels = labels.to(self.device)
                #labels = labels.type(torch.cuda.FloatTensor)
                inputs = inputs.view(inputs.shape[0],3,128,128)
                inputs = inputs.to(self.device)
                inputs = inputs.type(torch.cuda.FloatTensor)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase == 'train'):
    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #preds = preds.reshape(preds.size(0),-1)
                    loss = criterion(outputs, labels)
                   # print(preds)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                # statistics 
                running_loss += loss.item() * inputs.size(0)
    
                running_corrects = running_corrects + torch.sum(preds == labels.data)
                total += labels.size(0)
                
            epoch_loss = running_loss/(len(dataloaders[phase])*64)
            epoch_acc = running_corrects.double()/(len(dataloaders[phase])*64)
            #torch.save(model.state_dict(), './resnet50/genreweights-{}.h5'.format(epoch_acc))
              
            print('{} Loss: {:.4f} , acc: {:.4f}'.format(phase, epoch_loss , epoch_acc))
            
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        best_acc = epoch_acc
        print('Best val Acc: {:4f}'.format(best_acc))
        return model
