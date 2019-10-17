import numpy as np
import json
from torchvision import models
import torch.nn as nn
import torch
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
import os
import pickle
from dataset import Drive360Loader
from utils import add_results
if torch.cuda.is_available():
    print("Using GPU")
   
class SomeDrivingModel(nn.Module):
    """
    A very basic resnet and lstm based architecture
    """
    def __init__(self):
        super(SomeDrivingModel, self).__init__()
        final_concat_size = 0

        # Main CNN
        cnn = models.resnet50(pretrained=True)
        count = 0 
        for child in cnn.children():
            count+=1
            if count < 45:
                for param in child.parameters():
                    param.requires_grad = False


        self.features = nn.Sequential(*list(cnn.children())[:-1])
        self.intermediate = nn.Sequential(nn.Linear(
            cnn.fc.in_features, 512),
            nn.ReLU())
        final_concat_size += 512

        # Main GRU
        #self.lstm = nn.GRU(input_size=256,
        #                    hidden_size=128,
        #                    num_layers=4,
        #                    batch_first=False)
        #final_concat_size += 128

        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Speed Regressor
        #self.control_speed = nn.Sequential(
        #    nn.Linear(final_concat_size, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 1)
        #)

    def forward(self, data):
        #module_outputs = []
        #lstm_i = []
        # Loop through temporal sequence of
        # front facing camera images and pass 
        # through the cnn. 


        for k, v in data['cameraFront'].items():

            
            x = self.features(v.cuda())
            x = x.view(x.size(0), -1)
            x = self.intermediate(x)
            # 
            # lstm_i.append(x)
            # feed the current front facing camera
            # output directly into the 
            # regression networks.
            #if k == 0:
            #    module_outputs.append(x)
        
        # Feed temporal outputs of CNN into LSTM
        #i_lstm, _ = self.lstm(torch.stack(lstm_i))
        #module_outputs.append(i_lstm[-1])

        # Concatenate current image CNN output 
        # and LSTM output.
        #x_cat = torch.cat(module_outputs, dim=-1)

        # Feed concatenated outputs into the 
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x)),
                      'canSpeed': torch.squeeze(self.control_angle(x))}
        return prediction



if __name__ == "__main__":
    #config = json.load(open(os.path.join(os.getcwd(),'drive/My Drive/L2D','autonomous_driving/config.json')))
    config = json.load(open(os.path.join(os.getcwd(),'autonomous_driving/config.json')))

    train_loader = Drive360Loader(config, 'train')
    validation_loader = Drive360Loader(config, 'validation')
    test_loader = Drive360Loader(config, 'test')

    print('Loaded train loader with the following data available as a dict.')
    print(train_loader.drive360.dataframe.keys())

    model = SomeDrivingModel()
    model.cuda()
    steer_mean = config['target']['mean']['canSteering']
    steer_std = config['target']['std']['canSteering']
    speed_mean = config['target']['mean']['canSpeed']
    speed_std = config['target']['std']['canSpeed']

    def denorm(array, mean, std):
        return array * std + mean 


    def train_nn(trainloader, valloader, net, optimizer, criterion, epochs = 10):

        hist_train_loss_a = []
        hist_train_loss_s = []
        hist_val_acc_a = []
        hist_val_acc_s = []
    
        for epoch in range(epochs):
            #set to training
            net.train()

            for batch_idx, (data, target, _) in enumerate(train_loader):

                optimizer.zero_grad()
                prediction = net(data)

                loss_speed = criterion(prediction['canSpeed'], target['canSpeed'].cuda())
                loss_angle = criterion(prediction['canSteering'], target['canSteering'].cuda())

                hist_train_loss_s.append(loss_speed.cpu().detach().numpy())
                hist_train_loss_a.append(loss_angle.cpu().detach().numpy())

                loss = loss_angle #+ loss_speed 
                loss.backward()
                optimizer.step()
                if batch_idx > 50:
                    break

           
            val_pred_speed = [] #np.array((2,))
            val_target_speed = [] #np.array((2,))
            val_pred_angle = [] #np.array((2,))
            val_target_angle = [] #np.array((2,))
            
            #set to eval mode
            net.eval()

            with torch.no_grad():
                for batch_idx, (data, target, _) in enumerate(validation_loader):
                    outputs = net(data)
                    val_pred_speed.extend(outputs["canSpeed"].cpu().detach().numpy())
                    val_target_speed.extend(target["canSpeed"].cpu().detach().numpy())
                    val_pred_angle.extend(outputs["canSteering"].cpu().detach().numpy())
                    val_target_angle.extend(target["canSteering"].cpu().detach().numpy())
                    if batch_idx> 500:
                        break

            #calculate batch MSE
            val_speed_mse = ((denorm(np.array(val_target_speed),speed_mean,speed_std) - denorm(np.array(val_pred_speed),speed_mean,speed_std)) ** 2).mean()
            val_angle_mse = ((denorm(np.array(val_target_angle),steer_mean,steer_std) - denorm(np.array(val_pred_angle),steer_mean,steer_std)) ** 2).mean()


            hist_val_acc_s.append(val_speed_mse)
            hist_val_acc_a.append(val_angle_mse)
            
            #print val batch MSE
            print('Epoch: ' + str(epoch+1) + '/' + str(epochs) + '\n Val Angle MSE: {}'.format(str(round(val_angle_mse,3))))

        torch.save(model, "angle_reg_flattened_.pt")
        return (hist_train_loss_s, hist_train_loss_a, hist_val_acc_s, hist_val_acc_a)



    # optimizer

    adm_optimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999)) #optim.SGD(model.parameters(), lr)
    criterion = nn.MSELoss().cuda()

    run1 = train_nn(train_loader, validation_loader, model, adm_optimizer, criterion)
    
    
    #model output
    pickle.dump(run1,open("model_angle_regressor_flattened_4sq","wb"))

