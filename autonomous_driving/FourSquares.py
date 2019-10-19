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
from dataset4sq import Drive360Loader
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
        cnn = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(cnn.children())[:-1])
        self.intermediate = nn.Sequential(nn.Linear(
            cnn.fc.in_features, 256),
            nn.ReLU())
        final_concat_size += 256

        # Main LSTM
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=128,
                            num_layers=5,
                            batch_first=False)
        final_concat_size += 128

        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Speed Regressor
        self.control_speed = nn.Sequential(
            nn.Linear(final_concat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        module_outputs = []
        lstm_i = []
        # Loop through temporal sequence of
        # front facing camera images and pass 
        # through the cnn.
        for k, v in data['cameraFront'].items():
            x = self.features(v.cuda())
            x = x.view(x.size(0), -1)
            x = self.intermediate(x)
            lstm_i.append(x)
            # feed the current front facing camera
            # output directly into the 
            # regression networks.
            if k == 0:
                module_outputs.append(x)
        
        # Feed temporal outputs of CNN into LSTM
        i_lstm, _ = self.lstm(torch.stack(lstm_i))
        module_outputs.append(i_lstm[-1])

        # Concatenate current image CNN output 
        # and LSTM output.
        x_cat = torch.cat(module_outputs, dim=-1)

        # Feed concatenated outputs into the 
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction







if __name__ == "__main__":
    config = json.load(open(os.path.join(os.getcwd(),'drive/My Drive/L2D','autonomous_driving/config.json')))
    #config = json.load(open(os.path.join(os.getcwd(),'autonomous_driving/config.json')))

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

                loss = loss_speed + loss_angle
                loss.backward()
                optimizer.step()


           
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


            #calculate batch MSE
            val_speed_mse = ((denorm(np.array(val_target_speed),speed_mean,speed_std) - denorm(np.array(val_pred_speed),speed_mean,speed_std)) ** 2).mean()
            val_angle_mse = ((denorm(np.array(val_target_angle),steer_mean,steer_std) - denorm(np.array(val_pred_angle),steer_mean,steer_std)) ** 2).mean()


            hist_val_acc_s.append(val_speed_mse)
            hist_val_acc_a.append(val_angle_mse)
            
            #print val batch MSE
            print('Epoch: ' + str(epoch+1) + '/' + str(epochs) + '\n Val Speed MSE: {0} \n Val Angle MSE: {1}'.format(str(round(val_speed_mse,3)),str(round(val_angle_mse,3))))

        torch.save(model, "Foursq.pt")
        return (hist_train_loss_s, hist_train_loss_a, hist_val_acc_s, hist_val_acc_a)



    # optimizer

    adm_optimizer = optim.Adamax(model.parameters(),lr=0.002, betas=(0.8, 0.8)) #optim.SGD(model.parameters(), lr)
    criterion = nn.MSELoss().cuda()

    run1 = train_nn(train_loader, validation_loader, model, adm_optimizer, criterion)
    
    
    #model output
    pickle.dump(run1,open("model_4sq","wb"))

