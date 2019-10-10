import numpy as np
import json
from torchvision import models
import torch.nn as nn
import torch
import pandas as pd
import torch.optim as optim
from tqdm import tqdm

from autonomous_driving.dataset import Drive360Loader


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
            cnn.fc.in_features, 128),
            nn.ReLU())
        final_concat_size += 128

        # Main LSTM
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=64,
                            num_layers=3,
                            batch_first=False)
        final_concat_size += 64

        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
            x = self.features(v)
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


def add_results(results, output):
    steering = np.squeeze(output['canSteering'].cpu().data.numpy())
    speed = np.squeeze(output['canSpeed'].cpu().data.numpy())
    if normalize_targets:
        steering = (steering * target_std['canSteering']) + target_mean['canSteering']
        speed = (speed * target_std['canSpeed']) + target_mean['canSpeed']
    if np.isscalar(steering):
        steering = [steering]
    if np.isscalar(speed):
        speed = [speed]
    results['canSteering'].extend(steering)
    results['canSpeed'].extend(speed)


if __name__ == "__main__":
    config = json.load(open('./config.json'))

    train_loader = Drive360Loader(config, 'train')
    validation_loader = Drive360Loader(config, 'validation')
    test_loader = Drive360Loader(config, 'test')

    print('Loaded train loader with the following data available as a dict.')
    print(train_loader.drive360.dataframe.keys())

    model = SomeDrivingModel()

    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    num_epochs = 1

    best_speed_model_wts = model.state_dict()
    best_speed_mse = float("inf")
    best_angle_model_wts = model.state_dict()
    best_angle_mse = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss_speed = 0.0
        running_loss_angle = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            prediction = model(data)

            loss_speed = criterion(prediction['canSpeed'], target['canSpeed'])
            running_loss_speed += loss_speed.item()

            loss_angle = criterion(prediction['canSteering'], target['canSteering'])
            running_loss_angle += loss_angle.item()

            loss = loss_speed + loss_angle
            loss.backward()
            optimizer.step()

            if batch_idx % 2 == 1:
                print('[Epoch: %d, batch:  %5d] loss speed: %.5f loss angle: %.5f' %
                      (epoch + 1, batch_idx + 1, running_loss_speed / 2.0, running_loss_angle / 2.0))
                running_loss_speed = 0.0
                running_loss_angle = 0.0

            if batch_idx >= 20:
                break

        val_pred_speed = np.array((2,))
        val_target_speed = np.array((2,))
        val_pred_angle = np.array((2,))
        val_target_angle = np.array((2,))
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validation_loader):
                outputs = model(data)
                val_pred_speed = np.concatenate((val_pred_speed, outputs["canSpeed"].cpu().detach().numpy()), axis=0)
                val_target_speed = np.concatenate((val_target_speed, target["canSpeed"].cpu().detach().numpy()), axis=0)
                val_pred_angle = np.concatenate((val_pred_angle, outputs["canSteering"].cpu().detach().numpy()), axis=0)
                val_target_angle = np.concatenate((val_target_angle, target["canSteering"].cpu().detach().numpy()), axis=0)

                if batch_idx >= 20:
                    break

        val_speed_mse = ((val_target_speed - val_pred_speed) ** 2).mean()
        val_angle_mse = ((val_target_angle - val_pred_angle) ** 2).mean()
        print('Epoch: ' + str(epoch + 1) + '/' + str(num_epochs) + ' Val MSE Speed: ' + str(
            round(val_speed_mse, 3)) + " Val MSE Angle: " + str(round(val_angle_mse, 3)))

        if best_speed_mse > val_speed_mse:
            best_speed_mse = val_speed_mse
            best_speed_model_wts = model.state_dict()
            torch.save(model, "first_model_speed.pt")

        if best_angle_mse > val_angle_mse:
            best_angle_mse = val_angle_mse
            best_angle_model_wts = model.state_dict()
            torch.save(model, "first_model_angle.pt")

    print("-"*10)
    print("Training Complete")

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            prediction = model(data)
            mse = (np.square(prediction['canSpeed'] -
                             target['canSpeed'])).mean()
            print(mse)
            if batch_idx >= 5:
                break

    # Creating a submission file.
    normalize_targets = config['target']['normalize']
    target_mean = config['target']['mean']
    target_std = config['target']['std']

    file = './submission.csv'
    results = {
        "chapter": [],
        "frameIndex": [],
        "canSteering": [],
        "canSpeed": [],
    }

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            prediction = model(data)
            add_results(results, prediction)
            # Used to terminate early, remove.
             if batch_idx >= 5:
                 break

    df = pd.DataFrame.from_dict(results)
    df.to_csv(file, index=False)
