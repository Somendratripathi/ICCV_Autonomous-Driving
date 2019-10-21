from torchvision import models
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet

from autonomous_driving.dataset import Drive360Loader
from autonomous_driving.utils import add_results, get_device
from autonomous_driving.config import *


class AllCNNDrivingModel(nn.Module):
    """
    BasicDrivingModel (Initial Course staff model)
     + BatchNormalization
    """
    def __init__(self, device=get_device()):
        super(AllCNNDrivingModel, self).__init__()

        self.device = device
        final_concat_size = 0

        # Feature extractor
        cnn = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(cnn.children())[:-1])

        intermediate_fc_out = 256
        self.intermediate = nn.Sequential(
            nn.Linear(cnn.fc.in_features, intermediate_fc_out),
            nn.BatchNorm1d(intermediate_fc_out),
            nn.ReLU()
        )

        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(2*intermediate_fc_out, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Speed Regressor
        self.control_speed = nn.Sequential(
            nn.Linear(2*intermediate_fc_out, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        module_outputs = []
        cnn_i = []
        # Loop through temporal sequence of
        # front facing camera images and pass
        # through the cnn.
        for k, v in data['cameraFront'].items():
            x = self.features(v)
            x = x.view(x.size(0), -1)
            x = self.intermediate(x)
            cnn_i.append(x)

        # Concatenate current image CNN output
        # and LSTM output.
        x_cat = torch.cat(cnn_i, dim=-1)

        # Feed concatenated outputs into the
        # regression networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction

class BatchNormDrivingModel(nn.Module):
    """
    BasicDrivingModel (Initial Course staff model)
     + BatchNormalization
    """
    def __init__(self, device=get_device()):
        super(BatchNormDrivingModel, self).__init__()

        self.device = device
        final_concat_size = 0

        # Feature extractor
        cnn = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(cnn.children())[:-1])

        intermediate_fc_out = 256
        self.intermediate = nn.Sequential(
            nn.Linear(cnn.fc.in_features, intermediate_fc_out),
            nn.BatchNorm1d(intermediate_fc_out),
            nn.ReLU()
        )
        final_concat_size += intermediate_fc_out

        hidden_size = int(intermediate_fc_out/2)  # 128
        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=intermediate_fc_out,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False
        )
        final_concat_size += hidden_size

        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Speed Regressor
        self.control_speed = nn.Sequential(
            nn.Linear(final_concat_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
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
        # regression networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction



class BasicDrivingModel(nn.Module):
    """
    A rip off SomeDrivingModel
    - Only difference is that
    """
    def __init__(self, device=get_device()):
        super(BasicDrivingModel, self).__init__()

        self.device = device
        final_concat_size = 0

        # Main CNN
        # cnn = EfficientNet.from_pretrained("efficientnet-b0").to(device)
        # self.features = cnn.extract_features
        # self.intermediate = nn.Sequential(
        #     nn.Linear(cnn._fc.in_features*50, 128),
        #     nn.ReLU()
        # )
        cnn = models.resnet34(pretrained=True)
        # for parameters in cnn.parameters():  # We don't want to train feature extracter
        #     parameters.requires_grad = False
        self.features = nn.Sequential(*list(cnn.children())[:-1])
        self.intermediate = nn.Sequential(nn.Linear(
            cnn.fc.in_features, 128),
            nn.ReLU())
        final_concat_size += 128

        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=3,
            batch_first=False
        )
        final_concat_size += 64

        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(128, 64),
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
        # regression networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(module_outputs[0])),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction
