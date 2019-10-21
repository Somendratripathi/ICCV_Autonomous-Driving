import numpy as np
import json
from torchvision import models
import torch.nn as nn
import torch
import pandas as pd
import torch.optim as optim
from datetime import datetime
import os
from tqdm import tqdm

from autonomous_driving.dataset import Drive360Loader
from autonomous_driving.utils import add_results, get_device
from autonomous_driving.config import *
from autonomous_driving.models.BasicDrivingModel import *


def main():
    config = json.load(open(CONFIG_FILE))
    device = get_device()
    model_name = str(datetime.timestamp(datetime.now()))

    train_loader = Drive360Loader(config, 'train')
    validation_loader = Drive360Loader(config, 'validation')

    model = BatchNormDrivingModel()
    model = model.to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters())
    scheduler = None  # optim.lr_scheduler.StepLR(optimizer, step_size=10)
    num_epochs = 5 if not DEBUG else DEBUG_EPOCHS

    normalize_targets = config['target']['normalize']
    target_mean = config['target']['mean']
    target_std = config['target']['std']

    best_speed_mse = float("inf")
    best_angle_mse = float("inf")
    val_speed_mse_epochs = np.empty((0,))
    val_angle_mse_epochs = np.empty((0,))
    for epoch in range(num_epochs):
        if scheduler is not None:
            scheduler.step(epoch)

        ###############
        # TRAIN EPOCH
        ###############
        model.train()
        for batch_idx, (data, target, _) in enumerate(tqdm(train_loader)):
            # transfer stuff to device
            data, target = train_loader.load_batch_to_device(data, target, device)

            # get predictions
            optimizer.zero_grad()
            prediction = model(data)

            loss_angle = criterion(prediction['canSteering'], target['canSteering'])
            loss_speed = criterion(prediction['canSpeed'], target['canSpeed'])
            loss = loss_angle + loss_speed
            loss.backward()
            optimizer.step()

            if DEBUG and batch_idx >= DEBUG_BATCHES:
                break

        #################
        # EVALUATE EPOCH
        #################
        model.eval()
        val_pred_speed = np.empty((0,))
        val_target_speed = np.empty((0,))
        val_pred_angle = np.empty((0,))
        val_target_angle = np.empty((0,))
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(tqdm(validation_loader)):
                # transfer stuff to GPU
                data, target = validation_loader.load_batch_to_device(data, target, device)
                # get predictions
                outputs = model(data)

                outputs["canSpeed"] = outputs["canSpeed"].cpu().detach().numpy()
                target["canSpeed"] = target["canSpeed"].cpu().detach().numpy()
                outputs["canSteering"] = outputs["canSteering"].cpu().detach().numpy()
                target["canSteering"] = target["canSteering"].cpu().detach().numpy()
                if normalize_targets:
                    outputs["canSpeed"] = outputs["canSpeed"] * target_std["canSpeed"] + target_mean["canSpeed"]
                    target["canSpeed"] = target["canSpeed"] * target_std["canSpeed"] + target_mean["canSpeed"]
                    outputs["canSteering"] = outputs["canSteering"] * target_std["canSteering"] + target_mean[
                        "canSteering"]
                    target["canSteering"] = target["canSteering"] * target_std["canSteering"] + target_mean[
                        "canSteering"]

                # store predictions for calculating mse for entire epoch
                val_pred_speed = np.concatenate((val_pred_speed, outputs["canSpeed"]), axis=0)
                val_target_speed = np.concatenate((val_target_speed, target["canSpeed"]), axis=0)
                val_pred_angle = np.concatenate((val_pred_angle, outputs["canSteering"]), axis=0)
                val_target_angle = np.concatenate((val_target_angle, target["canSteering"]), axis=0)

                if DEBUG and batch_idx >= DEBUG_BATCHES:
                    break

        val_speed_mse = ((val_target_speed - val_pred_speed) ** 2).mean()
        val_angle_mse = ((val_target_angle - val_pred_angle) ** 2).mean()
        val_speed_mse_epochs = np.append(val_speed_mse_epochs, val_speed_mse)
        val_angle_mse_epochs = np.append(val_angle_mse_epochs, val_angle_mse)
        print('Epoch: ' + str(epoch + 1) + '/' + str(num_epochs) + ' Val MSE Speed: ' + str(
            round(val_speed_mse, 3)) + " Val MSE Angle: " + str(round(val_angle_mse, 3)))

        if not DEBUG and best_speed_mse > val_speed_mse:
            best_speed_mse = val_speed_mse
            torch.save(model, TRAINED_MODELS_DIR + model_name + "_speed.pt")

        if not DEBUG and best_angle_mse > val_angle_mse:
            best_angle_mse = val_angle_mse
            torch.save(model, TRAINED_MODELS_DIR + model_name + "_angle.pt")

    # epochs are done. Now save model details
    def add_model_details():
        def add_column(md, cn):
            if cn not in md:
                md[cn] = None

        if os.path.isfile(TRAINED_MODELS_DETAILS_CSV):
            model_details = pd.read_csv(TRAINED_MODELS_DETAILS_CSV)
        else:
            model_details = pd.DataFrame()

        index = 0 if len(model_details) == 0 else model_details.index.max() + 1

        add_column(model_details, "model_name")
        model_details.loc[index, "model_name"] = model_name

        add_column(model_details, "model_details")
        model_details.loc[index, "model_details"] = """
            BatchNorm Model(feature extractor layer and regressor layers now has batchnorms)
            Feature extractor is resnet34.Clipped and Smoothed angle across chapter
            lstm layer has 1 layer.Also increased hidden layer size from 64 to 128
            Note: Angle is taking lstm output in this model.Using random horizontal flips
            for improving turns(fingers crossed)
            Trained sample2 on sample2 images.
        """


        add_column(model_details, "train_csv")
        model_details.loc[index, "train_csv"] = config["data_loader"]["train"]["csv_name"]

        add_column(model_details, "history_length")
        model_details.loc[index, "history_length"] = config["data_loader"]["historic"]["number"]

        add_column(model_details, "training_batch_size")
        model_details.loc[index, "training_batch_size"] = config["data_loader"]["train"]["batch_size"]

        add_column(model_details, "validation_batch_size")
        model_details.loc[index, "validation_batch_size"] = config["data_loader"]["validation"]["batch_size"]

        add_column(model_details, "shuffle")
        model_details.loc[index, "shuffle"] = config["data_loader"]["train"]["shuffle"]

        add_column(model_details, "train_transformations")
        model_details.loc[index, "train_transformations"] = "None"

        add_column(model_details, "validation_transformations")
        model_details.loc[index, "validation_transformations"] = "None"

        cn = "normalization_image"
        add_column(model_details, cn)
        model_details.loc[index, cn] = json.dumps(config["image"])

        cn = "normalization_target"
        add_column(model_details, cn)
        model_details.loc[index, cn] = json.dumps(config["target"])

        cn = "criterion"
        add_column(model_details, cn)
        model_details.loc[index, cn] = "SmoothL1Loss"

        cn = "optimizer"
        add_column(model_details, cn)
        model_details.loc[index, cn] = "Adam, default params"

        cn = "num_epochs"
        add_column(model_details, cn)
        model_details.loc[index, cn] = num_epochs

        cn = "val_angle_mse_epochs"
        add_column(model_details, cn)
        model_details.loc[index, cn] = json.dumps(val_angle_mse_epochs.tolist())

        cn = "val_speed_mse_epochs"
        add_column(model_details, cn)
        model_details.loc[index, cn] = json.dumps(val_speed_mse_epochs.tolist())

        cn = "final_model_criteria"
        add_column(model_details, cn)
        model_details.loc[index, cn] = "Best validation MSE"

        if not DEBUG:
            model_details.to_csv(TRAINED_MODELS_DETAILS_CSV, index=False)

    add_model_details()


if __name__ == "__main__":
    main()
