import torch
import json
from tqdm import tqdm
import pandas as pd
import os
from time import time


from autonomous_driving.dataset import Drive360Loader
from autonomous_driving.config import *
from autonomous_driving.utils import add_results
from autonomous_driving.basic import SomeDrivingModel

if __name__ == "__main__":
    config = json.load(open(CONFIG_FILE))
    test_loader = Drive360Loader(config, "test")
    model = torch.load(TRAINED_MODELS_DIR + "sample3-3e-angle.pt")

    # Creating a submission file.
    normalize_targets = config['target']['normalize']
    target_mean = config['target']['mean']
    target_std = config['target']['std']

    results = {
        "chapter": [],
        "frameIndex": [],
        "canSteering": [],
        "canSpeed": [],
    }

    with torch.no_grad():
        for batch_idx, (data, target, ids) in enumerate(tqdm(test_loader)):
            prediction = model(data)
            add_results(results, prediction, ids, normalize_targets, target_mean, target_std)
            # # Used to terminate early, remove.
            # if batch_idx >= 5:
            #     break

    # Assuming sampled_predictions_file only has columns ["chapter", "frameIndex", "canSteering", "canSpeed"]
    # Also frame index is integer
    sampled_predictions = pd.DataFrame.from_dict(results)
    sampled_predictions.chapter = pd.to_numeric(sampled_predictions.chapter)
    sampled_predictions.frameIndex = pd.to_numeric(sampled_predictions.frameIndex)

    test_full = pd.read_csv(DATA_DIR + "test_full.csv", usecols=["chapter", "cameraFront"])

    # # Create an frame number column in both sampled_predictions and test_full
    print("creating frameIndex on test_full")
    since = time()
    test_full["frameIndex"] = test_full.apply(lambda row: int(os.path.basename(row.cameraFront).split(".")[0][3:]), axis=1)
    test_full = test_full[["chapter", "frameIndex"]]
    print("done in ", time() - since, "\n")

    # Join test_full with sampled_predictions => left join
    print("merging test_full and sampled_predictions")
    since = time()
    tm = pd.merge(test_full, sampled_predictions, how="left", left_on=["chapter", "frameIndex"],
                  right_on=["chapter", "frameIndex"])
    print("merged in ", time() - since, "\n")

    del sampled_predictions
    del test_full

    print("interpolating values")
    since = time()
    # find first index from top where
    steering_ind = list(tm.columns).index("canSteering")
    speed_ind = list(tm.columns).index("canSpeed")
    start = 0
    for i in range(len(tm)):
        if pd.isnull(tm.iloc[i, steering_ind]):
            continue
        tm.iloc[start:i, steering_ind] = tm.iloc[i, steering_ind]
        tm.iloc[start:i, speed_ind] = tm.iloc[i, speed_ind]
        start = i + 1
    tm.iloc[start:, steering_ind] = tm.iloc[start - 1, steering_ind]
    tm.iloc[start:, speed_ind] = tm.iloc[start - 1, speed_ind]

    print("interpolated in ", time() - since, "\n")

    print("filtering >100 rows")
    since = time()
    tm = tm.loc[tm.frameIndex > 100]
    print("filtered in ", time() - since, "\n")

    tm[["canSteering", "canSpeed"]].to_csv(SUBMISSIONS_DIR + "submission_full.csv")
