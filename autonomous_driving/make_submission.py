import pandas as pd
import os
from time import time

sampling_freq = 20
# Assuming sampled_predictions_file only has columns ["chapter", "frameIndex", "canSteering", "canSpeed"]
# Also frame index is integer
sampled_predictions_file = "submission2.csv"
test_full_csv = "test_full.csv"

sampled_predictions = pd.read_csv(sampled_predictions_file)
test_full = pd.read_csv(test_full_csv, usecols=["chapter", "cameraFront"])

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

tm[["canSteering", "canSpeed"]].to_csv("submission_full.csv")
