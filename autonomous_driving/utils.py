import numpy as np
import torch


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def add_results(results, output, ids, normalize_targets, target_mean, target_std):
    chapter = np.squeeze(ids["chapter"].cpu().data.numpy())
    frame_index = np.squeeze(ids["frameIndex"].cpu().data.numpy())
    steering = np.squeeze(output['canSteering'].cpu().data.numpy())
    speed = np.squeeze(output['canSpeed'].cpu().data.numpy())
    if normalize_targets:
        steering = (steering * target_std['canSteering']) + target_mean['canSteering']
        speed = (speed * target_std['canSpeed']) + target_mean['canSpeed']

    if np.ndim(chapter) == 0:
        chapter = [chapter]
    if np.ndim(frame_index) == 0:
        frame_index = [frame_index]
    if np.ndim(steering) == 0:
        steering = [steering]
    if np.ndim(speed) == 0:
        speed = [speed]

    results["chapter"].extend(chapter)
    results["frameIndex"].extend(frame_index)
    results['canSteering'].extend(steering)
    results['canSpeed'].extend(speed)
