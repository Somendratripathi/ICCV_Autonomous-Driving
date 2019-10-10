import numpy as np


def add_results(results, output, ids, normalize_targets, target_std, target_mean):
    chapter = np.squeeze(ids["chapter"].cpu().data.numpy())
    frame_index = np.squeeze(ids["frameIndex"].cpu().data.numpy())
    steering = np.squeeze(output['canSteering'].cpu().data.numpy())
    speed = np.squeeze(output['canSpeed'].cpu().data.numpy())
    if normalize_targets:
        steering = (steering * target_std['canSteering']) + target_mean['canSteering']
        speed = (speed * target_std['canSpeed']) + target_mean['canSpeed']

    if np.isscalar(chapter):
        chapter = [chapter]
    if np.isscalar(frame_index):
        frame_index = [frame_index]
    if np.isscalar(steering):
        steering = [steering]
    if np.isscalar(speed):
        speed = [speed]

    results["chapter"].extend(chapter)
    results["frameIndex"].extend(frame_index)
    results['canSteering'].extend(steering)
    results['canSpeed'].extend(speed)

