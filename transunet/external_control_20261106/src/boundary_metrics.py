"""Binary segmentation metrics on a fixed image grid; no physical spacing assumed."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def segmentation_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Expected matching two-dimensional masks")
    union = int((prediction | target).sum())
    intersection = int((prediction & target).sum())
    empty_prediction = not prediction.any()
    empty_target = not target.any()
    result = {
        "iou": float(intersection / union) if union else 1.0,
        "target_fraction": float(target.mean()),
        "empty_prediction": bool(empty_prediction),
        "empty_target": bool(empty_target),
    }
    if empty_prediction or empty_target:
        distance = 0.0 if empty_prediction and empty_target else float(
            np.hypot(prediction.shape[0] - 1, prediction.shape[1] - 1)
        )
        result.update(hd95_px=distance, assd_px=distance,
                      boundary_empty_penalty=bool(empty_prediction != empty_target))
        return result
    structure = generate_binary_structure(2, 1)
    predicted_border = prediction ^ binary_erosion(prediction, structure=structure, border_value=0)
    target_border = target ^ binary_erosion(target, structure=structure, border_value=0)
    distances = np.concatenate((distance_transform_edt(~target_border)[predicted_border],
                                distance_transform_edt(~predicted_border)[target_border]))
    result.update(hd95_px=float(np.percentile(distances, 95)),
                  assd_px=float(distances.mean()), boundary_empty_penalty=False)
    return result
