from pathlib import Path
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, Patch, Polygon

from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import torch

from skimage.measure import regionprops, regionprops_table
from scipy.ndimage import gaussian_filter, laplace
from skimage.segmentation import find_boundaries

from pycocotools.coco import COCO
from cellpose import models, io
from cellpose import metrics as cp_metrics

def load_cellpose_model(model_name: str = "cpsam_v2", device: str = "cpu"):
    """
    Helper to load the model
    """

    model = models.CellposeModel(pretrained_model=model_name, device=torch.device(device))

    return model


def coco_to_instance_mask(
    coco: COCO,
    image_id: int,
    dtype=np.uint16) -> np.ndarray:
    """
    A helper to convert COCO annotation into a integer mask
    """

    img_info = coco.loadImgs([image_id])[0]
    h, w = img_info["height"], img_info["width"]

    mask = np.zeros((h, w), dtype=dtype)

    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))

    for cell, annotation in enumerate(annotations, start=1):
        cell_mask = coco.annToMask(annotation).astype(bool)

        mask[cell_mask] = cell

    
    return mask


def inference(model: torch.nn.Module,
              path_to_image: str | Path,
              path_to_mask: str | Path) -> tuple[np.ndarray]:
    """
    Inference from the model

    Provide a paths (as Path or str) to the input RGB image and to the 
    true masks image

    Returns:
        tuple[np.ndarray]: original RGB image, true masks as np.array, predicted masks as np.array
    """

    if isinstance(path_to_image, str):
        path_to_image = Path(path_to_image)
    if isinstance(path_to_mask, str):
        path_to_mask = Path(path_to_mask)

    # Read the image and masks
    image = io.imread(path_to_image)
    true_cells = io.imread(path_to_mask)

    # Compute the prediction
    predicted_cells, _, _ = model.eval(image)

    return image, true_cells, predicted_cells


def create_cell_dataframe(
    true_cells: np.ndarray,
    groups: pd.DataFrame,
    pairs: pd.DataFrame,
    assignments: pd.DataFrame,
    match_iou: float = 0.5,
    ) -> pd.DataFrame:
    """
    Every row of the ouput is for a single true cell, its shape, descriptors and the
    estimated quality of predictions
    """
    properties = ["label", "area", "bbox", "centroid", "perimeter", "eccentricity", "solidity", "extent",
                  "orientation", "axis_major_length", "axis_minor_length", "equivalent_diameter_area"]

    cells = pd.DataFrame(regionprops_table(true_cells, properties=properties))

    # some better names for certain descriptors
    cells = cells.rename(columns={
        "label": "true_id",
        "area": "pixel_area",
        "bbox-0": "min_x",
        "bbox-1": "min_y",
        "bbox-2": "max_x",
        "bbox-3": "max_y",
        "centroid-0": "centroid_x",
        "centroid-1": "centroid_y",
        "axis_major_length": "major_axis_length",
        "axis_minor_length": "minor_axis_length",
        "equivalent_diameter_area": "equivalent_diameter",
    })


    cells["aspect_ratio"] = np.divide(cells["major_axis_length"], cells["minor_axis_length"].replace(0, np.nan))
    cells["circularity"] = np.divide(4 * np.pi * cells["pixel_area"], cells["perimeter"].replace(0, np.nan) ** 2)
    cells["pseudo_radius"] = np.sqrt(cells["pixel_area"] / np.pi)

    # Seraching what graph-derived group each true cell belongs to
    group_records = []
    for group_id, group in groups.iterrows():
        for true_id in group["true_ids"]:
            group_records.append({
                "true_id": int(true_id),
                "group_id": int(group_id),
                "group_type": group["type"], 
                "associated_pred_ids": list(group["predicted_ids"]),
                "n_associated_pred": len(group["predicted_ids"]),
                "n_associated_true": len(group["true_ids"]),
            })

    group_df = pd.DataFrame(group_records, columns=["true_id", "group_id", "group_type", "associated_pred_ids", "n_associated_pred", "n_associated_true"])
    cells = cells.merge(group_df, on="true_id", how="left", validate="one_to_one")

    # Add booelan columns for convience later
    for group_name in ["matched", "split", "merged", "complex", "missed"]:
        cells[group_name] = (cells["group_type"] == group_name)

    # Try to find best match by linear sum optimization
    if not pairs.empty:
        best_pairs = pairs.sort_values("iou", ascending=False).drop_duplicates(subset="true_id")[["true_id", "predicted_id", "iou", "overlap"]]
        best_pairs = best_pairs.rename(columns={"predicted_id": "best_associated_pred_id", "iou": "best_associated_iou","overlap": "best_associated_overlap"})
        cells = cells.merge(best_pairs, on="true_id", how="left", validate="one_to_one")
    else:
        cells["best_associated_pred_id"] = pd.nan
        cells["best_associated_iou"] = np.nan
        cells["best_associated_overlap"] = np.nan

    # Also add IOU score for the best found match
    assigned = assignments.rename(columns={"predicted_id": "assigned_pred_id", "iou": "assigned_iou"})
    cells = cells.merge(assigned, on="true_id", how="left", validate="one_to_one")
    cells["is_TP"] = (cells["assigned_iou"] >= match_iou)


    cells["assigned_pred_id"] = cells["assigned_pred_id"].astype('Int64')
    cells["best_associated_pred_id"] = (cells["best_associated_pred_id"].astype('Int64'))
    cells["group_id"] = cells["group_id"].astype('Int64')
    return cells.sort_values("true_id").reset_index(drop=True)


def map_cell_instances(
    true_cells: np.ndarray,
    predicted_cells: np.ndarray,
    min_overlap: float = 0.1,
    match_iou: float = 0.5,
) -> dict:
    """
    Map ground-truth and predicted cell instances.

    true_cells, predicted_cells: integer masks
    min_overlap: minimum intersection to consider instances related
    match_iou: minimum Jaccard score to call a match

    Returns a dict
        groups: pd.DataFrame with groups 'matched', 'split', 'merged', 'complex', 'missed', 'artifact'
        pairs: pd.DataFrame with intersections for pairs
        matches: pd.DataFrame with one-to-one pairs
    """

    if true_cells.shape != predicted_cells.shape:
        raise ValueError( f"Masks must have the same shape, but {true_cells.shape} != {predicted_cells.shape}")

    if not (0 < min_overlap <= 1 and 0 < match_iou <= 1):
        raise ValueError("min_overlap and match_iou must be in (0, 1]")

    if np.any(true_cells < 0) or np.any(predicted_cells < 0):
        raise ValueError("Instance labels cannot be negative")

    # Changing the mapping:
    # true_inv and pred_inv are of the same shape as true_cells / predicted_cells
    # but now masks are enumerated by the order of their encounter
    # in a flattened array, while _ids contain the old numbers

    true_ids, true_inv = np.unique(true_cells, return_inverse=True)
    pred_ids, pred_inv = np.unique(predicted_cells, return_inverse=True)

    n_true, n_pred = len(true_ids), len(pred_ids)

    # Pixel-wise confusion matrix:
    # true_inv * n_pred + pred_inv gives every pair of (true cell, predicted cell) a unique integer code
    # bincount counts these unique combinations pixel-wise
    # minlength=n_true * n_pred assures that even empty combinations are not skipped and assigned 0 instead
    # .reshape(n_true, n_pred) makes a matrix of shape (N of true cells, N of predicted cells), where each entry
    # ij shows the number of pixels shared by true cell i and predicted cell j
    intersection = np.bincount((true_inv * n_pred + pred_inv).flatten(), minlength=n_true * n_pred).reshape(n_true, n_pred)

    # Count total areas of every true cell and every predicted cell
    true_area = intersection.sum(axis=1)
    pred_area = intersection.sum(axis=0)

    # Getting non-zero indices
    # in principle can be just np.arange(1, len(true_ids)) but just in case background is not the first entry:
    true_idx = np.flatnonzero(true_ids != 0)
    pred_idx = np.flatnonzero(pred_ids != 0)

    # Exclude background
    # np.ix_(true_idx, pred_idx) creates a mesh of non-background indices
    intersection = intersection[np.ix_(true_idx, pred_idx)]
    true_area = true_area[true_idx]
    pred_area = pred_area[pred_idx]

    # Union for Jaccard:
    # Count of true pixel + Count of predicted pixels - their intersection
    union = true_area[:, None] + pred_area[None, :] - intersection

    # Jaccard:
    # intesection / union, with defence and zero division (however, not really necessary)
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)

    # Overlap = intersection / min(true pixel count, predicted pixel count) - symmetric idea
    overlap = np.divide(intersection,
                        np.minimum(true_area[:, None], pred_area[None, :]),
                        out=np.zeros_like(intersection, dtype=float),
                        where=(true_area[:, None] > 0) & (pred_area[None, :] > 0))

    # Identify what true and predicted segments are at least somehow related:
    # 1. Have intersecting pixels
    # 2. Have at least certain level of overlap
    adjacency = (intersection > 0) & (overlap >= min_overlap)

    # Drop background from original ids
    true_ids = true_ids[true_idx]
    pred_ids = pred_ids[pred_idx]

    # Get connections between related segments
    rows, cols = np.where(adjacency)

    # Store related pairs
    pairs = pd.DataFrame([
        {
            "true_id": int(true_ids[i]),
            "predicted_id": int(pred_ids[j]),
            "intersection": int(intersection[i, j]),
            "iou": float(iou[i, j]),
            "overlap": float(overlap[i, j]),
        }
        for i, j in zip(rows, cols)
    ])


    n_true = len(true_ids)
    n_pred = len(pred_ids)

    # Create a bipartite graph
    # total number of nodes = n_true + n_pred
    graph = np.zeros((n_true + n_pred, n_true + n_pred), dtype=np.uint8)

    # Put binary adjececy matrix into zones connecting true and predicted segments
    graph[:n_true, n_true:] = adjacency
    graph[n_true:, :n_true] = adjacency.T

    # Extract connected subgraphs using scipy.sparse.csgraph routine
    # labels is an array which assigns every node its subgraph index
    n_subgraphs, labels = connected_components(csr_matrix(graph), directed=False)

    # Classify subgraphs
    groups = []

    for subgaph in np.unique(labels):


        true_members = true_ids[labels[:n_true] == subgaph].astype(int).tolist()
        pred_members = pred_ids[labels[n_true:] == subgaph].astype(int).tolist()

        n_t = len(true_members)
        n_p = len(pred_members)

        if n_t == 1 and n_p == 1:
            category = "matched"
        elif n_t == 1 and n_p > 1:
            category = "split"
        elif n_t > 1 and n_p == 1:
            category = "merged"
        elif n_t > 1 and n_p > 1:
            category = "complex"
        elif n_t == 1 and n_p == 0:
            category = "missed"
        elif n_t == 0 and n_p == 1:
            category = "artifact"
        else:
            raise RuntimeError("Unexpected subgaph structure.")

        groups.append({
            "type": category,
            "true_ids": true_members,
            "predicted_ids": pred_members,
            "n_true": n_t,
            "n_predicted": n_p,
        })

    # Store subgraphs ain a dataframe
    groups = pd.DataFrame(groups)

    # Now registering exact matches:

    # using scipy optimization linear_sum_assignment to choose the best pair matches,
    # so that it maximizes the sum of Jaccard scores, while using every row and column
    # at most once (now adjusted to match the official pipeline)
    # Below:
    #   assignments: stand for complete dataframe of assignments
    #   matches: a subframe with only assignments with enough confidence
    if n_true and n_pred:
        n_min = min(n_true, n_pred)

        costs = (
            -(iou >= match_iou).astype(float)
            - iou / (2 * n_min)
        )

        r, c = linear_sum_assignment(costs)
    else:
        r = np.array([], dtype=int)
        c = np.array([], dtype=int)

    assignments = pd.DataFrame(
        [
            {
                "true_id": int(true_ids[i]),
                "predicted_id": int(pred_ids[j]),
                "iou": float(iou[i, j]),
            }
            for i, j in zip(r, c)
        ],
        columns=["true_id", "predicted_id", "iou"]
    )

    matches = assignments[
        assignments["iou"] >= match_iou
    ].copy()

    # NOTE: related subgraphs from bipartite graph connected components also return 'matched' but for groupes that has exctaly 1 
    # true cell and one predicted cell and has at least certain overlep (usually rather small). In theory, I could have used just
    # a match threshold there and call them matches. But I would overlook situations with, e.g. 2 predicted cells related to 1 true cell
    # where one of the predictions overlap with the true one heavily, and another one - weakly. Logically, the true one and heavily overlapping prediction
    # should also be a one-to-one match, so, I use a separate optimization algorithm for matches.


    # Finally, adding dataframe of individual cells:
    cells = create_cell_dataframe(
        true_cells=true_cells,
        groups=groups,
        pairs=pairs,
        assignments=assignments,
        match_iou=match_iou
    )

    if n_true and n_pred:
        best = np.argmax(iou, axis=1)
        best_ious = pd.DataFrame({"true_id": true_ids, "best_pred_id": pred_ids[best], "best_iou": iou[np.arange(n_true), best]})
    else:
        best_ious = pd.DataFrame(columns=["true_id", "best_pred_id", "best_iou"])

    cells = cells.merge(best_ious, on="true_id", how="left")
    cells["best_iou"] = cells["best_iou"].fillna(0.0)
    
    return {
        "groups": groups,
        "pairs": pairs,
        "matches": matches,
        "cells": cells,
        "iou_raw": iou,
        "inter_raw": intersection,
        "match_iou": match_iou,
        "min_overlap": min_overlap,
    }


def cellpose_ap(iou: np.ndarray, threshold: float) -> float:
    """
    Cellpose Average Precision (from official code)
    """

    n_true, n_pred = iou.shape

    if n_true == 0 or n_pred == 0:
        tp = 0
    else:
        n_min = min(n_true, n_pred)

        cost = (-(iou >= threshold).astype(float) - iou / (2 * n_min))
        rows, cols = linear_sum_assignment(cost)
        tp = np.count_nonzero(iou[rows, cols] >= threshold)

    fp = n_pred - tp
    fn = n_true - tp

    denom = tp + fp + fn

    return tp / denom if denom else np.nan


def cellpose_aji(
    true_cells: np.ndarray,
    predicted_cells: np.ndarray,
    iou: np.ndarray,
    inter: np.ndarray,
) -> float:
    """
    Cellpose official AJI
    """

    foreground_union = np.count_nonzero((true_cells > 0) | (predicted_cells > 0))

    if foreground_union == 0:
        return np.nan

    n_true, n_pred = iou.shape

    if n_true == 0 or n_pred == 0:
        return 0.0

    n_min = min(n_true, n_pred)

    # Official threshold is 0.5
    costs = (-(iou >= 0.5).astype(float) - iou / (2 * n_min))
    r, c = linear_sum_assignment(costs)

    intersection_sum = inter[r, c].sum()

    return float(intersection_sum / foreground_union)


def aggregate_cell_results(
    true_cells: np.ndarray,
    predicted_cells: np.ndarray,
    result: dict[str, pd.DataFrame],
    image_id: str | int | None = None,
    dataset: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate cell instance segmentation results into one row.
    Returns a pd.DataFrame
    """

    groups = result["groups"]
    matches = result["matches"]

    # Unique IDs without background
    n_true = len(np.unique(true_cells)) - int(np.any(true_cells == 0))
    n_pred = len(np.unique(predicted_cells)) - int(np.any(predicted_cells == 0))

    # Conventional metrics:

    # TP, FP, FN
    tp = len(matches)
    fp = n_pred - tp
    fn = n_true - tp

    # PREC, RECALL
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan

    # F1
    f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0 if n_true + n_pred > 0 else np.nan)

    # Average precision at thesholds
    iou = result["iou_raw"]
    ap50 = cellpose_ap(iou=iou, threshold=0.5)
    ap75 = cellpose_ap(iou=iou, threshold=0.75)
    ap90 = cellpose_ap(iou=iou, threshold=0.9)

    # Boundary metrics
    scales = [0.5, 0.75, 0.9]

    ## There is a problem that official metric expects a batch of as an input, so:
    b_precision, b_recall, b_f1 = cp_metrics.boundary_scores([true_cells], [predicted_cells], scales=scales)
    b_prec_50, b_prec_75, b_prec_90 = b_precision[:, 0]
    b_rec_50, b_rec_75, b_rec_90 = b_recall[:, 0]
    b_f1_50, b_f1_75, b_f1_90 = b_f1[:, 0]


    # Average similarity measures

    # Mean Jaccard score in TP, median Jaccard score in TP, mean Dice score in TP
    if tp:
        ious = matches["iou"].to_numpy(dtype=float)
        dices = 2 * ious / (1 + ious)

        mean_iou = float(ious.mean())
        median_iou = float(np.median(ious))
        mean_dice = float(dices.mean())
    else:
        mean_iou = np.nan
        median_iou = np.nan
        mean_dice = np.nan

    # Count group categories
    categories = groups["type"].value_counts() if len(groups) else pd.Series(dtype=int)

    # Overall pixel metrics: Jaccard, Dice
    # (Just how many pixels are attributed to cell segments overall in true masks and predicted masks -
    # regardless the matches)
    true_fg = true_cells != 0
    pred_fg = predicted_cells != 0

    fg_intersection = np.count_nonzero(true_fg & pred_fg)
    fg_union = np.count_nonzero(true_fg | pred_fg)

    foreground_iou = (fg_intersection / fg_union if fg_union else np.nan)
    fg_sum = np.count_nonzero(true_fg) + np.count_nonzero(pred_fg)
    foreground_dice = (2 * fg_intersection / fg_sum if fg_sum else np.nan)

    # Aggregated Jaccard Index AJI:
    # matched instance intersections / corresponding union
    # plus the areas of all unmatched instances.
    if n_true + n_pred:

        # Make dicts of IDs -> Areas
        true_ids, true_area = np.unique(true_cells[true_cells > 0], return_counts=True)
        pred_ids, pred_area = np.unique(predicted_cells[predicted_cells > 0], return_counts=True)
        true_areas = dict(zip(true_ids, true_area))
        pred_areas = dict(zip(pred_ids, pred_area))

        matched_true = set()
        matched_pred = set()

        aji_intersection = 0
        aji_union = 0

        # Iterate over paired matches
        for row in matches.itertuples(index=False):
            t_id = row.true_id
            p_id = row.predicted_id
            iou = row.iou

            # Restoring intersection from areas and Jaccard
            # intersection = Jaccard * (area_true + area_pred) / (1 +Jaccard)
            intersection = (iou * (true_areas[t_id] + pred_areas[p_id]) / (1 + iou))

            # Restoring union
            union = true_areas[t_id] + pred_areas[p_id] - intersection

            # Summing up
            aji_intersection += intersection
            aji_union += union

            matched_true.add(t_id)
            matched_pred.add(p_id)

        # Also adding areas that do not belong to matched pairs (they are without intersections)
        aji_union += sum(area for tid, area in true_areas.items() if tid not in matched_true)
        aji_union += sum(area for pid, area in pred_areas.items() if pid not in matched_pred)

        # AJI:
        aji = aji_intersection / aji_union if aji_union else np.nan

        # Official AJI
        aji_offcial = cellpose_aji(
            true_cells=true_cells,
            predicted_cells=predicted_cells,
            iou=result['iou_raw'],
            inter=result['inter_raw']
        )
    else:
        aji = np.nan
        aji_offcial = np.nan

    def count(category):
        return int(categories.get(category, 0))

    row = {

        "image_id": image_id,
        "dataset": dataset,
        "model": model,

        "n_true": n_true,
        "n_predicted": n_pred,
        "count_error": n_pred - n_true,

        # Standard
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": ap50,
        "ap75": ap75,
        "ap90": ap90,

        # Boundary metrics
        "b_prec_50": b_prec_50,
        "b_prec_75": b_prec_75,
        "b_prec_90": b_prec_90,
        "b_rec_50": b_rec_50,
        "b_rec_75": b_rec_75,
        "b_rec_90": b_rec_90,
        "b_f1_50": b_f1_50,
        "b_f1_75": b_f1_75,
        "b_f1_90": b_f1_90,

        # Mean Jac and Dice
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "mean_dice": mean_dice,

        # Global
        "foreground_iou": foreground_iou,
        "foreground_dice": foreground_dice,
        "aji_custom": aji,
        "aji_official": aji_offcial,

        # Counts of groups
        "n_matched_groups": count("matched"),
        "n_split_groups": count("split"),
        "n_merged_groups": count("merged"),
        "n_complex_groups": count("complex"),
        "n_missed_groups": count("missed"),
        "n_artifact_groups": count("artifact"),

        # Hyperparameters
        "match_iou": result["match_iou"],
        "min_overlap": result["min_overlap"],
    }

    return pd.DataFrame([row])


def analyze_image(
    image: np.ndarray,
    true_cells: np.ndarray | None = None,
    image_id: str | int | None = None,
    dataset: str | None = None) -> pd.DataFrame:
    """
    Image descriptors

    image: actual image
    true_cells: integer mask with true segments
    image_id: image identifier
    dataset: dataset identifier

    Returns
    -------
    pd.DataFrame: df of descriptors
    """

    h, w = image.shape[:2]
    n_channels = 1 if image.ndim == 2 else image.shape[-1]


    dtype = image.dtype
    bit_depth = dtype.itemsize * 8

    row = {
        "image_id": image_id,
        "dataset": dataset,
        "height": h,
        "width": w,
        "aspect_ratio": w / h,
        "n_pixels": h * w,
        "channels": n_channels,
        "dtype": str(dtype),
        "bit_depth": bit_depth,
    }

    # Convert to grayscale
    img = image.astype(np.float32)

    if image.ndim == 2:
        gray = img
    elif n_channels == 3:

        # Standard luminance definition:
        gray = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])
    else:
        gray = img.mean(axis=-1)

    # Percentiles
    p01, p99 = np.percentile(gray, [1, 99])

    row.update({
        "intensity_mean": float(gray.mean()),
        "intensity_std": float(gray.std()),
        "intensity_p01": float(p01),
        "intensity_p99": float(p99),
        "contrast": float(p99 - p01),
    })

    # Using iinfo to unify different bit depths => gray_norm
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        gray_norm = (gray - info.min) / (info.max - info.min)
    else:
        l, h = gray.min(), gray.max()
        gray_norm = (gray - l) / (h - l) if h > l else np.zeros_like(gray)

    # variance of Laplacian
    lap = laplace(gray_norm)

    # Estimatiing noise against Gaussian blurrign
    smooth = gaussian_filter(gray_norm, sigma=1.0)
    residual = gray_norm - smooth

    row.update({
        "laplacian_variance": float(lap.var()),
        "noise_estimate": float(residual.std()),
    })

    # Count cells in the true mask (if provided)
    if true_cells is not None:
        labels, counts = np.unique(true_cells[true_cells != 0], return_counts=True)

        n_cells = len(labels)
        foreground_pixels = int(counts.sum())

        mean_area = float(counts.mean()) if n_cells else np.nan
        median_area = float(np.median(counts)) if n_cells else np.nan

        area_cv = float(counts.std() / counts.mean()) if n_cells and counts.mean() > 0 else np.nan

        row.update({
            "n_cells": n_cells,
            "cell_density": n_cells / (h * w) * 1e6,
            "foreground_fraction": foreground_pixels / (h * w),
            "mean_cell_area": mean_area,
            "median_cell_area": median_area,
            "cell_area_cv": area_cv,
        })

    return pd.DataFrame([row])


def plot_cell_results(
    image: np.ndarray,
    true_cells: np.ndarray,
    predicted_cells: np.ndarray,
    result: dict,
    summary=None,
    show_ids: bool = True,
    show_boxes: bool = True,
    alpha: float = 0.45,
    figsize: tuple = (18, 7),
):
    """
    Visualize instance segmentation quality

    Parameters
    ----------
    image : ndarray
        original image
    true_cells, predicted_cells : ndarray
        masks with background = 0.
    result : dict
        output from map_cell_instances function
    summary : pd.DataFrame or dict, optional
        aggregate_cell_results
    show_ids : bool
        show segment ids
    show_boxes : bool
        draw boxes about suspicious groups of segments
    alpha : float
        opacity
    figsize : tuple
        dimensions

    Returns
    -------
    fig, axes
    """

    if true_cells.shape != predicted_cells.shape:
        raise ValueError(f"Instance masks must match in shape, {true_cells.shape} != {predicted_cells.shape}")

    if image.shape[:2] != true_cells.shape:
        raise ValueError(f"Image and masks must natch in shape, {image.shape[:2]} != {true_cells.shape}")

    # Color code for suspicious groups
    colors = {
        "matched":  "#35A853",   # g
        "split":    "#3288DB",   # b
        "merged":   "#F39C35",   # o
        "complex":  "#D94BC8",   # m
        "missed":   "#E34B4B",   # r
        "artifact": "#8D63D2",   # p
    }

    descriptions = {
        "matched":  f"One-to-one ({len(result['groups'][result['groups']['type'] == 'matched'])})",
        "split":    f"Split ({len(result['groups'][result['groups']['type'] == 'split'])})",
        "merged":   f"Merged ({len(result['groups'][result['groups']['type'] == 'merged'])})",
        "complex":  f"Complex ({len(result['groups'][result['groups']['type'] == 'complex'])})",
        "missed":   f"Missed ({len(result['groups'][result['groups']['type'] == 'missed'])})",
        "artifact": f"Artifact ({len(result['groups'][result['groups']['type'] == 'artifact'])})",
    }

    rgb = image.astype(np.uint8)
    
    # Create category mappings

    true_categories = {}
    pred_categories = {}
    group_boxes = []

    # Iterate over groups and store what segment ID is in each category
    for group in result["groups"].itertuples(index=False):
        category = group.type

        for t_id in group.true_ids:
            true_categories[int(t_id)] = category

        for p_id in group.predicted_ids:
            pred_categories[int(p_id)] = category

        if category != "matched":
            group_boxes.append(group)

    # Region descriptors for annotation and boxes
    true_regions = {r.label: r for r in regionprops(true_cells)}
    pred_regions = {r.label: r for r in regionprops(predicted_cells)}

    def create_overlay(mask, categories):
        """Create RGBA overlay for instance categories."""

        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)

        for instance_id, category in categories.items():
            pixels = mask == instance_id

            overlay[pixels, :3] = to_rgb(colors[category])
            overlay[pixels, 3] = alpha

        return overlay

    def draw_instance_ids(ax, regions, prefix):
        """Annotate with number."""

        for instance_id, region in regions.items():

            y, x = region.centroid

            ax.text(x, y, f"{prefix}{instance_id}", ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold", bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 1})

    def draw_group_boxes(ax, group, regions, ids):

        """Draw a box around instances."""

        selected = [regions[i] for i in ids if i in regions]

        if not selected:
            return

        min_row = min(r.bbox[0] for r in selected)
        min_col = min(r.bbox[1] for r in selected)
        max_row = max(r.bbox[2] for r in selected)
        max_col = max(r.bbox[3] for r in selected)

        pad = 3

        rect = Rectangle((min_col - pad, min_row - pad), max_col - min_col + 2 * pad, max_row - min_row + 2 * pad,
                        fill=False, edgecolor=colors[group.type], linewidth=1.5, linestyle="-", )

        ax.add_patch(rect)

    # Plots

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)
    ax_input, ax_true, ax_pred = axes
    
    # Show original RGB on every ax
    for i, ax in enumerate(axes):
        ax.imshow(rgb, alpha=1.0 if i == 0 else 0.4)
        ax.set_axis_off()

    # Find perimeter boundary for true and predicted segments
    true_edges = find_boundaries(true_cells, mode="inner")
    pred_edges = find_boundaries(predicted_cells, mode="inner")

    true_boundary_overlay = np.zeros((*true_cells.shape, 4))
    pred_boundary_overlay = np.zeros((*predicted_cells.shape, 4))

    # Set colors
    true_boundary_overlay[true_edges] = [0, 1, 0, 0.9]
    pred_boundary_overlay[pred_edges] = [1, 0.2, 0.2, 0.9]

    # Plot boundaries
    ax_input.imshow(true_boundary_overlay)
    ax_input.imshow(pred_boundary_overlay)

    # Create filled overlays for segments in true and pred axes
    ax_true.imshow(create_overlay(true_cells, true_categories))
    ax_pred.imshow(create_overlay(predicted_cells, pred_categories))

    # Annotate IDs if required
    if show_ids:
        draw_instance_ids(ax_true, true_regions, "T")
        draw_instance_ids(ax_pred, pred_regions, "P")

    # Draw bounding boxes for suspicious groups
    if show_boxes:
        for group in group_boxes:
            draw_group_boxes(ax_true, group, true_regions, group.true_ids)
            draw_group_boxes(ax_pred, group, pred_regions, group.predicted_ids)

    # Titles
    ax_input.set_title("Original image\nGreen: Labels / Red: Predictions")
    ax_true.set_title(f"True Labels ({len(true_regions)} cells)")
    ax_pred.set_title(f"Predictions ({len(pred_regions)} cells)")

    if summary is not None:
        row = summary.iloc[0]
        title = (f"Cellpose Segmentation | " +  f"F1: {row['f1']:.3f} | " +  f"Mean IoU: {row['mean_iou']:.3f} | " + f"AJI: {row['aji_official']:.3f}")
    else:
        title = "Cellpose Instance Segmentation Quality Analysis"

    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Create lagend
    legend = [Patch(facecolor=color, edgecolor="none", label=descriptions[category])
              for category, color in colors.items()]

    # Put legend under the figure
    fig.legend(handles=legend, loc="outside lower center", ncol=len(legend), frameon=False)

    return fig, axes


def plot_dependence(
    df: pd.DataFrame,
    variable: str,
    target: str,
    boolean_mode: bool = False,
    confidence_interval: bool = False,
    scale_x: str = "linear",
    scale_y: str = "linear",
    scatter_alpha: float = 0.01,
    scatter_subsample: float = 0.1,
    scatter_size: int = 3,
    scatter_color: str = "black",
    line_color: str = "orange",
    line_width: int = 3,
    agg_mode: str = "mean",
    window_size: float | None = None,
    convolve_steps: int = 1000,
    min_samples: int = 20,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    seed: int = 1234,
    ax: plt.Axes | None = None
    ) -> plt.Axes:
    """
    Utility to plot target against variable

    Args:
        df (pd.DataFrame): DataFrame to use
        variable (str): variable (x) as a column name
        target (str): target (y) as a column name
        boolean_mode (bool, optional): special mode for cases when target is boolean. Defaults to False.
        scale_x (str, optional): 'linear' or 'log'. Defaults to "linear".
        scale_y (str, optional): 'linear' or 'log'. Defaults to "linear".
        scatter_alpha (float, optional): opacity of scatterplot. Defaults to 0.01.
        scatter_subsample (float, optional): farction of data to take. Defaults to 0.1.
        scatter_size (int, optional): size of scatterplot markers. Defaults to 3.
        scatter_color (str, optional): color of scatterplot markers. Defaults to "black".
        line_color (str, optional): color of line plot. Defaults to "orange".
        line_width (int, optional): width of line plot. Defaults to 3.
        agg_mode (str, optional): mode of aggregation ('mean' or 'median'). Defaults to "mean".
        window_size (float | None, optional): window size for aggregation. Defaults to None.
        convolve_steps (int, optional): number of steps for kernel. Defaults to 1000.
        min_samples (int, optional): minimum samples to consider a sucessful convolution. Defaults to 20.
        xlabel (str | None, optional): label of x axis. Defaults to None.
        ylabel (str | None, optional): label of y axis. Defaults to None.
        title (str | None, optional): title of plot. Defaults to None.
        seed (int, optional): random seed for subsampling. Defaults to 1234.

    """

    if variable not in df.columns or target not in df.columns:
        raise KeyError(f"Not in DF: {variable} or {target}")

    if agg_mode not in ("mean", "median"):
        raise ValueError(f"agg_mode must be 'mean' or 'median', not {agg_mode}")

    if not (0 <= scatter_subsample <= 1):
        raise ValueError(f"scatter_subsample must be between 0 and 1, not {scatter_subsample}")

    if convolve_steps < 1:
        raise ValueError(f"convolve_steps must be positive, not {convolve_steps}")

    x = df[variable].to_numpy().astype(float)
    y = df[target].to_numpy().astype(float)

    # Drop problematic
    validity_mask = np.isfinite(x) & np.isfinite(y)
    if scale_x == "log":
        validity_mask &= (x > 0)
    if scale_y == "log":
        validity_mask &= (y > 0)

    x, y = x[validity_mask], y[validity_mask]

    if len(x) == 0:
        raise ValueError("No valid observations are found")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    np.random.seed(seed)
    n_samples = min(len(x), int(len(x) * scatter_subsample))

    if n_samples > 0:

        chosen_indices = np.random.choice(len(x), size=n_samples, replace=False)

        ax.scatter(x[chosen_indices], y[chosen_indices] if not boolean_mode else np.zeros_like(y[chosen_indices]) - 0.05,
                   alpha=scatter_alpha, s=scatter_size, c=scatter_color, rasterized=True,
                   marker='o' if not boolean_mode else '|')

    # Sort x
    x_for_line = np.log10(x) if scale_x == "log" else x
    order = np.argsort(x_for_line)
    x_sorted = x_for_line[order]
    y_sorted = y[order]
    x_min, x_max = x_sorted[0], x_sorted[-1]
    x_range = x_max - x_min

    # Choose a window size for a kernel
    if window_size is None:
        window_size = x_range / 50 if x_range > 0 else 1.0


    centers = np.linspace(x_min, x_max, convolve_steps)
    estimates = []

    patch_upper_bound = []
    patch_lower_bound = []

    # Iterate over centers of kernels
    for center in centers:

        # Get values inside the window
        left = center - window_size / 2
        right = center + window_size / 2
        mask = (x_for_line >= left) & (x_for_line <= right)
        values = y[mask]

        # Skip if too few points for stability
        if len(values) < min_samples:
            if len(patch_upper_bound) >= 2:
                patch = Polygon(np.array(patch_upper_bound + patch_lower_bound[::-1] + [patch_upper_bound[0]]), color=line_color, alpha=0.3, edgecolor=None, linewidth=0)
                ax.add_patch(patch)
            patch_upper_bound = []
            patch_lower_bound = []
            
            estimates.append(np.nan)
            continue

        # Estimate:
        if agg_mode == "mean" or boolean_mode:
            estimates.append(np.mean(values))
        elif agg_mode == "median":
            estimates.append(np.median(values))


        if confidence_interval:

            mu = np.mean(values)

            delta = 1.96 * np.std(values) / np.sqrt(len(values))
            ci = (10 ** center if scale_x == "log" else center, max(mu - delta, values.min())), (10 ** center if scale_x == "log" else center, min(mu + delta, values.max()))
            patch_upper_bound.append(ci[1])
            patch_lower_bound.append(ci[0])

    if confidence_interval and len(patch_upper_bound) >= 2:
        patch = Polygon(np.array(patch_upper_bound + patch_lower_bound[::-1]), color=line_color, alpha=0.3, edgecolor=None, linewidth=0)
        ax.add_patch(patch)
    

    estimates = np.array(estimates)

    # If log:
    x_curve = 10 ** centers if scale_x == "log" else centers

    ax.plot(x_curve, estimates, c=line_color, lw=line_width, label=f"Sliding {agg_mode}" if not boolean_mode else f"Local Probability")

    if boolean_mode:
        ax.set_ylim((-0.1, estimates[np.isfinite(estimates)].max() + 0.1))
        ax.grid('.', alpha=0.3)

    ax.set_xscale(scale_x)
    ax.set_yscale(scale_y)

    ax.set_xlabel(xlabel or variable)
    ax.set_ylabel(ylabel or (target if not boolean_mode else f'P({target} | {variable})'))
    ax.set_title(title or (f"{target} vs {variable}" if not boolean_mode else f"P({target} | {variable})"))

    ax.legend()

    plt.tight_layout()

    return ax

