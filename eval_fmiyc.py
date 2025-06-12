from typing import List, Dict

import hydra
import json
import torch
import numpy as np

from ls_ood_detect_cea.uncertainty_estimation import COCOParser, get_overall_open_set_results, get_baselines_thresholds, \
    convert_osod_results_to_pandas_df
from SAFE.shared.metric_utils import get_measures, convert_auroc_results_to_pandas_df

BASELINES_NAMES = ["safe"]
INVERT_SCORES = True


@hydra.main(config_path="configs", config_name="eval_fmiyc.yaml")
def main(cfg):
    # Load precalculated inference results
    base_data_folder = f"./data/{cfg.ind_dataset.upper()}-Detection/faster-rcnn/vanilla/random_seed_0/inference"
    inference_file_names = "standard_nms/corruption_level_0/coco_instances_results_SAFE_RCNN-RN50.json"
    # Load InD inference file
    ind_file_name = f"{base_data_folder}/{cfg.ind_dataset}_custom_val/{inference_file_names}"
    ind_data = json.load(open(ind_file_name, 'r'))

    # Filter inference results based on threshold
    ind_data = {
        "valid": eval_predictions_preprocess(
            ind_data,
            min_allowed_score=cfg.min_allowed_score,
            invert_scores=INVERT_SCORES
        )
    }
    ind_baseline_scores = {
        "safe": get_baseline_scores(data_predictions=ind_data["valid"], baseline_name=BASELINES_NAMES[0])
    }
    ood_data = {}
    ood_baseline_scores = {}
    for ood_dataset_name in cfg.ood_datasets:
        ood_file_name = f"{base_data_folder}/{ood_dataset_name}/{inference_file_names}"
        ood_data[ood_dataset_name] = json.load(open(ood_file_name, 'r'))
        # Filter inference results based on threshold
        ood_data[ood_dataset_name] = eval_predictions_preprocess(
            ood_data[ood_dataset_name],
            min_allowed_score=cfg.min_allowed_score,
            invert_scores=INVERT_SCORES
        )
        ood_baseline_scores[ood_dataset_name] = get_baseline_scores(
            data_predictions=ood_data[ood_dataset_name],
            baseline_name=BASELINES_NAMES[0]
        )
    # Get classic metrics
    classic_metrics = {}
    for ood_dataset_name in cfg.ood_datasets:
        auroc, aupr, fpr = get_measures(ind_baseline_scores[BASELINES_NAMES[0]], ood_baseline_scores[ood_dataset_name])
        classic_metrics[ood_dataset_name] = {
            "auroc": auroc,
            "aupr": aupr,
            "fpr": fpr,
        }
    metrics_df = convert_auroc_results_to_pandas_df(classic_metrics, cfg.ood_datasets, dataset_as_data=False)
    print(metrics_df)

    # Read GT annotations
    # ind_annotations = COCOParser(cfg.ind_annotations_path[cfg.ind_dataset])
    ood_annotations = {}
    for ood_dataset_name in cfg.ood_datasets:
        ood_annotations[ood_dataset_name] = COCOParser(cfg.ood_annotations_paths[cfg.ind_dataset][ood_dataset_name])

    # Complete the missing images entries with empty predictions
    for ood_dataset_name in cfg.ood_datasets:
        ood_data[ood_dataset_name] = fill_missing_im_ids_predictions(
            ood_annotations[ood_dataset_name],
            ood_data[ood_dataset_name]
        )
    baselines_thresholds = get_baselines_thresholds(
        baselines_names=BASELINES_NAMES,
        baselines_scores_dict=ind_baseline_scores,
        z_score_percentile=cfg.z_score_thresholds
    )
    open_set_results = get_overall_open_set_results(
        ind_dataset_name=cfg.ind_dataset,
        ind_gt_annotations_path=cfg.ind_annotations_path[cfg.ind_dataset],
        ind_data_dict=ind_data,
        ood_data_dict=ood_data,
        ood_datasets_names=cfg.ood_datasets,
        ood_annotations_paths=cfg.ood_annotations_paths[cfg.ind_dataset],
        methods_names=["safe"],
        methods_thresholds=baselines_thresholds,
        metric_2007=cfg.metric_2007,
        evaluate_on_ind=True,
        get_known_classes_metrics=cfg.get_known_classes_metrics,
        using_id_val_subset=list(ind_data["valid"].keys())
    )
    osod_pd_dfs = {}
    for ood_dataset_name in cfg.ood_datasets:
        osod_pd_dfs[ood_dataset_name] = convert_osod_results_to_pandas_df(
            open_set_results=open_set_results[ood_dataset_name],
            methods_names=BASELINES_NAMES,
            save_method_as_data=False
        )
        print(ood_dataset_name)
        print(osod_pd_dfs[ood_dataset_name])
    # print(open_set_results)
    print("Done!")


def fill_missing_im_ids_predictions(annotations: COCOParser, predictions: Dict) -> Dict:
    for im_id in annotations.annIm_dict.keys():
        if im_id not in predictions.keys():
            predictions[im_id] = {
                "boxes": torch.tensor([], dtype=torch.float32),
                "logits": torch.tensor([], dtype=torch.float32),
                "safe": [],
            }

    return predictions


def get_baseline_scores(data_predictions, baseline_name):
    baseline_scores = []
    for im_id, preds in data_predictions.items():
        baseline_scores.extend(preds[baseline_name])
    return np.array(baseline_scores)


def eval_predictions_preprocess(
        predicted_instances: List[Dict],
        min_allowed_score: float,
        invert_scores: bool,
) -> Dict:
    """
    The function preprocesses by filtering the predictions that are below a certain threshold. It can optionally invert
    scores (multiply by minus sign)
    """
    predictions = dict()

    for predicted_instance in predicted_instances:
        cls_prob = predicted_instance['cls_prob']
        box_inds = predicted_instance['bbox']
        box_inds = np.array([box_inds[0],
                             box_inds[1],
                             box_inds[0] + box_inds[2],
                             box_inds[1] + box_inds[3]])
        logistic_score = predicted_instance['logistic_score']
        if invert_scores:
            logistic_score = -logistic_score
        if len(cls_prob) == 21 or len(cls_prob) == 11:
            discard_pred = np.array(cls_prob[:-1]).max(0) < min_allowed_score
        else:
            discard_pred = np.array(cls_prob).max(0) < min_allowed_score

        if discard_pred:
            if not predicted_instance['image_id'] in predictions.keys():
                predictions[predicted_instance['image_id']] = {
                    "boxes": torch.tensor([], dtype=torch.float32),
                    "logits": torch.tensor([], dtype=torch.float32),
                    "safe": [],
                }

        else:
            if not predicted_instance['image_id'] in predictions.keys():
                predictions[predicted_instance['image_id']] = {
                    "boxes": torch.as_tensor([box_inds], dtype=torch.float32),
                    "logits": torch.log(torch.as_tensor([cls_prob], dtype=torch.float32)),
                    "safe": [logistic_score],
                }
            else:
                predictions[predicted_instance['image_id']]["boxes"] = torch.cat(
                    (
                        predictions[predicted_instance['image_id']]["boxes"],
                        torch.as_tensor([box_inds], dtype=torch.float32)
                    )
                )
                predictions[predicted_instance['image_id']]["logits"] = torch.cat(
                    (
                        predictions[predicted_instance['image_id']]["logits"],
                        torch.log(torch.as_tensor([cls_prob], dtype=torch.float32))
                    )
                )
                predictions[predicted_instance['image_id']]["safe"].append(logistic_score)

    return predictions


if __name__ == '__main__':
    main()
