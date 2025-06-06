from collections import defaultdict
from typing import List, Dict

import hydra
import json
import torch
import numpy as np


@hydra.main(config_path="configs", config_name="eval_fmiyc.yaml")
def main(cfg):
    # Load precalculated inference results
    base_data_folder = f"./data/{cfg.ind_dataset.upper()}-Detection/faster-rcnn/vanilla/random_seed_0/inference"
    inference_file_names = "standard_nms/corruption_level_0/coco_instances_results_SAFE_RCNN-RN50.json"
    ood_data_raw = {}
    ood_data_processed = {}
    if cfg.ind_dataset == "voc":
        # coco_name = 'coco_ood_val{}'.format('_bdd' if 'BDD' in args.config_file else '')
        # names = [args.test_dataset, "coco_ood_near", "coco_ood_far", "openimages_ood_near", "openimages_ood_far"]
        ind_file_name = f"{base_data_folder}/{cfg.ind_dataset}_custom_val/{inference_file_names}"
        ind_data = json.load(open(ind_file_name, 'r'))
        print(len(ind_data))
        for ood_dataset_name in cfg.ood_datasets:
            ood_file_name = f"{base_data_folder}/{ood_dataset_name}/{inference_file_names}"
            ood_data_raw[ood_dataset_name] = json.load(open(ood_file_name, 'r'))
            ood_data_processed[ood_dataset_name] = eval_predictions_preprocess(ood_data_raw[ood_dataset_name], min_allowed_score=0.56)


    # BDD as ID
    else:
        # names = [args.test_dataset, "coco_ood_farther", "openimages_ood_farther"]
        # data_dir = dirname(dirname(args.dataset_dir))
        raise NotImplementedError

    # Filter inference results based on threshold


def eval_predictions_preprocess(
        predicted_instances: List[Dict],
        min_allowed_score: float = 0.0,
        is_odd=True,
):
    predictions = dict()

    for predicted_instance in predicted_instances:
        # Remove predictions with undefined category_id. This is used when the training and
        # inference datasets come from different data such as COCO-->VOC or COCO-->OpenImages.
        # Only happens if not ODD dataset, else all detections will be removed.

        # if len(predicted_instance['cls_prob']) == 81 or len(predicted_instance['cls_prob']) == 21 or len(predicted_instance['cls_prob']) == 11:
        #     cls_prob = predicted_instance['cls_prob'][:-1]
        # else:
        #     cls_prob = predicted_instance['cls_prob']
        cls_prob = predicted_instance['cls_prob']
        box_inds = predicted_instance['bbox']
        box_inds = np.array([box_inds[0],
                             box_inds[1],
                             box_inds[0] + box_inds[2],
                             box_inds[1] + box_inds[3]])
        logistic_score = predicted_instance['logistic_score']
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
                    "logits": torch.as_tensor([cls_prob], dtype=torch.float32),
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
                        torch.as_tensor([cls_prob], dtype=torch.float32)
                    )
                )
                predictions[predicted_instance['image_id']]["safe"].append(logistic_score)

    return predictions


if __name__ == '__main__':
    main()
