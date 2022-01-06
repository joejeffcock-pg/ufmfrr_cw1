import argparse
import torch
import torchvision
from machine_learning.model import get_model_faster_rcnn
import machine_learning.utilities.transforms as T
from machine_learning.data.minne_apple_dataset import MinneAppleDataset
from eval.evaluate import match_predictions_to_ground_truths, compute_metrics

import numpy as np
import cv2

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    test_set = MinneAppleDataset(args.data_path, get_transform(train=False))

    model = get_model_faster_rcnn(num_classes)
    model.load_state_dict(torch.load(args.weights_path))
    model.to(device)
    model.eval()

    results = []
    for i in range(len(test_set)):
        if i % 100 == 0:
            print("Evaluating image {} of {}".format(i, len(test_set)))
        # ground truth
        img, target = test_set[i]
        gt_boxes = target['boxes'].cpu().detach().numpy()
        gt_labels = target['labels'].cpu().detach().numpy()

        # NN predictions
        predictions = model([img.to(device)])
        pred_boxes = predictions[0]['boxes'].cpu().detach().numpy()
        pred_labels = predictions[0]['labels'].cpu().detach().numpy()
        pred_scores = predictions[0]['scores'].cpu().detach().numpy()

        # non-maximum suppression
        indices = torchvision.ops.batched_nms(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels'], 0.3)

        # filter by score
        pred_boxes_temp = []
        for index in indices:
            x1, y1, x2, y2 = [int(v) for v in pred_boxes[index]]
            score = pred_scores[index]
            if score > 0.1:
                pred_boxes_temp.append([x1,y1,x2,y2])
        pred_boxes = pred_boxes_temp

        # compute metrics
        if len(pred_boxes):
            matches = match_predictions_to_ground_truths(pred_boxes, gt_boxes)
        else:
            matches = []
        metrics = compute_metrics(len(pred_boxes), len(gt_boxes), len(matches))
        results.append(metrics)

        # display
        if args.display:
            print(metrics)
            frame = img.cpu().detach().numpy()
            frame = np.moveaxis(frame, 0, 2)
            frame = frame[:,:,::-1].copy()
            for box in gt_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0.75,0.25,1), 2)
            for box in pred_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (1,1,0), 2)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

    counting_accuraccy = [metrics["Counting Accuracy"] for metrics in results]
    print("Average counting accuracy:", np.mean(counting_accuraccy))
    tp = np.sum([metrics["TP"] for metrics in results])
    fp = np.sum([metrics["FP"] for metrics in results])
    fn = np.sum([metrics["FN"] for metrics in results])
    total_metrics = compute_metrics(tp + fp, tp + fn, tp)
    print("Total counting accuracy:", total_metrics["Counting Accuracy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to image data")
    parser.add_argument("weights_path", help="path to model weights")
    parser.add_argument("--display", action="store_true", help="path to model weights")
    args = parser.parse_args()
    main(args)