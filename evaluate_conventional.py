import argparse
from machine_learning.model import get_model_faster_rcnn
import machine_learning.utilities.transforms as T
from machine_learning.data.minne_apple_dataset import MinneAppleDataset
from eval.evaluate import match_points_to_box_centres, compute_metrics
from conventional.conventional import conventional

import numpy as np
import cv2
import pickle
import time

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(args):
    test_set = MinneAppleDataset(args.data_path, get_transform(train=False))

    # HSV colour thresholds for apples
    hsv_thresholds = [
        ([20, 127, 240], [40, 255, 255]), # yellow
        ([10, 127, 200], [25, 255, 255]), # orange
        ([160, 127, 63], [200, 255, 255]) # red
    ]

    results = []
    for i in range(len(test_set)):
        if i % 10 == 0:
            print("Evaluating image {} of {}".format(i, len(test_set)))
        # ground truth
        img, target = test_set[i]
        gt_boxes = target['boxes'].cpu().detach().numpy()
        gt_labels = target['labels'].cpu().detach().numpy()

        # convert torch img to opencv
        img = img.cpu().detach().numpy()
        img = np.moveaxis(img, 0, 2)
        img = img[:,:,::-1].copy() * 255
        img = img.astype(np.uint8)

        predictions = conventional(img, hsv_thresholds, args.kernel_size, args.iterations)

        # compute metrics
        if len(predictions):
            matches = match_points_to_box_centres(predictions, gt_boxes)
        else:
            matches = []
        metrics = compute_metrics(len(predictions), len(gt_boxes), len(matches))
        results.append(metrics)

        # display
        if args.display:
            print(metrics)
            frame = img.copy()
            for box in gt_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (192,63,255), 2)
            for point in predictions:
                x1, y1 = [int(v) for v in point]
                cv2.circle(frame, (x1,y1), 5, (255,255,0), -1)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

    counting_accuraccy = [metrics["Counting Accuracy"] for metrics in results]
    print("Average counting accuracy:", np.mean(counting_accuraccy))
    tp = np.sum([metrics["TP"] for metrics in results])
    fp = np.sum([metrics["FP"] for metrics in results])
    fn = np.sum([metrics["FN"] for metrics in results])
    total_metrics = compute_metrics(tp + fp, tp + fn, tp)
    print("Total counting accuracy:", total_metrics["Counting Accuracy"])

    filename = 'results_{}.pkl'.format(time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print('Results written to file {}'.format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to image data")
    parser.add_argument("--display", action="store_true", help="path to model weights")
    parser.add_argument("--kernel_size", type=int, default="2", help="size of opening kernel")
    parser.add_argument("--iterations", type=int, default="2", help="iterations of opening for noise removal")
    args = parser.parse_args()
    print("Applying conventional approach with kernel size {} and {} iterations".format(args.kernel_size, args.iterations))
    main(args)