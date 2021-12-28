import argparse
import torch
import torchvision
from machine_learning.model import get_model_faster_rcnn
import machine_learning.utilities.transforms as T
from machine_learning.data.minne_apple_dataset import MinneAppleDataset
from eval.evaluate import match_predictions_to_ground_truths, compute_metrics
from sort.sort import Sort
from PIL import Image

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

        # new tracker
        sort = Sort()

        identities = set()
        half_width = int(img.shape[2]/2)
        border_dist = 70
        border = [half_width - int(border_dist/2), half_width + int(border_dist/2)]
        offset = -border[1]
        stride = 5

        while offset < border[1]:
            # translate image
            img_affine = torchvision.transforms.functional.affine(img, 0, [offset,0], 1, 0)

            # NN predictions
            predictions = model([img_affine.to(device)])
            pred_boxes = predictions[0]['boxes'].cpu().detach().numpy()
            pred_labels = predictions[0]['labels'].cpu().detach().numpy()
            pred_scores = predictions[0]['scores'].cpu().detach().numpy()

            # non-maximum suppression
            indices = torchvision.ops.batched_nms(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels'], 0.3)

            # filter by score
            dets = []
            for index in indices:
                x1, y1, x2, y2 = [int(v) for v in pred_boxes[index]]
                score = pred_scores[index]
                area = (x2 - x1) * (y2 - y1)
                if score > 0.1:
                    dets.append([x1,y1,x2,y2,score])

            # update sort
            if len(dets):
                tracks = sort.update(np.array(dets))
            else:
                tracks = sort.update(np.empty((0, 5)))

            # add identities within user-defined borders
            for track in tracks:
                x1, y1, x2, y2 = [int(v) for v in track[:4]]
                cx = (x1 + x2)/2
                identity = track[4]
                if cx > border[0] and cx < border[1]:
                    identities.add(identity)

            # display
            if args.display:
                frame = img_affine.cpu().detach().numpy()
                frame = np.moveaxis(frame, 0, 2)
                frame = frame[:,:,::-1].copy() - 0.35
                cv2.line(frame, (border[0], 0), (border[0], frame.shape[0]), (0,0,255), 2)
                cv2.line(frame, (border[1], 0), (border[1], frame.shape[0]), (0,0,255), 2)
                for box in gt_boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1 += offset
                    x2 += offset
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0.75,0.25,1), 1)
                for track in tracks:
                    x1, y1, x2, y2 = [int(v) for v in track[:4]]
                    identity = int(track[4])
                    cv2.putText(frame, str(identity), (x1,y1+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,1,0), 1)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,1,0), 1)
                cv2.putText(frame, "apple count: {}/{}".format(len(identities), len(gt_boxes)), (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(1)

            offset += stride

        actual = len(gt_boxes)
        counted = len(identities)
        results.append((counted - actual)/actual)
        if args.display:
            print(results[-1])
            cv2.waitKey(0)

    print('mean error:', np.mean(results))
    print('mean abs error:', np.mean(np.absolute(results)))
    print('std:', np.std(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to image data")
    parser.add_argument("weights_path", help="path to model weights")
    parser.add_argument("--display", action="store_true", help="path to model weights")
    args = parser.parse_args()
    main(args)