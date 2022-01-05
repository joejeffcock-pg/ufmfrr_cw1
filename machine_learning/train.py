"""
Reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import argparse
import torch
from model import get_model_faster_rcnn
import utilities.utils as utils
from utilities.engine import train_one_epoch, evaluate
from data.minne_apple_dataset import MinneAppleDataset
from data.coco_dataset import CocoDataset
from data.coco_dataset import get_fiftyone_dataset

import utilities.transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and APPLE
    num_classes = 2

    # use our dataset and defined transformations
    if args.dataset == "MinneApple":
        dataset = MinneAppleDataset(args.train_data_path, get_transform(train=True))
        dataset_test = MinneAppleDataset(args.val_data_path, get_transform(train=False))
    elif args.dataset == "COCO":
        # first build the 'fiftyone' datasets
        dataset_51 = get_fiftyone_dataset(args.train_data_path)
        dataset_test_51 = get_fiftyone_dataset(args.val_data_path)

        dataset = CocoDataset(dataset_51, get_transform(train=True))
        dataset_test = CocoDataset(dataset_test_51, get_transform(train=False))
    else:
        raise ValueError("Dataset: {} is not supported".format(args.dataset))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # create model and load it onto device
    model = get_model_faster_rcnn(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), './weights.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path", help="path to training data")
    parser.add_argument("val_data_path", help="path to validation data")
    # parser.add_argument("--dataset", dest="dataset", default="MinneApple", help="Choice of dataset: MinneApple/COCO")
    parser.add_argument("--dataset", dest="dataset", default="COCO", help="Choice of dataset: MinneApple/COCO")
    args = parser.parse_args()
    main(args)
