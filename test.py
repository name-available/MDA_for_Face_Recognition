import argparse

import numpy as np

from dataloader_MNIST import MNISTDataset
from model.MDA_test import MDA


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ORL")
    parser.add_argument("--data_path", type=str, default="data/ORL")
    parser.add_argument("--train_image_number", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-3)

    parser.add_argument("--dataset_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    return args


def train(args, train_set, train_label):
    train_set = train_set.reshape(train_set.shape[0] * train_set.shape[1], *train_set.shape[2:])
    # train_label = train_label.reshape(train_label.shape[0] * train_label.shape[1], *train_label.shape[2:])
    model = MDA(input_dim=list(train_set[0].shape),
                output_dim=[10, 20],
                epochs=args.epochs,
                epsilon=args.epsilon)
    model.fit(train_set, train_label)


def main():
    args = arg_parser()
    dataset = MNISTDataset()
    dataset.load_data()
    train_set, train_label, test_set, test_label = dataset.split_test(test_ratio=args.dataset_ratio, dataset_ratio=args.dataset_ratio)
    train(args, train_set, train_label)


if __name__ == '__main__':
    main()


