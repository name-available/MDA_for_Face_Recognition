import argparse

import numpy as np

from dataLoader import ORLDataset
from model.MDA_test import MDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ORL")
    parser.add_argument("--data_path", type=str, default="data/ORL")
    parser.add_argument("--train_image_number", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    args = parser.parse_args()
    return args


def train(args, train_set, train_label):
    print(train_set[0].shape)
    print(train_label.shape)
    model = MDA(input_dim=list(train_set[0].shape),
                output_dim=[10, 20],
                epochs=args.epochs,
                epsilon=args.epsilon)
    model.fit(train_set, train_label)

    return model


def main():
    args = arg_parser()
    dataset = ORLDataset(args.data_path)
    dataset.load_data()
    train_set, train_label, test_set, test_label = dataset.split_personal_image(5)
    train_set = train_set.reshape(train_set.shape[0] * train_set.shape[1], *train_set.shape[2:])
    train_label = train_label.reshape(train_label.shape[0] * train_label.shape[1], *train_label.shape[2:])

    train_set = train_set.reshape(train_set.shape[0], -1)
    print(train_set.shape)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_set, train_label)

    print(f"Train accuracy: {accuracy_score(knn.predict(train_set), train_label)}")


if __name__ == '__main__':
    main()
