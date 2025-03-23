import argparse

import numpy as np

from dataLoader import ORLDataset
from model.LDA import LDAProcessor
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


def train(args, train_set, train_label, test_set):
    model = LDAProcessor(n_components=2)

    output = model.predict(train_set, train_label, test_set)

    return output


def main():
    args = arg_parser()
    dataset = ORLDataset(args.data_path)
    dataset.load_data()
    train_set, train_label, test_set, test_label = dataset.split_personal_image(5)
    train_set = train_set.reshape(train_set.shape[0] * train_set.shape[1], *train_set.shape[2:])
    train_label = train_label.reshape(train_label.shape[0] * train_label.shape[1], *train_label.shape[2:])
    test_label = test_label.reshape(test_label.shape[0] * test_label.shape[1], *test_label.shape[2:])

    print(f"Test data shape: {test_set.shape}")
    print(f"Test labels shape: {test_label.shape}")

    output = train(args, train_set, train_label, test_set)

    print(f"Test accuracy: {accuracy_score(test_label, output)}")
    

if __name__ == '__main__':
    main()
