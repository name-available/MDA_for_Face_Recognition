import argparse

from dataLoader import ORLDataset


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ORL")
    parser.add_argument("--data_path", type=str, default="data/ORL")
    parser.add_argument("--train_image_number", type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    dataset = ORLDataset(args.data_path)
    dataset.load_data()
    train_set, train_label, test_set, test_label = dataset.split_personal_image(5)


if __name__ == '__main__':
    main()
