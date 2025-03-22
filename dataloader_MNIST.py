import numpy as np
from torchvision import datasets, transforms

class MNISTDataset:
    def __init__(self, data_path=None):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5
        ])
        self.data_path = data_path
        

    def load_data(self, train=True):
        # Load MNIST dataset
        if self.data_path is not None:
            dataset = datasets.MNIST(root=self.data_path, train=train, download=False, transform=self.transform)
        else:
            dataset = datasets.MNIST(root='./data', train=train, download=True, transform=self.transform)
        
        data = []
        labels = []
        for img, label in dataset:
            data.append(img.numpy())
            labels.append(label)
        
        data = np.array(data)
        labels = np.array(labels)
        return data, labels
    
    def split_personal_image(self, train_size):
        train_set = []
        train_labels = []
        test_set = []
        test_labels = []

        unique_labels = np.unique(self.load_data(train=True)[1])
        for label in unique_labels:
            label_indices = np.where(self.load_data(train=True)[1] == label)[0]
            np.random.shuffle(label_indices)

            train_indices = label_indices[:train_size]
            test_indices = label_indices[train_size:]

            train_set.append(self.load_data(train=True)[0][train_indices])
            train_labels.append(self.load_data(train=True)[1][train_indices])
            test_set.append(self.load_data(train=True)[0][test_indices])
            test_labels.append(self.load_data(train=True)[1][test_indices])

        return (np.concatenate(train_set), np.concatenate(train_labels),
                np.concatenate(test_set), np.concatenate(test_labels))
    
    def split_test(self, dataset_ratio=0.1, test_ratio=0.1):
        train_set = []
        train_labels = []
        test_set = []
        test_labels = []

        # Load the full dataset and reduce it to the specified dataset_ratio
        data, labels = self.load_data(train=True)
        total_size = int(len(labels) * dataset_ratio)
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        reduced_indices = indices[:total_size]

        reduced_data = data[reduced_indices]
        reduced_labels = labels[reduced_indices]

        unique_labels = np.unique(reduced_labels)
        for label in unique_labels:
            label_indices = np.where(reduced_labels == label)[0]
            np.random.shuffle(label_indices)

            test_size = int(len(label_indices) * test_ratio)
            test_indices = label_indices[:test_size]
            train_indices = label_indices[test_size:]

            train_set.append(reduced_data[train_indices])
            train_labels.append(reduced_labels[train_indices])
            test_set.append(reduced_data[test_indices])
            test_labels.append(reduced_labels[test_indices])

        return (np.concatenate(train_set), np.concatenate(train_labels),
                np.concatenate(test_set), np.concatenate(test_labels))

# Example usage
dataset = MNISTDataset()
train_set, train_label, test_set, test_label = dataset.split_test(dataset_ratio=0.1, test_ratio=0.1)


print(f"Train data shape: {train_set.shape}")
print(f"Train labels shape: {train_label.shape}")
print(f"Test data shape: {test_set.shape}")
print(f"Test labels shape: {test_label.shape}")