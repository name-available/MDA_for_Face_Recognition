from sklearn.neighbors import KNeighborsClassifier
import numpy as np

np.random.seed(42)


class FRClassification:

    def __init__(self):
        pass

    def project_tensor(self, tensor, projections):
        for projection in projections:
            tensor = np.dot(projection.T, tensor)
        return tensor

    def classify(self, train_data, train_labels, test_data, projections):
        # 投影到低维空间
        train_proj = [self.project_tensor(t, projections) for t in train_data]
        test_proj = [self.project_tensor(t, projections) for t in test_data]
        # 使用最近邻分类器
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(train_proj, train_labels)
        return classifier.predict(test_proj)
