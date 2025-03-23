from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class LDAProcessor:
    def __init__(self, n_components):
        self.n_components = n_components
        self.lda = LDA(n_components=n_components)
        self.knn = None  # Placeholder for KNN model

    def fit_transform(self, X, y):
        """
        Apply LDA to reduce dimensionality of the data.
        
        :param X: Feature matrix
        :param y: Target labels
        :return: Transformed feature matrix
        """
        return self.lda.fit_transform(X, y)

    def transform(self, X):
        """
        Transform new data using the fitted LDA model.
        
        :param X: New feature matrix
        :return: Transformed feature matrix
        """
        return self.lda.transform(X)

    def fit(self, X, y):
        """
        Fit the LDA model to the data.
        
        :param X: Feature matrix
        :param y: Target labels
        """
        self.lda.fit(X, y)

    def predict(self, X_train, y_train, X_test, n_neighbors=3):
        # Flatten the train_set and test_set to 2D arrays
        X_train = X_train.reshape(-1, X_train.shape[-1] * X_train.shape[-2]) # (n_samples * train_image_number, height * width)
        X_test = X_test.reshape(-1, X_test.shape[-1] * X_test.shape[-2])    # (n_samples * train_image_number, height * width)

        # Transform training and test data using LDA
        X_train_transformed = self.fit_transform(X_train, y_train)
        X_test_transformed = self.transform(X_test)

        # Train KNN model
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.knn.fit(X_train_transformed, y_train)

        # Predict and return results
        return self.knn.predict(X_test_transformed)


if __name__ == "__main__":
    pass