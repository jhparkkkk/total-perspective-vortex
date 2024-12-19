import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh


class CSP(BaseEstimator, TransformerMixin):
    """
    Corrected CSP implementation with proper initialization of attributes.
    """

    def __init__(self, n_components=4, reg=1e-6):
        self.n_components = n_components
        self.reg = reg
        self.filters = None
        self.mean = None
        self.std = None

    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        cov = np.dot(X_centered, X_centered.T) / X_centered.shape[1]
        return cov / np.trace(cov)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit CSP filters based on the data.
        X: ndarray, shape (n_epochs, n_channels, n_times)
        y: ndarray, shape (n_epochs,)
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP supports only two classes.")

        # Covariance for each class
        identity_regularization = np.eye(X.shape[1]) * self.reg
        covs = [
            np.mean([self._compute_covariance(epoch) for epoch in X[y == c]], axis=0)
            + identity_regularization
            for c in classes
        ]

        # eigenvalue problem
        eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1])
        
        # Sort eigenvectors based on eigenvalues
        sorted_indices = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        self.filters = eigenvectors[:, sorted_indices[: self.n_components]].T

        # Compute transformed data for standardization
        X_transformed = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X_features = np.log((X_transformed**2).mean(axis=2))
        self.mean = X_features.mean(axis=0)
        self.std = X_features.std(axis=0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using CSP filters.
        :param X: ndarray, shape (n_epochs, n_channels, n_times)
        :return: Transformed data, shape (n_epochs, n_components)
        """
        if self.filters is None or self.mean is None or self.std is None:
            raise ValueError(
                "The CSP instance is not fitted yet. Call 'fit' before using this method."
            )

        X_transformed = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X_features = np.log((X_transformed**2).mean(axis=2))
        return (X_features - self.mean) / self.std

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit CSP and transform the data.
        """
        self.fit(X, y)
        return self.transform(X)
