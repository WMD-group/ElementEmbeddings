"""Preprocessing utilities for data preparation."""

import torch


class TorchStandardScaler:
    """Standard scaler for torch tensors.

    Analogous to sklearn.preprocessing.StandardScaler.

    """

    def __init__(
        self,
        mean: float = None,
        std: float = None,
        eps: float = 1e-7,
        correction: int = 1,
    ):
        """Initialise the standard scaler.

        Args:
            mean (torch.Tensor): Mean of the data.
            std (torch.Tensor): Standard deviation of the data.
            eps (float): Epsilon to add to the standard deviation
            to avoid division by zero.
            correction (int): Degrees of freedom correction for the standard deviation.

        """
        self.mean = mean
        self.std = std
        self.eps = eps
        self.correction = correction

    def fit(self, X: torch.Tensor):
        """Compute the mean and standard deviation of the data.

        Args:
            X (torch.Tensor): Data to compute the mean and standard deviation of.
        """
        dims = list(range(X.dim() - 1))
        self.mean = X.mean(dim=dims)
        self.std = X.std(dim=dims, correction=self.correction)

    def transform(self, X: torch.Tensor):
        """Standardise the data.

        Args:
            X (torch.Tensor): Data to standardise.
        """
        X -= self.mean
        X /= self.std + self.eps
        return X

    def fit_transform(self, X: torch.Tensor):
        """Fit to the data and standardise it.

        Args:
            X (torch.Tensor): Data to fit and standardise.
        """
        self.fit(X)
        return self.transform(X)
