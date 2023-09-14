"""Tools for training pytorch models using lightning."""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class LightningRegressor(pl.LightningModule):
    """Lightning module for regression."""

    def __init__(self, backbone):
        """Initialise LightningRegressor  class.

        Args:
            backbone (nn.Module): Backbone model.

        """
        super().__init__()
        self.backbone = backbone
        self.save_hyperparameters()

    def mean_absolute_error(self, y_hat, y):
        """Compute the mean absolute error.

        Args:
            y_hat (torch.Tensor): Predicted values.
            y (torch.Tensor): Actual values.

        Returns:
            torch.Tensor: Mean absolute error.

        """
        return F.l1_loss(y_hat, y)

    def mean_squared_error(self, y_hat, y):
        """Compute the mean squared error.

        Args:

            y_hat (torch.Tensor): Predicted values.
            y (torch.Tensor): Actual values.

        Returns:
            torch.Tensor: Mean squared error.

        """
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.

        """
        x, y = batch
        y_hat = self.backbone(x)
        # Compute loss
        loss = self.mean_absolute_error(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.

        """  # noqa: D401 .
        x, y = batch
        y_hat = self.backbone(x)
        # Compute loss
        loss = self.mean_absolute_error(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers.

        Returns:
            torch.optim.Optimizer: Optimizer.

        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class LightningClassifier(pl.LightningModule):
    """Lightning module for classification."""

    def __init__(self, backbone, loss_fn=F.cross_entropy):
        """Initialise LightningClassifier class.

        Args:
            backbone (nn.Module): Backbone model.
            loss_fn (function): Loss function.

        """
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def cross_entropy(self, y_hat, y):
        """Compute the cross entropy.

        Args:
            y_hat (torch.Tensor): Predicted values.
            y (torch.Tensor): Actual values.

        Returns:
            torch.Tensor: Cross entropy.

        """
        return F.cross_entropy(y_hat, y)

    def accuracy(self, y_hat, y):
        """Compute the accuracy.

        Args:
            y_hat (torch.Tensor): Predicted values.
            y (torch.Tensor): Actual values.

        Returns:
            torch.Tensor: Accuracy.

        """
        return (y_hat.argmax(dim=1) == y).float().mean()

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.

        """
        x, y = batch
        y_hat = self.backbone(x)
        # Compute loss
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.

        """  # noqa: D401 .
        x, y = batch
        y_hat = self.backbone(x)
        # Compute loss
        loss = self.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers.

        Returns:
            torch.optim.Optimizer: Optimizer.

        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
