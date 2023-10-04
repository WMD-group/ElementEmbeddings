from abc import ABCMeta, abstractmethod

import lightning.pytorch as pl
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanAbsoluteError, R2Score


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    """Base class for all models."""

    def __init__(self, num_classes: int):
        """Initialise the model."""
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        if self.num_classes > 1:
            self.classification = True
        else:
            self.classification = False
        if self.classification:
            self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        else:
            self.r2 = R2Score()
            self.mae = MeanAbsoluteError()

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        pass
