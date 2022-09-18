"""
PyTorch Lightning module for baseline FCNN model training.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
import torch
from torch.nn import BCELoss, MSELoss
from typing import Any, Dict

from baseline.fcnn import FCNN
from data.PMBB import PMBBSample


class FCNNModule(pl.LightningModule):
    """PyTorch Lightning module for baseline FCNN model training."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 128,
        num_layers: int = 8,
        activation: str = "ReLU",
        task: str = "classification",
        A1C_threshmin: float = 6.5,
        lr: float = 0.001,
        lr_step_size: int = 40,
        lr_gamma: float = 0.5
    ):
        """
        Args:
            in_chans: number of input channels for the first layer.
            out_chans: number of output channels from the last layer.
            chans: number of channels in the intermediate layers.
            num_layers: total number of layers.
            activation: activation function. One of [`Sigmoid`, `Softplus`,
                `ReLU`, `LeakyReLU`].
            task: model task. One of [`classification`, `regression`].
            A1C_threshmin: A1C threshold value for positive classification.
            lr: learning rate. Default 0.001.
            lr_step_size: learning rate step size. Default 40.
            lr_gamma: learning rate decay constant. Default 0.5.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_layers = num_layers
        self.activation = activation
        self.task = task
        self.model = FCNN(in_chans, out_chans, chans, num_layers, activation)
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        if self.task.lower() == "regression":
            self.loss = MSELoss()
        elif self.task.lower() == "classification":
            self.loss = BCELoss()
            self.A1C_threshmin = A1C_threshmin
        else:
            raise NotImplementedError(f"Unrecognized model task {self.task}.")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Input:
            X: input feature vector with flattened size of self.in_chans.
        Returns:
            FCNN(X) with a flattened size of self.out_chans.
        """
        return self.model(torch.flatten(X, start_dim=1))

    def training_step(self, batch: PMBBSample, batch_idx: int) -> float:
        X = batch.data[..., :-1].float()
        Y = batch.data[..., -1].float()
        output = torch.flatten(torch.sigmoid(self(X)))

        if self.task.lower() == "regression":
            loss = self.loss(output, batch.data["A1C"])
        elif self.task.lower() == "classification":
            loss = self.loss(output, (Y >= self.A1C_threshmin).float())

        self.log("train_loss", torch.mean(loss))
        return loss

    def validation_step(
        self, batch: PMBBSample, batch_idx: int
    ) -> Dict[str, Any]:
        X = batch.data[..., :-1].float()
        Y = batch.data[..., -1].float()
        output = torch.flatten(torch.sigmoid(self(X)))

        if self.task.lower() == "regression":
            loss = self.loss(output, batch.data["A1C"])
        elif self.task.lower() == "classification":
            loss = self.loss(output, (Y >= self.A1C_threshmin).float())
        self.log("val_metrics/loss", loss, prog_bar=True)
        return {
            "valid_loss": loss
        }

    def test_step(self, batch: PMBBSample, batch_idx: int) -> float:
        X = batch.data[..., :-1].float()
        Y = batch.data[..., -1].float()
        output = torch.flatten(torch.sigmoid(self(X)))

        if self.task.lower() == "regression":
            loss = self.loss(output, batch.data["A1C"])
        elif self.task.lower() == "classification":
            loss = self.loss(output, (Y >= self.A1C_threshmin).float())
        self.log("val_metrics/loss", loss, prog_bar=True)
        return {
            "test_loss": loss,
            "preds": torch.sigmoid(output).detach().cpu().numpy().tolist(),
            "gts": Y.detach().cpu().numpy().tolist()
        }

    def test_epoch_end(self, test_logs: Dict[str, Any]) -> None:
        preds, gts = [], []
        for log in test_logs:
            preds += log["preds"]
            gts += [x >= self.A1C_threshmin for x in log["gts"]]

        thresholds = np.array(list(range(1000))) / 1000.0
        scores = []
        for thresh in thresholds:
            scores.append(accuracy_score(gts, preds >= thresh))
        best_thresh = thresholds[np.argmax(np.array(scores))]
        tp, fp, tn, fn = 0, 0, 0, 0
        for y, yhat in zip(gts, preds >= best_thresh):
            if y:
                if yhat:
                    tp += 1
                else:
                    fn += 1
            else:
                if yhat:
                    fp += 1
                else:
                    tn += 1
        eps = np.finfo(np.float64).eps
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        print(f"N = {len(preds)}")
        print(f"Sensitivity (Recall): {recall:.3f}")
        print(f"Precision (PPV): {precision:.3f}")
        print(f"Specificity: {(tn / (tn + fp + eps)):.3f}")
        print(f"Accuracy: {np.max(np.array(scores)):.3f}")

        return None

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]
