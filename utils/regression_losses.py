from typing import List

import torch
import torch.nn as nn
import numpy as np


class RegressionLoss:
    def __init__(self):
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss()


class CombinedLoss(RegressionLoss):
    def __init__(self):
        super().__init__()
        print("Initializing CombinedLoss engine")

    def weighted_combined_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, labels: torch.Tensor,
                               alpha: torch.Tensor):
        """
        @param predictions: Flattened 1D tensor of shape (N,), containing the predicted intensity values.
	    @param ground_truth: Flattened 1D tensor of shape (N,), containing the actual intensity values.
        @param labels: Flattened 1D tensor of shape (N,), containing the label for each point.
        @param alpha: Torch tensor of shape (num_labels,), where num_labels is the number of
                      unique labels in the labels tensor. This tensor is not learnable.
        """
        total_loss = 0
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = (labels == label)  # Mask for current label
            if mask.sum() > 0:  # Avoid empty masks
                mae = self.mae_loss(predictions[mask], ground_truth[mask]).mean()
                huber = self.huber_loss(predictions[mask], ground_truth[mask]).mean()
                total_loss += alpha[label] * mae + (1 - alpha[label]) * huber
        return total_loss

    def dynamic_combined_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, labels: torch.Tensor,
                              alpha: nn.Parameter):
        """
        Use this loss function for learnable weights in combined loss. Pass alpha parameter like this:

        ```
        num_labels = len(torch.unique(labels))
        alpha = nn.Parameter(torch.ones(num_labels) * 0.5)  # Start with equal weights
        # Add alpha to your optimizer
        optimizer = torch.optim.Adam(model.parameters() + [alpha], lr=0.001)
        ```
        @param predictions: Flattened 1D tensor of shape (N,), containing the predicted intensity values.
	    @param ground_truth: Flattened 1D tensor of shape (N,), containing the actual intensity values.
        @param labels: Flattened 1D tensor of shape (N,), containing the label for each point.
        @param alpha: Learnable weights for each label<
        """
        total_loss = 0
        unique_labels = torch.unique(labels)

        alpha_sigmoid = torch.sigmoid(alpha)

        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                mae = self.mae_loss(predictions[mask], ground_truth[mask]).mean()
                huber = self.huber_loss(predictions[mask], ground_truth[mask]).mean()
                # total_loss += alpha[label] * mae + (1 - alpha[label]) * huber
                total_loss += alpha_sigmoid[label] * mae + (1 - alpha_sigmoid[label]) * huber
        return total_loss

    def switching_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, labels: torch.Tensor, threshold: float=0.5):
        """
        Dynamically decide which loss to use for each label based on the gradient or the magnitude of the loss during training

        @param predictions: Flattened 1D tensor of shape (N,), containing the predicted intensity values.
	    @param ground_truth: Flattened 1D tensor of shape (N,), containing the actual intensity values.
        @param labels: Flattened 1D tensor of shape (N,), containing the label for each point.
        @param threshold: If the Huber loss for a label is below a certain threshold, switch to MAE, and vice versa
        """
        total_loss = 0
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                huber = self.huber_loss(predictions[mask], ground_truth[mask]).mean()
                if huber.item() < threshold:  # Use threshold to decide
                    total_loss += self.mae_loss(predictions[mask], ground_truth[mask]).mean()
                else:
                    total_loss += huber
        return total_loss

    def custom_combined_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, labels: torch.Tensor, label_preferences: dict):
        """
        Manually assign a preference for each label. These preferences are passed with 'label_preferences', which is
        a dictionary like this:
        ``` label_preferences = {0: 1, 1: 0}  # Example: label 0 -> MAE, label 1 -> Huber ```
        @param predictions:
        @param ground_truth:
        @param labels:
        @param label_preferences:
        @return:
        """
        total_loss = 0
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                if label_preferences[label.item()] == 1:
                    total_loss += self.mae_loss(predictions[mask], ground_truth[mask]).mean()
                else:
                    total_loss += self.huber_loss(predictions[mask], ground_truth[mask]).mean()
        return total_loss