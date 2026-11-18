from abc import ABC, abstractmethod

import torch.nn as nn


class EmbeddingBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):

        raise NotImplementedError(
            "Subclasses must implement the forward() method to define their "
            "embedding transformation. The forward method should:\n"
            "  1. Accept input tensor(s) with appropriate shape\n"
            "  2. Apply the embedding transformation (e.g., learned filters, "
            "projections, etc.)\n"
            "  3. Return the embedded representation\n\n"
            "For sequential data, inputs typically have shape "
            "(batch_size, sequence_length, feature_dim)."
        )
