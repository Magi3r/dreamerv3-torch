from torch.nn.functional import F
from torch import nn, Tensor

class MCDropout(nn.Dropout):
    """
    Randomly randomly zeroes some of the elements of the input tensor with probability 'p'.

    This happens during training and inference when enabled
    """
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p, inplace)
        self.enabled = True

    def disable(self) -> None:
        """
        Disables the dropout
        """
        self.enabled = False

    def enable(self) -> None:
        """
        Enables the dropout
        """
        self.enabled = True


    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        With 'training' always set True when enabled, so also dropping out during inference.
        """
        return F.dropout(input, self.p, self.enable, self.inplace)