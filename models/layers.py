import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """See https://arxiv.org/pdf/1809.10932.pdf"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.compute_alphas = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Softmax(dim=-1),
        )

    def forward(self, h, segment_length):
        # h.size() = [Batch size, Sequence length, Input size]
        b, n, f = h.shape
        h = h.permute(0, 2, 1).unfold(-1, segment_length, segment_length).permute(0, 2, 3, 1)
        alphas = self.compute_alphas(h)
        c = (alphas.unsqueeze(1) * h.permute(0, 3, 1, 2)).sum(dim=-1)

        return c, alphas


if __name__ == "__main__":
    x = torch.rand([32, 2, 5 * 60 * 128])
    # att = AdditiveAttention(1024, 256)
    # c, alpha = att(x, 30)
    # print(c)
    # print(alpha)
    # print(c.shape)
    # print(alpha.shape)
