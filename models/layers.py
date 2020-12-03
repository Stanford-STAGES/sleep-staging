import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters, kernel_size, strided=False):
        super().__init__()
        self.filters_in = n_filters_in
        self.filters = n_filters
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = 2 if strided else 1
        self.net = nn.Sequential(
            nn.BatchNorm1d(self.filters_in),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters_in, out_channels=self.filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                bias=False,
            ),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters, out_channels=self.filters * 4, kernel_size=1, bias=False),
        )
        self.projection = nn.Conv1d(
            in_channels=self.filters_in,
            out_channels=self.filters * 4,
            kernel_size=self.stride,
            stride=self.stride,
            bias=False,
        )

    def forward(self, x):
        z = self.net(x)
        # print(x.shape)
        return self.projection(x) + z


class ResidualBlock(nn.Module):
    # TODO: Fix this
    def __init__(self, n_filters_in, n_filters_out, kernel_size, strided=False, projection_type="identity"):
        super().__init__()
        self.filters_in = n_filters_in
        self.filters_out = n_filters_out
        self.kernel_size = kernel_size
        self.projection_type = projection_type
        self.padding = kernel_size // 2
        self.stride = 2 if strided else 1

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=self.filters_in,
                out_channels=self.filters_out,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,  # We stride on the first conv operator
                bias=False,
            ),
            nn.BatchNorm1d(self.filters_out),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.filters_out,
                out_channels=self.filters_out,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=False,
            ),
            nn.BatchNorm1d(self.filters_out),
        )
        if self.projection_type == "identity":
            self.maxpool = nn.MaxPool1d(self.stride, self.stride)
        elif self.projection_type == "projection":
            self.projection = nn.Conv1d(
                in_channels=self.filters_in,
                out_channels=self.filters_out,
                kernel_size=self.stride,
                stride=self.stride,
                bias=False,
            )

    def forward(self, x):
        z = self.net(x)
        # print(x.shape)
        if self.filters_in != self.filters_out:
            if self.projection_type == "projection":
                shortcut = self.projection(x)
            elif self.projection_type == "identity":
                top = (self.filters_out - self.filters_in) // 2
                bottom = (self.filters_out - self.filters_in) - top
                shortcut = F.pad(self.maxpool(x), (0, 0, top, bottom))
        else:
            shortcut = x
        return F.relu(z + shortcut)


class SimpleBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters, kernel_size, strided=False):
        super().__init__()
        self.filters_in = n_filters_in
        self.filters = n_filters
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = 2 if strided else 1

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.filters_in,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            ),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.block(x)
        return z


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
    blocks = nn.Sequential(
        ResidualBlock(2, 4, 3, strided=True, projection_type="identity"),
        ResidualBlock(4, 4, 3, strided=False, projection_type="identity"),
        ResidualBlock(4, 8, 3, strided=True, projection_type="identity"),
        ResidualBlock(8, 8, 3, strided=False, projection_type="identity"),
    )
    z = blocks(x)
    print(z.shape)

    # ResBlock = ResidualBlock(5, 8, 3, strided=False, projection_type="identity")
    # z = ResBlock(x)

    # ResBlock = ResidualBlock(5, 8, 9, strided=True, projection_type="identity")
    # z = ResBlock(x)
    # print(z.shape)
    # print("Hej")
    # att = AdditiveAttention(1024, 256)
    # c, alpha = att(x, 30)
    # print(c)
    # print(alpha)
    # print(c.shape)
    # print(alpha.shape)
