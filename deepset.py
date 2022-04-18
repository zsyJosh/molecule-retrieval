import torch

class DeepSet(torch.nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, dim=-2):
        # x has input shape [batch_size, num_elements, d]
        x = self.encoder(x).mean(dim)  # aggr over elements [batch_size, d]
        x = self.decoder(x) # output shape [batch_size, d]
        return x