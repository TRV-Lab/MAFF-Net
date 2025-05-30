import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformerLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Linear transformations for Q, K, V
        self.linear_q = nn.Linear(input_dim, output_dim * num_heads)
        self.linear_k = nn.Linear(input_dim, output_dim * num_heads)
        self.linear_v = nn.Linear(input_dim, output_dim * num_heads)

        # Output linear transformation
        self.linear_out = nn.Linear(output_dim * num_heads, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Assume x is of shape [batch_size, num_points, input_dim]
        q = self.linear_q(x)  # [batch_size, num_points, output_dim * num_heads]
        k = self.linear_k(x)  # [batch_size, num_points, output_dim * num_heads]
        v = self.linear_v(x)  # [batch_size, num_points, output_dim * num_heads]

        q = q.view(q.size(0), q.size(1), self.num_heads,
                   self.output_dim)  # [batch_size, num_points, num_heads, output_dim]
        k = k.view(k.size(0), k.size(1), self.num_heads,
                   self.output_dim)  # [batch_size, num_points, num_heads, output_dim]
        v = v.view(v.size(0), v.size(1), self.num_heads,
                   self.output_dim)  # [batch_size, num_points, num_heads, output_dim]

        q = q.transpose(1, 2)  # [batch_size, num_heads, num_points, output_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, num_points, output_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, num_points, output_dim]

        # Attention mechanism
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
                    self.output_dim ** 0.5)  # [batch_size, num_heads, num_points, num_points]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, num_points, num_points]

        # Apply dropout
        attn_probs = self.dropout(attn_probs)

        # Weighted sum using attention scores
        weighted_sum = torch.matmul(attn_probs, v)  # [batch_size, num_heads, num_points, output_dim]

        # Reshape and concatenate heads
        weighted_sum = weighted_sum.transpose(1, 2).contiguous().view(x.size(0), x.size(1),
                                                                      -1)  # [batch_size, num_points, output_dim * num_heads]

        # Output linear transformation
        output = self.linear_out(weighted_sum)  # [batch_size, num_points, output_dim]

        # Residual connection and layer normalization
        output = self.layer_norm1(output + x)

        return output


class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=1, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Graph transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(input_dim if i == 0 else hidden_dim, hidden_dim, num_heads, dropout)
            for i in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Apply graph transformer layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]

        # Output layer
        output = self.output_layer(x)  # [batch_size, output_dim]

        return output


# Example usage
input_dim = 64
hidden_dim = 64
output_dim = 10
num_layers = 2
num_heads = 4
dropout = 0.1

# Create the model
model = GraphTransformer(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)

# Example input
batch_size = 32
num_points = 20
x = torch.randn(batch_size, num_points, input_dim)

# Forward pass
output = model(x)
print(output.shape)  # Output shape: [batch_size, output_dim]
