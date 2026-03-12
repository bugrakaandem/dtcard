import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 1 == 0
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        self.conv = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            groups=self.channels,
            bias=True,
        )
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        pad_left = self.kernel_size - 1
        x = F.pad(x, (pad_left, 0))
        x = self.conv(x)
        x = self.drop(x)
        return x.transpose(1, 2)


class DCBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, conv_kernel: int = 3):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mix_r = CausalDepthwiseConv1d(hidden_size, kernel_size=conv_kernel, dropout=dropout)
        self.mix_s = CausalDepthwiseConv1d(hidden_size, kernel_size=conv_kernel, dropout=dropout)
        self.mix_a = CausalDepthwiseConv1d(hidden_size, kernel_size=conv_kernel, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def _token_mixer(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        assert T % 3 == 0, "Expected sequence length to be multiple of 3 (R,S,A interleave)."
        K = T // 3

        r = x[:, 0::3, :]  # (B,K,H)
        s = x[:, 1::3, :]
        a = x[:, 2::3, :]

        r = self.mix_r(r)
        s = self.mix_s(s)
        a = self.mix_a(a)

        y = torch.stack([r, s, a], dim=2).reshape(B, 3 * K, H)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._token_mixer(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionConvformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        max_length: int = 20,
        max_ep_len: int = 100,
        n_layer: int = 3,
        conv_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.hidden_size = int(hidden_size)
        self.max_length = int(max_length)

        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = nn.Embedding(self.act_dim, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_timestep = nn.Embedding(int(max_ep_len), self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.blocks = nn.Sequential(*[
            DCBlock(self.hidden_size, dropout=float(dropout), conv_kernel=int(conv_kernel))
            for _ in range(int(n_layer))
        ])
        self.ln_f = nn.LayerNorm(self.hidden_size)

        self.predict_action = nn.Linear(self.hidden_size, self.act_dim)

    def forward(self, states, actions, returns, timesteps):
        B, K, _ = states.shape

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        return_embeddings = return_embeddings + time_embeddings

        x = torch.stack((return_embeddings, state_embeddings, action_embeddings), dim=2)
        x = x.reshape(B, 3 * K, self.hidden_size)

        x = self.embed_ln(x)
        x = self.blocks(x)
        x = self.ln_f(x)

        x = x.reshape(B, K, 3, self.hidden_size)
        state_reps = x[:, :, 1, :]

        action_logits = self.predict_action(state_reps)
        return action_logits
