import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_ctx, drop_p):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(drop_p)
        self.resid_dropout = nn.Dropout(drop_p)
        self.n_head = n_head
        self.n_embd = n_embd

        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx))
                             .view(1, 1, n_ctx, n_ctx))

    def forward(self, x):
        B, T, C = x.size()  # Batch, Time, Channel (n_embd)

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_ctx, drop_p):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, n_ctx, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, max_length=20, max_ep_len=100, n_layer=3, n_head=4, n_inner=4,
                 dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Embedding(act_dim, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.blocks = nn.Sequential(*[
            Block(hidden_size, n_head, n_ctx=max_length * 3, drop_p=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Linear(hidden_size, act_dim)

    def forward(self, states, actions, returns, timesteps):
        B, K, _ = states.shape

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        return_embeddings = return_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(B, 3 * K, self.hidden_size)

        x = self.embed_ln(stacked_inputs)
        x = self.blocks(x)
        x = self.ln_f(x)

        x = x.reshape(B, K, 3, self.hidden_size)

        state_reps = x[:, :, 1, :]

        action_logits = self.predict_action(state_reps)

        return action_logits