import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.kernel_size = 5

        self.conv = nn.Conv1d(
            in_channels=config.n_embd,
            out_channels=config.n_embd,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=config.n_embd
        )

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_dropout),
        )

        self.dropout = nn.Dropout(config.attn_dropout)

    def forward(self, x):
        residual = x
        x = self.ln1(x)

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class DecisionConvFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = 'reward_conditioned'

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_dropout)

        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.n_embd),
            nn.Tanh(),
        )

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size, config.n_embd),
            nn.Tanh()
        )

        self.blocks = nn.Sequential(*[ConvBlock(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate,
                                      betas=(train_config.optimizer_beta1, train_config.optimizer_beta2))
        return optimizer

    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        state_embeddings = self.state_encoder(states)
        rtg_embeddings = self.ret_emb(rtgs.float())

        if actions is not None:
            action_embeddings = self.action_embeddings(actions.long().squeeze(-1))
        else:
            action_embeddings = torch.zeros_like(state_embeddings)

        batch_size, seq_len, _ = state_embeddings.shape
        token_embeddings = torch.zeros(
            (batch_size, seq_len * 3, self.config.n_embd),
            device=states.device
        )

        token_embeddings[:, 0::3, :] = rtg_embeddings
        token_embeddings[:, 1::3, :] = state_embeddings

        if actions is not None:
            token_embeddings[:, 2::3, :] = action_embeddings

        x = self.drop(token_embeddings)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        logits = logits[:, 1::3, :]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss