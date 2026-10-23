import torch
import torch.nn as nn


class _TemporalBlock(nn.Module):
    def __init__(self, local_channels, output_size):
        super().__init__()
        self.local_projection = nn.Sequential(
            nn.Conv1d(1, local_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(local_channels),
            # nn.GroupNorm(1, local_channels),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=local_channels,
            hidden_size=output_size,
            num_layers=1,
            batch_first=True,
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.local_projection(x)
        x = x.transpose(1, 2)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        return self.activation(x[:, -1, :])


class DynamicVariableIdentityAdapter(nn.Module):

    def __init__(self, identity_dim, feature_dim):
        super().__init__()
        self.identity_dim = identity_dim
        self.embeddings = nn.ParameterDict()
        self.conditioner = nn.Sequential(
            nn.Linear(identity_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2 * feature_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

        # Initially preserve the temporal representation exactly.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    @staticmethod
    def _key(variable_id):
        return f"id_{int(variable_id)}"

    def register_variables(self, variable_ids):
        """Create embeddings for newly observed IDs before building an optimizer."""
        ids = torch.as_tensor(variable_ids).detach().reshape(-1).cpu().tolist()
        if any(int(variable_id) < 0 for variable_id in ids):
            raise ValueError("Variable IDs must be non-negative integers.")

        device = self.conditioner[0].weight.device
        for variable_id in sorted(set(int(value) for value in ids)):
            key = self._key(variable_id)
            if key not in self.embeddings:
                embedding = torch.empty(1, self.identity_dim, device=device)
                nn.init.xavier_uniform_(embedding)
                self.embeddings[key] = nn.Parameter(embedding)

    def forward(self, features, variable_ids):
        batch_size, num_variables, _ = features.shape
        if variable_ids.ndim == 1:
            variable_ids = variable_ids.unsqueeze(0).expand(batch_size, -1)
        elif variable_ids.ndim != 2:
            raise ValueError(
                "variable_ids must have shape [C] or [B, C], but received "
                f"{tuple(variable_ids.shape)}"
            )
        if variable_ids.shape != (batch_size, num_variables):
            raise ValueError(
                "Expected one ID for every variable with shape "
                f"{(batch_size, num_variables)}, received {tuple(variable_ids.shape)}"
            )
        if variable_ids.numel() == 0:
            raise ValueError("At least one observed variable is required.")

        flat_ids = variable_ids.detach().reshape(-1).cpu().tolist()
        missing = [
            int(variable_id)
            for variable_id in flat_ids
            if self._key(variable_id) not in self.embeddings
        ]
        if missing:
            raise ValueError(
                f"Unregistered variable IDs {sorted(set(missing))}. Call "
                "register_variable_ids before constructing the optimizer."
            )
        identity = torch.cat(
            [self.embeddings[self._key(variable_id)] for variable_id in flat_ids],
            dim=0,
        ).reshape(batch_size, num_variables, self.identity_dim)
        gamma, beta = self.conditioner(identity).chunk(2, dim=-1)
        return self.norm((1.0 + gamma) * features + beta)


class FixedSlotProjection(nn.Module):
    """Project an arbitrary-size variable set into a fixed number of slots."""

    def __init__(self, feature_dim, num_slots, num_heads, ffn_dim, dropout):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.empty(num_slots, feature_dim))
        nn.init.xavier_uniform_(self.slot_queries)
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, feature_dim),
        )

    def forward(self, variable_features):
        batch_size = variable_features.size(0)
        queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        projected, attention = self.cross_attention(
            queries,
            variable_features,
            variable_features,
            need_weights=True,
            average_attn_weights=False,
        )
        slots = self.norm1(queries + self.dropout(projected))
        slots = self.norm2(slots + self.dropout(self.ffn(slots)))
        return slots, attention


class SlotClassifier(nn.Module):
    """Pool the fixed slots directly and classify the complete sample."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.pool_score = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, slots):
        pool_weights = torch.softmax(self.pool_score(slots), dim=1)
        representation = (pool_weights * slots).sum(dim=1)
        return self.classifier(representation), pool_weights


class TemporalEncoder(nn.Module):
    """Complete dynamic-variable time-series classifier."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        feature_dim = configs.temporal_size
        if feature_dim % configs.num_heads != 0:
            raise ValueError("temporal_size must be divisible by num_heads.")

        # One temporal encoder is shared by all observed variables.
        self.time = _TemporalBlock(
            configs.temporal_channels,
            feature_dim,
        )
        self.identity_adapter = DynamicVariableIdentityAdapter(
            configs.identity_dim,
            feature_dim,
        )
        self.slot_projection = FixedSlotProjection(
            feature_dim,
            configs.num_slots,
            configs.num_heads,
            configs.ffn_dim,
            configs.dropout,
        )
        self.slot_classifier = SlotClassifier(
            feature_dim,
            configs.num_class,
        )

    def register_variable_ids(self, variable_ids):
        self.identity_adapter.register_variables(variable_ids)

    def _to_bct(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"TemporalEncoder expects a 3-D tensor, but received {tuple(x.shape)}"
            )
        if x.size(1) == self.seq_len:
            return x.transpose(1, 2)
        if x.size(2) == self.seq_len:
            return x
        raise ValueError(
            "Cannot identify the temporal dimension: neither input dimension "
            f"matches seq_len={self.seq_len}; received {tuple(x.shape)}"
        )

    def encode_variables(self, x):
        """Encode each variable at the original resolution as ``[B, C, D]``."""
        x = self._to_bct(x)
        batch_size, num_variables, seq_len = x.shape
        x = x.reshape(batch_size * num_variables, 1, seq_len)
        encoded = self.time(x)
        return encoded.reshape(batch_size, num_variables, -1)

    def forward(self, x, variable_ids, return_aux=False):
        temporal_features = self.encode_variables(x)
        variable_features = self.identity_adapter(temporal_features, variable_ids)
        slots, variable_to_slot_attention = self.slot_projection(variable_features)
        logits, slot_pool_weights = self.slot_classifier(slots)

        if return_aux:
            return logits, {
                "temporal_features": temporal_features,
                "variable_features": variable_features,
                "slots": slots,
                "variable_to_slot_attention": variable_to_slot_attention,
                "slot_pool_weights": slot_pool_weights,
            }
        return logits
