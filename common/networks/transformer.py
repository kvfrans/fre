from typing import Any, Callable, Optional, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x):
        return x


class AddPositionEmbs(nn.Module):
    # Need to define function that adds the poisition embeddings to the input.
    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

    @nn.compact
    def __call__(self, inputs):
        """
            inputs.shape is (batch_size, timesteps, emb_dim).
            Output tensor with shape `(batch_size, timesteps, in_dim)`.
        """
        assert inputs.ndim == 3, ('Number of dimensions should be 3, but it is: %d' % inputs.ndim)

        position_ids = jnp.arange(inputs.shape[1])[None] # (1, timesteps)
        pos_embeddings = nn.Embed(
            128, # Max Positional Embeddings
            inputs.shape[2],
            embedding_init=self.posemb_init,
            dtype=inputs.dtype,
        )(position_ids)
        print("For Input Shape {}, Pos Embes Shape is {}".format(inputs.shape, pos_embeddings.shape))
        return inputs + pos_embeddings

        # pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        # pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
        # return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
                features=actual_out_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
        output = nn.Dropout(
                rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.
    Given a sequence, it passes it through an attention layer, then through a mlp layer.
    In each case it is a residual block with a layer norm.
    """

    mlp_dim: int
    num_heads: int
    causal: bool
    dropout_rate: float
    attention_dropout_rate: float
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, deterministic, train=True):

        if self.causal:
            causal_mask = nn.make_causal_mask(jnp.ones((inputs.shape[0], inputs.shape[1]),
                                                        dtype="bool"), dtype="bool")
            print("Using Causal Mask with shape", causal_mask.shape, "and inputs shape", inputs.shape, ".")
        else:
            causal_mask = None

        # Attention block.
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            decode=False,
            num_heads=self.num_heads)(x, x, causal_mask)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block. This does NOT change the embedding dimension!
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(y, deterministic=deterministic)

        return x + y


class Transformer(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.
    """

    num_layers: int
    emb_dim: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float
    attention_dropout_rate: float
    causal: bool = True

    @nn.compact
    def __call__(self, x, *, train):
        assert x.ndim == 3  # (batch, len, emb)
        assert x.shape[-1] == self.emb_dim

        # Input Encoder. Each layer processes x, but the shape of x does not change.
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                    mlp_dim=self.mlp_dim,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    name=f'encoderblock_{lyr}',
                    causal=self.causal,
                    num_heads=self.num_heads)(
                            x, deterministic=not train, train=train)
        encoded = nn.LayerNorm(name='encoder_norm')(x)

        return encoded
    
def get_default_config():
    import ml_collections

    config = ml_collections.ConfigDict({
        'num_layers': 4,
        'emb_dim': 256,
        'mlp_dim': 256,
        'num_heads': 4,
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
        'causal': True,
    })
    return config