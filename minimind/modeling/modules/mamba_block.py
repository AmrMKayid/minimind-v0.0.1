import flax.linen as nn
import jax.numpy as jnp
from einops import einsum


class MambaBlock(nn.Module):
    """MambaBlock is a custom neural network block that incorporates
    normalization, convolution, and dense layers to process input sequences.

    This block is designed for sequence modeling tasks and includes
    specialized components like selective scan for dynamic computation.
    """

    d_inner: int
    d_conv: int
    dt_rank: int
    d_state: int
    d_model: int
    seq_len: int
    bias: bool
    conv_bias: bool

    def setup(self):
        self.norm = nn.RMSNorm(self.d_model)
        self.in_proj = nn.Dense(features=self.d_inner * 2, use_bias=self.bias)

        self.conv1d = nn.Conv(
            features=self.seq_len,
            kernel_size=(self.d_conv,),
            strides=(1,),
            padding="SAME",
            use_bias=self.conv_bias,
            feature_group_count=self.d_inner,
        )

        self.x_proj = nn.Dense(features=self.dt_rank + self.d_state * 2, use_bias=False)
        self.dt_proj = nn.Dense(features=self.d_inner, use_bias=True)
        self.out_proj = nn.Dense(features=self.d_model, use_bias=self.bias)

        # Parameter initialization
        A = jnp.tile(jnp.arange(1, self.d_state + 1), (self.d_inner, 1))
        self.A_log = self.variable("params", "A_log", lambda: jnp.log(A))
        self.D = self.variable("params", "D", lambda: jnp.ones((self.d_inner,)))

    def __call__(self, inputs: jnp.ndarray):
        u = self.norm(inputs)
        A = -jnp.exp(self.A_log.value)
        D = self.D.value
        x_and_res = self.in_proj(u)
        x, res = jnp.split(x_and_res, 2, axis=-1)
        x = jnp.transpose(x, (0, 2, 1))
        x = self.conv1d(x)[:, :, : u.shape[1]]
        x = jnp.transpose(x, (0, 2, 1))
        x = nn.silu(x)

        x_dbl = self.x_proj(u)
        delta, B, C = jnp.split(x_dbl, indices_or_sections=[self.dt_rank, self.dt_rank + self.d_state], axis=-1)
        delta = nn.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        y = y * nn.silu(res)
        return self.out_proj(y) + inputs

    def selective_scan(
        self, u: jnp.ndarray, delta: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, D: jnp.ndarray
    ) -> jnp.ndarray:
        b, l, d_in = u.shape
        n = A.shape[1]

        deltaA = jnp.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))

        deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

        x = jnp.zeros((b, d_in, n))
        ys = []

        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)

        return jnp.stack(ys, axis=1) + u * D
