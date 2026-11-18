from dataclasses import dataclass

import torch
import torch.nn as nn

from rqf_ssl._core.embeddings.base import EmbeddingBase


@dataclass(frozen=True)
class FeatureSpec:
    width: int
    is_complex: bool


class BaseRQFEncoder(EmbeddingBase):
    _block_class: type[nn.Module]

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int | list[int],
        num_layers: int,
        norm_type: str = "none",
        filter_scales: list[dict] | list[list[dict]] | None = None,
    ) -> None:
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.norm_type = norm_type
        self._use_batch_norm = norm_type == "batch"

        if isinstance(hidden_dims, int):
            self._layer_dims: list[int] = [hidden_dims] * num_layers
        else:
            if len(hidden_dims) != num_layers:
                raise ValueError(
                    f"hidden_dims list must have {num_layers} entries, got {len(hidden_dims)}"
                )
            self._layer_dims = list(hidden_dims)

        self._layer_filter_scales = self._resolve_filter_scales(
            filter_scales, num_layers
        )

        self.input_proj = nn.Linear(input_dims, self._layer_dims[0])

        self.blocks = self._create_blocks(self._layer_filter_scales)

        if norm_type == "none":
            self.final_norm: nn.Module | None = None
        elif norm_type == "batch":
            self.final_norm = nn.BatchNorm1d(self._layer_dims[-1])
        else:
            self.final_norm = nn.LayerNorm(self._layer_dims[-1])
        if self.final_norm is not None and self._is_complex_output():
            raise ValueError(
                "complex-output encoders require norm_type='none' "
                "(a real final norm cannot apply to complex features)"
            )
        self._feature_width = self._layer_dims[-1]

    @property
    def feature_spec(self) -> FeatureSpec:

        return FeatureSpec(
            width=self._feature_width, is_complex=self._is_complex_output()
        )

    def _is_complex_output(self) -> bool:

        return False

    @staticmethod
    def _resolve_filter_scales(
        filter_scales: list[dict] | list[list[dict]] | None,
        num_layers: int,
    ) -> list[list[dict] | None]:

        if (
            filter_scales is not None
            and len(filter_scales) > 0
            and isinstance(filter_scales[0], list)
        ):
            if len(filter_scales) != num_layers:
                raise ValueError(
                    f"per-layer filter_scales must have {num_layers} entries, "
                    f"got {len(filter_scales)}"
                )
            return filter_scales

        return [filter_scales for _ in range(num_layers)]

    def _resolve_filter_factors(self, num_layers: int) -> list[float] | None:

        base = self._block_kwargs.get("filter_max_freq_factor")
        if base is None:
            return None
        if isinstance(base, list | tuple):
            if len(base) != num_layers:
                raise ValueError(
                    f"filter_max_freq_factor list length ({len(base)}) must match "
                    f"num_layers ({num_layers})"
                )
            return [float(f) for f in base]
        return [float(base) / (2**i) for i in range(num_layers)]

    def _create_blocks(
        self, layer_filter_scales: list[list[dict] | None]
    ) -> nn.ModuleList:

        factors = self._resolve_filter_factors(self.num_layers)
        blocks = []
        for i in range(self.num_layers):
            d_out = self._layer_dims[i + 1] if i < self.num_layers - 1 else None
            block_kwargs = dict(self._block_kwargs)
            if factors is not None:
                block_kwargs["filter_max_freq_factor"] = factors[i]
            block_kwargs = self._block_kwargs_for_layer(i, block_kwargs)
            blocks.append(
                self._block_class(
                    d_model=self._layer_dims[i],
                    d_model_out=d_out,
                    filter_scales=layer_filter_scales[i],
                    **block_kwargs,
                )
            )
        return nn.ModuleList(blocks)

    def _block_kwargs_for_layer(
        self,
        layer_idx: int,
        block_kwargs: dict[str, object],
    ) -> dict[str, object]:

        return block_kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        if self.final_norm is not None:
            if self._use_batch_norm:
                x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.final_norm(x)

        return x
