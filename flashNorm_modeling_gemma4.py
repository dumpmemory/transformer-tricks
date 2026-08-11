# FlashNorm strict modeling for Gemma 4 (transformer-tricks)
# Folded norm weights are removed from the state dict; additionally the pre-attention
# RMSNorm is cancelled entirely (Proposition 3 of arXiv:2407.09577), which is exact up
# to epsilon effects because Gemma 4 re-normalizes queries, keys, and values per head.
import torch
import torch.nn as nn
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


class WeightlessRMSNorm(nn.Module):
  """Gemma-4-style RMSNorm without a gain (the gain is folded into the next linear)."""

  def __init__(self, eps):
    super().__init__()
    self.eps = eps

  def forward(self, x):
    h = x.float()
    ms = h.pow(2).mean(-1, keepdim=True) + self.eps
    return (h * torch.pow(ms, -0.5)).type_as(x)


class Gemma4FlashNorm(Gemma4ForConditionalGeneration):
  """Gemma4ForConditionalGeneration with input_layernorm cancelled and pre_feedforward_layernorm weightless."""

  def __init__(self, config):
    super().__init__(config)
    layers = None
    for path in ('model.language_model.layers', 'model.layers', 'language_model.model.layers'):
      obj = self
      try:
        for part in path.split('.'):
          obj = getattr(obj, part)
        layers = obj
        break
      except AttributeError:
        continue
    assert layers is not None, 'could not locate decoder layers'
    for ly in layers:
      eps = ly.pre_feedforward_layernorm.eps
      ly.input_layernorm = nn.Identity()
      ly.pre_feedforward_layernorm = WeightlessRMSNorm(eps)
