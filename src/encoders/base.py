from __future__ import annotations

import abc

import torch


class EncoderBase(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract encoder interface.

    forward(scene_pcd: Tensor[B,N,3], context_vec: Tensor[B,C]) -> Tensor[B,E]
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, scene_pcd: torch.Tensor, context_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
