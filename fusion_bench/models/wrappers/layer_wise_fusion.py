import functools
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional  # noqa: F401

import torch
from torch import Tensor, nn
from torch.func import functional_call

from fusion_bench.utils.type import StateDictType

__all__ = ["get_layer_wise_weights", "fuse_weights", "LayerWiseMergedModel"]

log = logging.getLogger(__name__)


def del_attr(obj, names: List[str]):
    """
    Deletes an attribute from an object recursively.

    Args:
        obj (object): Object to delete attribute from.
        names (list): List of attribute names to delete recursively.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def get_attr(obj, names: List[str]):
    """
    Gets an attribute of an object recursively.

    Args:
        obj (object): Object to get attribute of.
        names (list): List of attribute names to get recursively.

    Returns:
        object: The attribute of the object.
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def get_layer_wise_weights(num_models: int, num_layers: int, init_values: float = None):
    """
    Return a tensor of layer-wise weights for the given number of models and layers.

    Args:
        num_models (int): The number of models to fuse.
        num_layers (int): The number of layers in each model.
        init_values (float, optional): The initial value for each weight. Defaults to 1.0 / num_models.

    Returns:
        Tensor: A tensor of shape (num_models, num_layers) containing the layer-wise weights.
    """
    assert num_models >= 1, f"num_models must be >= 1, got {num_models}"
    assert num_layers >= 1, f"num_layers must be >= 1, got {num_layers}"
    if init_values is None:
        init_values = 1.0 / num_models
    return torch.full((num_models, num_layers), init_values, dtype=torch.float32)


def _fuse_weights(layer_wise_weight: Tensor, tensors: List[Tensor]):
    """
    Fuse the layer-wise weights with the given state dictionaries.

    Args:
        layer_wise_weight (Tensor): A tensor of shape (num_models,) containing the layer-wise weights.
        state_dicts (List[Tensor]): A list of state dictionaries, each containing the weights for a single layer.

    Returns:
        Tensor: A tensor of shape (num_params,) containing the fused weights.
    """
    assert len(layer_wise_weight) == len(
        tensors
    ), f"layer_wise_weight.shape={layer_wise_weight.shape}, len(tensors)={len(tensors)}"
    return sum(
        layer_wise_weight[i] * w.to(layer_wise_weight.device)
        for i, w in enumerate(tensors)
    )


def fuse_weights(
    layer_wise_weight: Tensor, state_dicts: List[StateDictType]
) -> StateDictType:
    """
    Fuse the weights of multiple models using layer-wise fusion.

    Args:
        layer_wise_weight (Tensor): A tensor of shape (num_models, num_layers) representing the weight of each layer for each model.
        state_dicts (List[StateDict]): A list of state dictionaries, one for each model.

    Returns:
        A dictionary mapping each weight tensor key to the fused weight tensor.
    """
    num_models = len(state_dicts)
    num_layers = len(state_dicts[0])
    assert layer_wise_weight.shape == (
        num_models,
        num_layers,
    ), f"layer_wise_weight.shape={layer_wise_weight.shape}, expected (num_models, num_layers): ({num_models}, {num_layers})"
    return {
        k: _fuse_weights(
            layer_wise_weight[:, i], [state_dict[k] for state_dict in state_dicts]
        )
        for i, k in enumerate(state_dicts[0].keys())
    }


class LayerWiseMergedModel(nn.Module):
    _merged_state_dict: StateDictType = None

    def __init__(
        self,
        layer_wise_weight: Tensor,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        clamp_weights: bool = True,
        tie_weights: bool = False,
        strict: bool = True,
    ):
        super().__init__()
        self.clamp_weights = clamp_weights
        self.tie_weights = tie_weights
        self.strict = strict

        self.merge_weight = nn.Parameter(layer_wise_weight, requires_grad=True)

        for name, param in pretrained_model.named_parameters():
            if not param.requires_grad:
                for m in finetuned_models:
                    del_attr(m, name.split("."))
            else:
                for m in finetuned_models:
                    get_attr(m, name.split(".")).data = (
                        get_attr(m, name.split(".")) - param
                    )
        self.pretrained_model = pretrained_model.requires_grad_(False)
        for m in finetuned_models:
            m.requires_grad_(False)
        self.task_vectors = nn.ModuleList(finetuned_models)

    @property
    def forward_model(self):
        return functools.partial(
            functional_call,
            self.pretrained_model,
            self._merged_state_dict,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )

    def merge_and_unload(self, task_vector_mask: Optional[Dict[str, Tensor]] = None):
        self.merge_weights(task_vector_mask=task_vector_mask)
        self.pretrained_model.load_state_dict(self._merged_state_dict)
        return self.pretrained_model

    def merge_weights(self, task_vector_mask: Optional[Dict[str, Tensor]] = None):
        """
        Merges the weights of the model.
        Call this after each update step.
        """
        if self.clamp_weights:
            layer_wise_weight = self.merge_weight.clamp(0, 1)
        else:
            layer_wise_weight = self.merge_weight

        state_dict = self.pretrained_model.state_dict(keep_vars=True)
        # shape of layer_wise_weight: (num_models, num_layers)
        for weight, task_vector in zip(layer_wise_weight, self.task_vectors):
            assert len(list(task_vector.named_parameters())) == weight.size(0)
            if task_vector_mask is not None:
                weight = [
                    w * task_vector_mask[name]
                    for w, (name, param) in zip(weight, task_vector.named_parameters())
                ]
            for w, (name, param) in zip(weight, task_vector.named_parameters()):
                state_dict[name] = state_dict[name] + param * w
        self._merged_state_dict = state_dict

    def forward(self, *args, **kwargs):
        if self._merged_state_dict is None:
            self.merge_weights()
        return self.forward_model(args=args, kwargs=kwargs)

    # def __getattr__(self, name: str) -> Any:
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         attr = getattr(self.model, name)
    #         if isinstance(attr, Callable):
    #             warnings.warn(
    #                 f"forwarding `{name}` to the underlying model", UserWarning
    #             )
    #         return attr

    # def __setattr__(self, name: str, value: Any) -> None:
    #     try:
    #         super().__setattr__(name, value)
    #     except AttributeError:
    #         setattr(self.model, name, value)
