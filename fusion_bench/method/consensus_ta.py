"""
This script contains the general implementation of the Task Arithmetic method.

http://arxiv.org/abs/2212.04089
"""

import logging
from copy import deepcopy
from collections import OrderedDict
import copy

from typing import Dict, List, Mapping, TypeVar, Union  # noqa: F401

import torch
from torch import nn, Tensor

from fusion_bench.method.base_algorithm import BaseModelFusionAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)


@torch.no_grad()
def task_arithmetic_merge(
    pretrained_model: nn.Module,
    finetuned_models: List[nn.Module],
    scaling_factor: float,
    inplace: bool = True,
) -> nn.Module:
    """
    Merges the task vectors from multiple fine-tuned models into a single pre-trained model.

    Args:
        pretrained_model (nn.Module): The pre-trained model to which the task vectors will be added.
        finetuned_models (List[nn.Module]): A list of fine-tuned models from which task vectors will be calculated.
        scaling_factor (float): A factor by which the task vectors will be scaled before merging.
        inplace (bool, optional): If True, the pre-trained model will be modified in place.
                                  If False, a copy of the pre-trained model will be modified. Defaults to True.

    Returns:
        nn.Module: The pre-trained model with the merged task vectors.
    """
    if not inplace:
        pretrained_model = deepcopy(pretrained_model)
    task_vector: StateDictType = None
    # Calculate the total task vector
    for model in finetuned_models:
        if task_vector is None:
            task_vector = state_dict_sub(
                model.state_dict(keep_vars=True),
                pretrained_model.state_dict(keep_vars=True),
            )
        else:
            task_vector = state_dict_add(
                task_vector,
                state_dict_sub(
                    model.state_dict(keep_vars=True),
                    pretrained_model.state_dict(keep_vars=True),
                ),
            )
    # scale the task vector
    task_vector = state_dict_mul(task_vector, scaling_factor)
    # add the task vector to the pretrained model
    state_dict = state_dict_add(
        pretrained_model.state_dict(keep_vars=True), task_vector
    )
    pretrained_model.load_state_dict(state_dict)
    return pretrained_model

# Model conversion utils
def state_dict_to_vector(state_dict, remove_keys=[]):
    """
    Convert a state dictionary to a vector, removing specified keys.

    Args:
        state_dict (dict): The state dictionary to convert.
        remove_keys (list): List of keys to remove from the state dictionary.

    Returns:
        Tensor: A vector representation of the state dictionary.
    """
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    """
    Convert a vector back to a state dictionary, removing specified keys.

    Args:
        vector (Tensor): The vector to convert.
        state_dict (dict): The reference state dictionary.
        remove_keys (list): List of keys to remove from the state dictionary.

    Returns:
        dict: A state dictionary representation of the vector.
    """
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the reference dict
    nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

def get_tall_mask(tv_flat, lambda_factor, merged_tv):
    mask = tv_flat.abs() > lambda_factor * (merged_tv - tv_flat).abs()
    log.info(f"Average sparsity for task specific mask is: {mask.float().mean()}")
    return mask

class ConsensusTAAlgorithm(
    BaseModelFusionAlgorithm,
    SimpleProfilerMixin,
):
    """
    Task Arithmetic Algorithm for model fusion.

    This class implements the Task Arithmetic method for fusing models. It inherits from
    BaseModelFusionAlgorithm and SimpleProfilerMixin to provide the necessary functionality
    for model fusion and profiling.

    Attributes:
        scaling_factor (int): The factor by which the task vectors will be scaled before merging.
    """

    _config_mapping = BaseModelFusionAlgorithm._config_mapping | {
        "scaling_factor": "scaling_factor",
        "lambda_": "lambda_",
        "k": "k"
    }

    def __init__(self, scaling_factor: int, lambda_:float, k: int):
        """
        Initializes the TaskArithmeticAlgorithm with the given scaling factor.

        Args:
            scaling_factor (int): The factor by which the task vectors will be scaled before merging.
        """
        self.scaling_factor = scaling_factor
        self.lambda_ = lambda_
        self.k = k
        super().__init__()

    @torch.no_grad()
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        log.info("Fusing models using consensus ta")
        lambad_factor = self.lambda_
        k = self.k
        
        remove_keys = self.config.get("remove_keys", [])
        
        pretrained_model = modelpool.load_model("_pretrained_")

        # Load the state dicts of the models
        ft_checks: List[StateDictType] = [
            modelpool.load_model(model_name).state_dict(keep_vars=True)
            for model_name in modelpool.model_names
        ]
        ptm_check: StateDictType = pretrained_model.state_dict(keep_vars=True)

        # Compute the task vectors
        flat_ft: Tensor = torch.vstack(
            [state_dict_to_vector(check, remove_keys) for check in ft_checks]
        )
        flat_ptm: Tensor = state_dict_to_vector(ptm_check, remove_keys)
        tv_flat_checks = flat_ft - flat_ptm
        
        merged_tv = torch.sum(tv_flat_checks, dim=0)
        # merged_check = flat_ptm + self.scaling_factor * merged_tv
        
        tall_masks = torch.vstack([get_tall_mask(tv_flat, lambad_factor, merged_tv) for tv_flat in tv_flat_checks])
        consensus_mask = torch.sum(tall_masks, dim=0) >= k
        log.info(f"Average sparsity for the consensus mask is {consensus_mask.float().mean()}")
        
        merged_tv = torch.where(consensus_mask, merged_tv, torch.zeros_like(merged_tv))
        merged_check = flat_ptm + self.scaling_factor * merged_tv
        
        merged_state_dict = vector_to_state_dict(
            merged_check, ptm_check, remove_keys=remove_keys
        )
        pretrained_model.load_state_dict(merged_state_dict)

        # self.print_profile_summary()
        # pretrained_model.load_state_dict(state_dict)
        return pretrained_model
