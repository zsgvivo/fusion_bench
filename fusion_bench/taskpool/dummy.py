"""
This is the dummy task pool that is used for debugging purposes.
"""

from typing import Optional

from fusion_bench.models.separate_io import separate_save
from fusion_bench.taskpool.base_pool import BaseTaskPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.parameters import count_parameters, print_parameters
from fusion_bench.mixins import LightningFabricMixin
import lm_eval
from lm_eval.models.vllm_causallms import VLLM
from transformers import AutoTokenizer
import logging
import os
import evalplus
import json
log = logging.getLogger(__name__)

class DummyTaskPool(BaseTaskPool, LightningFabricMixin):
    """
    This is a dummy task pool used for debugging purposes. It inherits from the base TaskPool class.
    """

    def __init__(self,tokenizer: Optional[str], model_save_path: Optional[str] = None):
        super().__init__()
        self.model_save_path = model_save_path
        self.tokenizer = tokenizer

    def evaluate(self, model):
        """
        Evaluate the given model.
        This method does nothing but print the parameters of the model in a human-readable format.

        Args:
            model: The model to evaluate.
        """
        print_parameters(model, is_human_readable=True)

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                separate_save(model, self.model_save_path)

        report = {}
        training_params, all_params = count_parameters(model)
        report["model_info"] = {
            "trainable_params": training_params,
            "all_params": all_params,
            "trainable_percentage": training_params / all_params,
        }

        cache_dir = '/mnt/data/dy24/model_cache'
        log.info(f"Saving merged model to {cache_dir}")
        model.save_pretrained(cache_dir)
        log.info(f"Saving tokenizer {self.tokenizer} to {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        tokenizer.save_pretrained(cache_dir)
        
        # evaluating code
        # evalplus.evaluate(dataset='humaneval', model=cache_dir, backend='vllm', greedy=True)
        os.system(f"evalplus.evaluate --model {cache_dir} \
                  --dataset humaneval                   \
                  --backend vllm                         \
                  --greedy")
        # read the newest json file from evalplus_results/
        files = os.listdir('evalplus_results/humaneval/')
        files = [os.path.join('evalplus_results/humaneval/', f) for f in files]
        files.sort(key=os.path.getmtime)
        files = [f for f in files if f.endswith('.json')]
        newest_file = files[-1]
        with open(newest_file) as f:
            code_res = json.load(f)
        code_res = {'humaneval': code_res['pass_at_k']}
        report.update(code_res)
        log.info(f"Results: {code_res}")
        
        
        vllm_model = VLLM(cache_dir, batch_size="auto")
        
        # task_dict = lm_eval.tasks.get_task_dict(
        #     [
        #         "gsm8k", # A stock task
        #     ],
        #     )

        # results = lm_eval.evaluate(
        #     lm=vllm_model,
        #     task_dict=task_dict,
        # )
        
        results = lm_eval.simple_evaluate(
            model=vllm_model,
            tasks=["gsm8k"],
            fewshot_as_multiturn=True,
            apply_chat_template=True,
        )
        report.update(results['results'])
        
        log.info(f"Results: {results['results']}")

        # clean up the model
        os.system(f"rm -r {cache_dir}")

        return report
