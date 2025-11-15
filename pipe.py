import re
import time
from typing import Any, Dict, Optional, List, Text, Tuple, cast
import os
from pathlib import Path
import warnings
import wandb
import pandas as pd
import pdb
import numpy as np
import torch
import torch.utils
import torch.utils.data

import hydra
import logging
import colorlog
from omegaconf import OmegaConf, DictConfig
import datasets
from datasets import Dataset, load_dataset, Features, Value, ClassLabel, Sequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2TokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedModel,
    EvalPrediction,
    is_datasets_available,
)
from transformers.generation.configuration_utils import GenerationConfig
from trl import SFTTrainer, SFTConfig



from utils.dplm_utils.dplm.generate_dplm import generate
from utils.lf_utils.protein_tokenizer import DistMatrixTokenizer
from utils.openfold_utils import OpenfoldProtein
from utils.lf_utils import (
    DistMatrixTokenizer,
    DPLMProteinTokenizer,
    TextTokenizer,
    ProteinProcessor, 
    SortishApproxBatchDataloader,
    TextCollator,
    ExtraColumnCollator,
    DynamicMultimodalLogitsProcessor,
    DATASET_SPLIT, DATASET_RAW_ROOT,GT_STRUCT_ROOT
)

# log color whenrank=0 & silent when rank>0
rank = int(os.environ.get("RANK", 0))
logger = logging.getLogger(__name__)
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s" + f"[rank{rank}]" + "[%(asctime)s][%(levelname)s]" + " %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    }
))
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False




class SFTTrainerWithEval(SFTTrainer):
    
    def __init__(
        self,
        processor: ProteinProcessor,
        eval_collator: ExtraColumnCollator,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.eval_collator = eval_collator
    
    ## overide to support extra columns ##
    # - keep any extra columns
    # - batch-size = 1, since we have to constraint generation length
    
    def get_eval_dataloader(self, eval_dataset: Any = None) -> torch.utils.data.DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_collator
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader
        return self.accelerator.prepare(eval_dataloader)
    
    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func: Any = None,
        add_special_tokens=True,
        remove_unused_columns=False,
    ):
        
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element["text"] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation='prompt' not in element.keys(), # True for training; False for eval
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True
            # keep any other columns besides signature columns +++
            if 'prompt' in element.keys():
                outputs_prompt = tokenizer(
                    element["prompt"] if not use_formatting_func else formatting_func(element),
                    add_special_tokens=add_special_tokens,
                    truncation=False,
                    padding=False,
                    max_length=max_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                )
                outputs['labels'] = outputs["input_ids"] # complete label
                outputs['input_ids'] = outputs_prompt["input_ids"]
            else:
                outputs['labels'] = outputs["input_ids"]
                
            return {
                "input_ids":        outputs["input_ids"],
                "attention_mask":   outputs["attention_mask"],
                "labels":           outputs["labels"],
                **{k: element[k] for k in element.keys() if k not in outputs.keys()}
            }

        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )
        return tokenized_dataset
    
    @torch.no_grad()
    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        model.eval()
        tokenizer = self.processor.tokenizer
        
        # update generation config
        eval_max_new_tokens = inputs['struct_length'][0].item()
        eval_min_new_tokens = inputs['struct_length'][0].item()
        eval_config = GenerationConfig(
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            max_new_tokens=eval_max_new_tokens,
            min_new_tokens=eval_min_new_tokens,
        )

        generated_token_ids: torch.Tensor = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=eval_config,
        ) # type: ignore [B, L] where B = 1 for now
        target_token_ids = inputs['labels'][0]
        
        # for regex, append a '</struct>' token at the end if not present
        if not generated_token_ids[0].tolist()[-1] == tokenizer.eostruct_token_id:
            generated_token_ids = torch.cat([
                generated_token_ids,
                torch.tensor([[tokenizer.eostruct_token_id]], device=generated_token_ids.device)
            ], dim=1)
        
        generation_length = len(generated_token_ids[0])
        prompt_length = len(inputs['input_ids'][0])
        target_length = len(target_token_ids)
        if generation_length != target_length:
            logger.error(f"Generation length mismatch: generated {generation_length} vs target {target_length}, constraint {eval_min_new_tokens}")
        
        # decode generated tokens
        prompt_str = self.processor.tokenizer.decode(inputs['input_ids'][0])
        generated_str = self.processor.tokenizer.decode(generated_token_ids[0, prompt_length:])
        target_str = self.processor.tokenizer.decode(target_token_ids[prompt_length:])
        logger.info(f"=== Prompt ===\n{prompt_str}\n=== Generated ===\n{generated_str}\n=== Target ===\n{target_str}\n")
        
        # pattern = rf'^{re.escape(tokenizer.bostruct_token)}(({tokenizer.struct_regex})+){re.escape(tokenizer.eostruct_token)}$'
        # skip_ar = re.match(pattern, generated_str) is None


        # 假设 generated_str 这一整段里只有一对 <struct>...</struct>
        pattern = rf'{re.escape(tokenizer.bostruct_token)}(({tokenizer.struct_regex})+){re.escape(tokenizer.eostruct_token)}'
        m = re.search(
            pattern,
            generated_str,
        )
        if m:
            # struct_block = m.group(0)  # "<struct>...<|sXXXX|>...</struct>"
            # 再检查是否“刚好等于”这段（也可以不用再检查了，直接认为OK）
            skip_ar = False
        else:
            skip_ar = True



        # pdb.set_trace()
        if skip_ar:
            logger.warning(f"Generated structure string does not match the expected format {pattern}")
        
        # metrics include
        # - exposure loss
        # - <vq, nature> tm-score/rmsd-local/rmsd-global
        # - <ar, nature> tm-score/rmsd-local/rmsd-global
        metrics = {}
        
        # pdb_name = inputs['pdb_name'][0]
        # split = inputs['split'][0]
        gt_struct_path = inputs['gt_struct_path'][0]
        root, format = GT_STRUCT_ROOT, ".pdb"
        # metrics['split'] = DATASET_SPLIT[split] # due to transformer's param constraint
        

        
        exposure_loss: torch.Tensor = model(
            input_ids=inputs['labels'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'],
        ).loss
        metrics['exposure_loss'] = exposure_loss.cpu().item()
        
        device = inputs['input_ids'].device

        # pdb.set_trace()
        # p_nature = OpenfoldProtein.from_file(Path(root)/f"{pdb_name}{format}").to(device)
        p_nature = OpenfoldProtein.from_file(inputs['gt_struct_path'][0]).to(device)
        # pdb.set_trace()
        p_vq = self.processor.multimodal_decode(target_token_ids, ref=p_nature)['entity'][-1].to(device)
        
        # pdb.set_trace()
        tm_vq, rmsd_l_vq, rmsd_g_vq = self.processor.compute_tm_align(p_vq, p_nature, ref=p_nature)
        metrics['tm_vq'] = tm_vq
        metrics['rmsd_l_vq'] = rmsd_l_vq
        metrics['rmsd_g_vq'] = rmsd_g_vq

        if not skip_ar:
            p_ar = self.processor.multimodal_decode(generated_token_ids[0], ref=p_nature)['entity'][-1].to(device)
            tm_ar, rmsd_l_ar, rmsd_g_ar = self.processor.compute_tm_align(p_ar, p_nature, ref=p_nature)
        else:
            tm_ar, rmsd_l_ar, rmsd_g_ar = 0.0, 20.0, 20.0 # dummy large value
        metrics['tm_ar'] = tm_ar
        metrics['rmsd_l_ar'] = rmsd_l_ar
        metrics['rmsd_g_ar'] = rmsd_g_ar
        
        # Summary Here
#         logger.info(f"""Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}]:
# Exposure Loss:  {metrics['exposure_loss']:.4f}
# VQ v.s. Nature: TM-score = {metrics['tm_vq']:.4f}, RMSD_L = {metrics['rmsd_l_vq']:.4f}, RMSD_G = {metrics['rmsd_g_vq']:.4f}
# AR v.s. Nature: TM-score = {metrics['tm_ar']:.4f}, RMSD_L = {metrics['rmsd_l_ar']:.4f}, RMSD_G = {metrics['rmsd_g_ar']:.4f}
# """)
        logger.info(f"""Evaluated [{inputs['gt_struct_path'][0]}]:
# Exposure Loss:  {metrics['exposure_loss']:.4f}
# VQ v.s. Nature: TM-score = {metrics['tm_vq']:.4f}, RMSD_L = {metrics['rmsd_l_vq']:.4f}, RMSD_G = {metrics['rmsd_g_vq']:.4f}
# AR v.s. Nature: TM-score = {metrics['tm_ar']:.4f}, RMSD_L = {metrics['rmsd_l_ar']:.4f}, RMSD_G = {metrics['rmsd_g_ar']:.4f}
# """)
        model.train()
        preds = {k:torch.tensor(v).to(device) for k, v in metrics.items()} # metrics to tensor
        return (exposure_loss, preds, inputs['input_ids'])
        
        
def lf_metrics(eval_pred: EvalPrediction):
    preds: Dict[str, np.ndarray] = eval_pred.predictions # type: ignore
    # group average by `dev` field in `preds`
    # add prefix `overfit`(when `dev`=1) or `test`(when `dev`=2)
    df = pd.DataFrame({k: v for k, v in preds.items()})
    metrics = {}
    # pdb.set_trace()
    # for i, group in df.groupby('split'):
    #     prefix = list(DATASET_SPLIT.keys())[int(i)] # type: ignore
    #     metrics[f'{prefix}/exposure_loss'] = group['exposure_loss'].mean()
    #     metrics[f'{prefix}/tm_vq'] = group['tm_vq'].mean()
    #     metrics[f'{prefix}/rmsd_l_vq'] = group['rmsd_l_vq'].mean()
    #     metrics[f'{prefix}/rmsd_g_vq'] = group['rmsd_g_vq'].mean()
    #     metrics[f'{prefix}/tm_ar'] = group['tm_ar'].mean()
    #     metrics[f'{prefix}/rmsd_l_ar'] = group['rmsd_l_ar'].mean()
    #     metrics[f'{prefix}/rmsd_g_ar'] = group['rmsd_g_ar'].mean()
    
    prefix = "eval"   # 或 'train'，看你想记成什么

    metrics[f'{prefix}/exposure_loss'] = df['exposure_loss'].mean()
    metrics[f'{prefix}/tm_vq']         = df['tm_vq'].mean()
    metrics[f'{prefix}/rmsd_l_vq']     = df['rmsd_l_vq'].mean()
    metrics[f'{prefix}/rmsd_g_vq']     = df['rmsd_g_vq'].mean()
    metrics[f'{prefix}/tm_ar']         = df['tm_ar'].mean()
    metrics[f'{prefix}/rmsd_l_ar']     = df['rmsd_l_ar'].mean()
    metrics[f'{prefix}/rmsd_g_ar']     = df['rmsd_g_ar'].mean()
    return metrics


# Implementation of SFT trainer
@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def sft(config: DictConfig):
    
    start_time = time.time()
    config_dataset, config_lm, config_trainer = config.dataset, config.lm, config.trainer #相应config设置
    config.name = "{}@{}@{}".format(
        config_lm.get('model_type', 'dummy'),
        config_dataset.get('dataset_type', 'dummy'),
        int(os.environ["WORLD_SIZE"]),
    )
    config_trainer.output_dir = str(Path(__file__).parent/f'output/checkpoints/{config.name}')
    if (rank := int(os.environ.get("RANK", 0))) == 0:
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "LLMFolding"),
            project=os.environ.get("WANDB_PROJECT", "s2s_v1"),
            name=os.environ.get("WANDB_RUN_NAME", config.name),
            config=OmegaConf.to_container(config, resolve=True)
        ) # type: ignore
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Loaded config ...')
    
    # pdb.set_trace()
    start_time = time.time()
    dataset_eval = load_dataset('json', streaming=False, split='train', data_files=config_dataset.fpath_eval)
    dataset_train = load_dataset('json', streaming=True, split='train', data_files=config_dataset.fpath_train)
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Loaded dataset ...\n- {config_dataset.fpath_train}\n- {config_dataset.fpath_eval}')
    
    
    start_time = time.time()
    protein_tokenizer = {
        "dist": DistMatrixTokenizer,
        "dplm": DPLMProteinTokenizer,
    }[str(config_dataset.type)].get_instance()
    qwen2_tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(config_lm.model_dir)
    qwen2_tokenizer.padding_side = "left"
    qwen2_tokenizer.truncation_side = "right"
    qwen2_tokenizer.boseq_token = '<seq>'
    qwen2_tokenizer.eoseq_token = '</seq>'
    qwen2_tokenizer.bostruct_token = '<struct>'
    qwen2_tokenizer.eostruct_token = '</struct>'
    qwen2_tokenizer.struct_regex = r"<\|s(\d{4})\|>"
    qwen2_tokenizer.struct_template = "<|s{token_id:0>4d}|>"
    qwen2_tokenizer.struct_vsz = protein_tokenizer.vsz
    qwen2_tokenizer.add_special_tokens({
        'additional_special_tokens': \
        [qwen2_tokenizer.boseq_token, qwen2_tokenizer.eoseq_token, qwen2_tokenizer.bostruct_token, qwen2_tokenizer.eostruct_token] + \
        [qwen2_tokenizer.struct_template.format(token_id=i) for i in range(qwen2_tokenizer.struct_vsz)] # type: ignore
    }, replace_additional_special_tokens=False)
    qwen2_tokenizer.boseq_token_id = qwen2_tokenizer.convert_tokens_to_ids(qwen2_tokenizer.boseq_token)
    qwen2_tokenizer.eoseq_token_id = qwen2_tokenizer.convert_tokens_to_ids(qwen2_tokenizer.eoseq_token)
    qwen2_tokenizer.bostruct_token_id = qwen2_tokenizer.convert_tokens_to_ids(qwen2_tokenizer.bostruct_token)
    qwen2_tokenizer.eostruct_token_id = qwen2_tokenizer.convert_tokens_to_ids(qwen2_tokenizer.eostruct_token)
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Loaded and updated tokenizers ...')
    
    
    start_time = time.time()
    qwen3_model = AutoModelForCausalLM.from_pretrained(
        config_lm.model_dir,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    # HINT: `len(tokenizer)` is always right, rather than `vocab_size`
    qwen3_model.resize_token_embeddings(len(qwen2_tokenizer))
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Loaded and updated model ...')
    
    # pdb.set_trace()
    start_time = time.time()
    # WARN: for packing we have to use <|endoftext|> rather than <|im_end|>
    eod_token, eos_token = qwen2_tokenizer.pad_token, qwen2_tokenizer.eos_token
    qwen2_tokenizer.eos_token = eod_token
    qwen2_tokenizer.eos_token_id = qwen2_tokenizer.pad_token_id
    protein_processor = ProteinProcessor(qwen2_tokenizer, protein_tokenizer)
    sft_trainer = SFTTrainerWithEval(
        processor=protein_processor,
        model=qwen3_model,
        tokenizer=qwen2_tokenizer,
        args=SFTConfig(**config_trainer),
        train_dataset=dataset_train, # type: ignore
        eval_dataset=dataset_eval,   # type: ignore
        eval_packing=False,
        eval_collator=ExtraColumnCollator(),
        compute_metrics=lf_metrics,
    )
    # pdb.set_trace()
    sft_trainer.train() # type: ignore
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Finished SFT training ...')


if __name__ == "__main__":
    sft()