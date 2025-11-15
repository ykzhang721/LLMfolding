import time
from typing import Any, Dict, Optional, List, Tuple

import re
from pathlib import Path
from unittest import result

import numpy as np
import torch
import pdb
import transformers
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from .text_tokenizer import TextTokenizer
from .protein_tokenizer import ProteinTokenizer
from ..openfold_utils.io import OpenfoldProtein



__all__ = ['ProteinProcessor']

class ProteinProcessor(ProcessorMixin):
    """ Organize components. """
    tokenizer: Any
    struct_tokenizer: ProteinTokenizer
    struct_vsz: int
    struct_regex: str
    struct_template: str
    constant: Dict[str, int | List[int] | Any]
    
    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerFast"
    
    def __init__(
        self,
        tokenizer: Any,
        struct_tokenizer: ProteinTokenizer,
    ):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        # HINT tokenizer's vsz could be larger than actual vsz
        self.struct_tokenizer = struct_tokenizer
        self.struct_vsz = struct_tokenizer.vsz
        self.struct_regex = tokenizer.struct_regex
        self.struct_template = tokenizer.struct_template

    @torch.no_grad()
    def __call__(
        self,
        seq_input: str | List[str],
        struct_input: OpenfoldProtein | List[OpenfoldProtein],
        **kwargs,
    ) -> Tuple[BatchFeature, BatchFeature]:
        
        if isinstance(struct_input, OpenfoldProtein): struct_input = [struct_input]
        if isinstance(seq_input, str): seq_input = [seq_input]
        struct_input = [s.to(self.struct_tokenizer.device) for s in struct_input]

        right_out = self.struct_tokenizer(struct_input)
        
        # convert to structure text
        seq_text: List[str] = seq_input
        struct_text: List[str] = []
        batch_token_ids: torch.Tensor = right_out['batch_token_ids']           # [B, L_seq]
        batch_padding_mask: torch.Tensor = (batch_token_ids == -100).long()    # [B, L_seq]
        for token_ids, padding_mask in zip(batch_token_ids, batch_padding_mask):
            token_ids = token_ids[~padding_mask.bool()]
            struct_text.append("".join([self.struct_template.format(token_id=i) for i in token_ids]))
        
        train_folding = lambda t, s: self.tokenizer.bos_token + self.tokenizer.boseq_token + t + self.tokenizer.eoseq_token \
                    + self.tokenizer.bostruct_token + s + self.tokenizer.eostruct_token + self.tokenizer.eos_token
        eval_folding = lambda t, s: self.tokenizer.bos_token + self.tokenizer.boseq_token + t + self.tokenizer.eoseq_token \
                    + self.tokenizer.bostruct_token

        train_inputs = self.tokenizer(list(map(train_folding, seq_text, struct_text)), **kwargs)
        eval_inputs = self.tokenizer(list(map(eval_folding, seq_text, struct_text)), **kwargs)
        train_inputs.pop('token_type_ids', None)    # not used
        eval_inputs.pop('token_type_ids', None)     # not used
        
        # copy k,v other than batch_token_ids, batch_padding_mask
        # these keys will be passed to mutimodal_decode()
        for k, v in right_out.items():
            if k not in ['batch_token_ids']:
                train_inputs[k] = v                 # [B, L_seq]
                eval_inputs[k] = v                  # [B, L_seq]
        
        if kwargs.get('return_tensors') != 'pt':
            raise NotImplementedError('Only support pt tensors') # TODO

        return BatchFeature(train_inputs), BatchFeature(eval_inputs) # type: ignore


    @torch.no_grad()
    def multimodal_decode(self, token_ids: torch.Tensor, **kwargs) -> Dict[str, Any]:
        
        # HINTL token_ids is left-padded, and kwargs might be right padded depending on tokenizer call
        # specifying additional kwargs for structure decoding e.g. residue_mask
        
        string = self.tokenizer.decode(token_ids)
        # pattern = rf'^{re.escape(self.tokenizer.bostruct_token)}(({self.tokenizer.struct_regex})+){re.escape(self.tokenizer.eostruct_token)}$'
        pattern = re.compile(
                rf'{re.escape(self.tokenizer.bostruct_token)}'
                rf'(({self.tokenizer.struct_regex})+)'
                rf'{re.escape(self.tokenizer.eostruct_token)}'
            )
        # chunks = re.split(pattern, string)
        chunks = []
        matches = list(pattern.finditer(string))
        for m in matches:
            full_struct_block = m.group(0)   # 包含 <struct> ... </struct>
            struct_tokens = m.group(1)       # 里面那一串 <|sXXXX|>...
            chunks.append(full_struct_block)
        
        seq_output, struct_output, entity_output = [], [], []
        
        # import pdb; pdb.set_trace()
        for i, c in enumerate(chunks[1:]): #这里不解析输入的结构，因为给的是gt text，所以算输出结构就行,取-1
            # pdb.set_trace()
            if len(c) == 0: continue
            if self.tokenizer.bostruct_token in c and self.tokenizer.eostruct_token in c:
                # as structure
                protein = self.struct_tokenizer.decode(
                    token_ids=torch.tensor(
                        [int(i) for i in re.findall(self.struct_regex, c)], device=self.device
                    ),
                    **kwargs
                )
                struct_output.append(c)
                entity_output.append(protein)
            else:
                # as text
                seq_output.append(c)
        # import pdb; pdb.set_trace()
        return {
            'text':         string,
            'seq':          seq_output,
            'struct':       struct_output,
            'entity':       entity_output
        }
    
    def constant_helper(self) -> Dict[str, int | List[int] | Any]:
        (
            pad_token,
            boseq_token,
            eoseq_token,
            bostruct_token,
            eostruct_token,
            bos_token,
            eos_token,
        ) = self.tokenizer.encode(''.join([
            self.tokenizer.pad_token,
            self.tokenizer.boseq_token,
            self.tokenizer.eoseq_token,
            self.tokenizer.bostruct_token,
            self.tokenizer.eostruct_token,
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
        ]))
        
        seq_vocab_ids = self.tokenizer.seq_vocab_ids
        struct_vocab_ids = self.tokenizer.struct_vocab_ids
        return {
            'pad_token': pad_token,
            'boseq_token': boseq_token,
            'eoseq_token': eoseq_token,
            'bostruct_token': bostruct_token,
            'eostruct_token': eostruct_token,
            'bos_token': bos_token,
            'eos_token': eos_token,
            'seq_vocab_ids': seq_vocab_ids,
            'struct_vocab_ids': struct_vocab_ids
        }

    @staticmethod
    def compute_tm_align(structure1: OpenfoldProtein, structure2: OpenfoldProtein, ref: OpenfoldProtein | None) -> Tuple[float, float, float]:
        if ref is not None:
            structure1.inherit(ref)
            structure2.inherit(ref)
        # pdb.set_trace()
        return structure1.align_with(structure2, chain_wise=True)
    
    @staticmethod
    def compute_kbastch_align(structure1: OpenfoldProtein, structure2: OpenfoldProtein) -> Tuple[float, float]:
        raise NotImplementedError()

    def preprocess_dataset(self, dataset_name: str, batch: List[OpenfoldProtein], verbose: bool = True) -> List[dict]:
        batch = [b.to(self.device) for b in batch]
        out = self.struct_tokenizer(batch)
        results = []
        batch_token_ids = out['batch_token_ids']                # [B, L]
        batch_padding_mask = (batch_token_ids == -100).long()   # [B, L]
        for protein, token_ids, padding_mask in zip(batch, batch_token_ids, batch_padding_mask):
            seq_text = ' '.join(list(str(protein)))
            seq_length = len(protein)
            token_ids = token_ids[~padding_mask.bool()]
            struct_text = "".join([self.struct_template.format(token_id=i) for i in token_ids])
            struct_length = len(token_ids)
            text = f"<seq>{seq_text}</seq><struct>{struct_text}</struct>"
            prompt = f"<seq>{seq_text}</seq><struct>"
            results.append({
                "pdb_name": protein.entry,
                "plddt": protein.plddt,
                "text": text,
                "prompt": prompt,
                "seq_length": seq_length,
                "struct_length": struct_length,
                "split": dataset_name,
            })
        return results
        
    def to(self, device: str | torch.device):
        self.struct_tokenizer.to(device)
        return self
    
    @property
    def device(self) -> torch.device:
        return self.struct_tokenizer.device