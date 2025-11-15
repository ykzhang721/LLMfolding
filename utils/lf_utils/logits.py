from typing import Any, Dict, Optional, Tuple

import torch
from transformers import LogitsProcessor

__all__ = ['DynamicMultimodalLogitsProcessor']


class DynamicMultimodalLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        pad_token: int,
        bos_token: int,
        eos_token: int,
        boseq_token: int,
        eoseq_token: int,
        bostruct_token: int,
        eostruct_token: int,
        seq_vocab_ids: list[int], # constraint for token range
        struct_vocab_ids: list[int],
        seq_length: list[int], # constraint for token num (batched)
        struct_length: list[int],
        **kwargs
    ):
        # tokens
        self.pad_token = pad_token          # text sequence
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.boseq_token = boseq_token
        self.eoseq_token = eoseq_token
        self.bostruct_token = bostruct_token
        self.eostruct_token = eostruct_token
        
        # constraints
        self.seq_vocab_ids = seq_vocab_ids
        self.struct_vocab_ids = struct_vocab_ids
        self.seq_length = seq_length
        self.struct_length = struct_length

        # fsm states
        self.states = {
            'INITIAL': 0,
            'TEXT': 1,
            'SEQ': 2,
            'STRUCTURE': 3,
            'TERMINAL': 4,
        }
        self.states_reverse = {v: k for k, v in self.states.items()}
        # item-wise fsm
        bsz = len(self.seq_length)
        self.batch_current_state = [self.states['INITIAL'] for _ in range(bsz)]
        self.batch_current_cnt = [{'SEQ_CNT': 0, 'STRUCTURE_CNT': 0} for _ in range(bsz)]


    def _constraint_step(self, batch_id: int, last_token: int) -> Optional[Tuple[int, ...]]:
        # current_token \in f(current_state, last_token)
        current_state: int = self.batch_current_state[batch_id]
        current_cnt: Dict[str, int] = self.batch_current_cnt[batch_id]
        
        constraint1 = self.seq_length[batch_id]
        constraint2 = self.struct_length[batch_id]
        
        if current_state == self.states['INITIAL']:
            constraint = (self.bos_token,)
        elif current_state == self.states['TEXT']:
            # TODO remove after leveraging text ?
            if last_token == self.eostruct_token:
                constraint = (self.eos_token,)
            else:   
                constraint = None
        elif current_state == self.states['SEQ']:
            if current_cnt['SEQ_CNT'] == constraint1:
                constraint = (self.eoseq_token,)
            else:
                constraint = tuple(self.seq_vocab_ids)
        elif current_state == self.states['STRUCTURE']:
            if current_cnt['STRUCTURE_CNT'] == constraint2:
                constraint = (self.eostruct_token,)
            else:
                constraint = tuple(self.struct_vocab_ids)
        elif current_state == self.states['TERMINAL']:
            constraint = (self.pad_token,)
        else:
            raise ValueError()
        return constraint

    def _fsm_step(self, batch_id: int, current_token: int) -> int:
        
        # next_state = f(current_state, current_token)
        current_state: int = self.batch_current_state[batch_id]
        current_cnt: Dict[str, int] = self.batch_current_cnt[batch_id]
        constraint1 = self.seq_length[batch_id]
        constraint2 = self.struct_length[batch_id]
               
        if current_state == self.states['INITIAL']:
            if current_token == self.pad_token:
                next_state = self.states['INITIAL']
            elif current_token == self.bos_token:
                next_state = self.states['TEXT']
            else:
                raise ValueError()
        elif current_state == self.states['TEXT']:
            if current_token == self.boseq_token:
                next_state = self.states['SEQ']
            elif current_token == self.bostruct_token:
                next_state = self.states['STRUCTURE']
            elif current_token == self.eos_token:
                next_state = self.states['TERMINAL']
            else:
                # Might include tokens from seq or struct
                next_state = self.states['TEXT']    
        elif current_state == self.states['SEQ']:
            if current_token in self.seq_vocab_ids:
                current_cnt['SEQ_CNT'] += 1
                if current_cnt['SEQ_CNT'] <= constraint1:
                    next_state = self.states['SEQ']
                else:
                    raise ValueError()
            elif current_token == self.eoseq_token:
                next_state = self.states['TEXT']
                # TODO necessary ? 
                current_cnt['SEQ_CNT'] = 0
                current_cnt['STRUCTURE_CNT'] = 0
            else:
                raise ValueError()
        elif current_state == self.states['STRUCTURE']:
            if current_token in self.struct_vocab_ids:
                current_cnt['STRUCTURE_CNT'] += 1
                if current_cnt['STRUCTURE_CNT'] <= constraint2:
                    next_state = self.states['STRUCTURE']
                else:
                    raise ValueError()
            elif current_token == self.eostruct_token:
                next_state = self.states['TEXT']
                current_cnt['SEQ_CNT'] = 0
                current_cnt['STRUCTURE_CNT'] = 0
            else:
                raise ValueError()
        elif current_state == self.states['TERMINAL']:
            next_state = self.states['TERMINAL']
        else:
            raise ValueError()
        return next_state
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        B, V = scores.shape
        scores_mask = torch.full_like(scores, float('-inf'))
        
        # HINT param name: last_token -> current_state -> current_token -> next_state
        for batch_id in range(B):
            
            # step1 simulating all input ids: 
            if (current_state := self.batch_current_state[batch_id]) == self.states['INITIAL']:
                for i in range(L - 1):
                    current_token = int(input_ids[batch_id, i].item())
                    next_state = self._fsm_step(batch_id, current_token)
                    self.batch_current_state[batch_id] = next_state
            
            current_state = self.batch_current_state[batch_id]
            current_token = int(input_ids[batch_id, -1].item())
            next_state = self._fsm_step(batch_id, current_token)
            self.batch_current_state[batch_id] = next_state
            
            # step2: apply constraints
            constraint = self._constraint_step(batch_id, current_token)
            if constraint is None:
                scores_mask[batch_id, :] = 0
            else:
                scores_mask[batch_id, list(constraint)] = 0
        
        return scores + scores_mask
    