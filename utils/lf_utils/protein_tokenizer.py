from sre_parse import Tokenizer
import tokenize
from typing import Dict, List, Optional, Union, cast

import warnings
from pathlib import Path
from omegaconf import OmegaConf
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..openfold_utils.io import OpenfoldBackbone, OpenfoldProtein, atom_order


__all__ = [
    'ProteinTokenizer',
    'DPLMProteinTokenizer',
    'DistMatrixTokenizer',
]


class ProteinTokenizer(nn.Module):
    """ protein-in(wo/ mask), protein-out(w/ mask) """
    
    def __init__(self):
        super().__init__()
        pass
    
    def __call__(self, batch_proteins: List[OpenfoldProtein]) -> Dict[str, torch.Tensor]:
        # support batch encode, thus padding_mask is returned
        raise NotImplementedError()

    def batch_decode(self, batch_token_ids: torch.Tensor, **kwargs) -> List[OpenfoldProtein]:
        # support batch decode, thus padding_mask is expected(in kwargs)
        raise NotImplementedError()

    def encode(self, protein: OpenfoldProtein) -> torch.Tensor:
        raise NotImplementedError()
    
    def decode(self, token_ids: torch.Tensor, **kwargs) -> OpenfoldProtein:
        raise NotImplementedError()
    
    @property
    def device(self) -> torch.device:
        raise NotImplementedError()
    
    @property
    def vsz(self) -> int:
        raise NotImplementedError()
    
    
class DPLMProteinTokenizer(ProteinTokenizer):
    """ Wrapper implementation for DPLM2 tokenizer. """    
    def __init__(
        self,
        config_path: Path,
        ckpt_path: Optional[Path] = None,
        eval_mode: bool = True
    ):
        from ..dplm_utils import VQModel as DPLMVQModel
        super().__init__()
        config = OmegaConf.load(config_path)
        OmegaConf.resolve(config)
        tokenizer = DPLMVQModel(**config) # type: ignore
        if ckpt_path is not None:
            pretrained_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            missing_keys, unexpected_keys = tokenizer.load_state_dict(pretrained_state_dict, strict=True)
            tokenizer = tokenizer.requires_grad_(False)
            tokenizer = tokenizer.train(not eval_mode)
        self.tokenizer = tokenizer
        self.config = config
        
    @classmethod
    def get_instance(cls):
        dplm_protein_tokenizer_path = Path(__file__).parent.parent/'dplm_utils/checkpoints/struct_tokenizer'
        torch.hub.set_dir(dplm_protein_tokenizer_path)
        return cls(
            config_path=dplm_protein_tokenizer_path/'config.yaml',
            ckpt_path=dplm_protein_tokenizer_path/'dplm2_struct_tokenizer.ckpt',
            eval_mode=True
        )

    def __call__(self, batch_proteins: List[OpenfoldProtein]) -> Dict[str, torch.Tensor]:
        # support batch encode, thus batch_padding_mask is returned
        batch_proteins = [p.to(self.device) for p in batch_proteins]
        collect_residue_atom37_coord = [p.residue_atom37_coord for p in batch_proteins]
        collect_residue_mask = [p.residue_mask for p in batch_proteins]
        
        # organized as padded batch, with corresponding padding_mask
        max_length: int = max([len(p) for p in batch_proteins])
        batch_residue_atom37_coord = torch.stack([
            torch.nn.functional.pad(
                p, (0, 0, 0, 0, 0, max_length - len(p)), value=0.0
            ) for p in collect_residue_atom37_coord
        ], dim=0) # [l, 37, 3]... pad > stack > [B, L, 37, 3]
        
        # residue_mask includes both padding and missing residues
        batch_residue_mask = torch.stack([
            torch.nn.functional.pad(
                p, (0, max_length - len(p)), value=0.0
            ) for p in collect_residue_mask
        ], dim=0) # [l,]... pad > stack > [B, L]
        
        # TODO [B] representation is enough to generate right padding mask
        batch_lengths = torch.tensor(
            [len(p) for p in batch_proteins],
            dtype=torch.long, device=batch_residue_mask.device
        ) # [B]
        batch_padding_mask = 1 - (
            torch.arange(batch_residue_mask.shape[1], device=batch_residue_mask.device)[None, :]
            < batch_lengths[:, None]
        ).to(batch_residue_mask.dtype) # [B, L]
        
        # core implementation
        output: torch.Tensor = self.tokenizer.tokenize(
            atom_positions=batch_residue_atom37_coord,  # [B, L, 37, 3]
            res_mask=batch_residue_mask,                # [B, L]
            seq_length=batch_lengths                    # [B]
        )
        
        # set right-padding tokens to -100
        output = output.masked_fill(batch_padding_mask.bool(), -100)
        
        # convert to left-padding
        return {
            'batch_token_ids': output,                  # [B, L]
            'batch_residue_mask': batch_residue_mask    # [B, L]
        }
        
    def batch_decode(self, batch_token_ids: torch.Tensor, **kwargs) -> List[OpenfoldProtein]:
        # before calling, -100 should be set for right-padding tokens
        # construct batch_length from batch_token_ids(-100 is set)
        
        batch_length = (batch_token_ids != -100).sum(dim=1) # [B]
        if 'batch_residue_mask' not in kwargs:
            warnings.warn("Expect 'batch_residue_mask' in kwargs for decoding. Assuming no missing residues.")
            
        # NOTE while decoding, we ignore paddings by batch_residue_mask
        # and each sequence is decoded to a backbone structure(w/ missing resiudes)
        output = self.tokenizer.detokenize(
            struct_tokens=batch_token_ids,              # [B, L]
            res_mask=kwargs.get('batch_residue_mask')   # [B, L] or None
        )
        
        batch_proteins = []
        for i, l in enumerate(batch_length):
            residue_atom37_coord = output['atom37_positions'][i, :l, :] # [l, 37, 3]
            residue_atom37_mask = output['atom37_mask'][i, :l, :]       # [l, 37]
            if kwargs.get('batch_residue_mask') is not None: # inherit missing residues
                residue_atom37_mask *= kwargs['batch_residue_mask'][i, :l].unsqueeze(1)  # [l, 1]
            backbone = OpenfoldBackbone.from_dict(dict(
                residue_atom37_coord=residue_atom37_coord,
                residue_atom37_mask=residue_atom37_mask
            ))
            protein = OpenfoldProtein.from_backbone(backbone)
            batch_proteins.append(protein)
        return batch_proteins
    
    def encode(self, protein: OpenfoldProtein) -> torch.Tensor:
        # single encode, thus no padding_mask is returned
        output = self.__call__([protein])
        return output['batch_token_ids'].squeeze(0) # [L]
    
    def decode(self, token_ids: torch.Tensor, **kwargs) -> OpenfoldProtein:
        # batch_lengths is not necessary for single decode
        if 'residue_mask' not in kwargs:
            assert 'ref' in kwargs
            ref_residue_mask = self([kwargs.pop('ref')])['batch_residue_mask'][0]
            kwargs['residue_mask'] = ref_residue_mask
            # this 'leakage' function is to conveniently evaluate structure prediction
        
        residue_mask: torch.Tensor = kwargs.pop('residue_mask', None)
        return self.batch_decode(
            batch_token_ids=token_ids.unsqueeze(0),
            batch_residue_mask=residue_mask.unsqueeze(0) if residue_mask is not None else None
        )[0]
        
    @property
    def device(self) -> torch.device:
        return next(self.tokenizer.parameters()).device
    
    @property
    def vsz(self) -> int:
        return self.config.codebook_config.num_codes


class DistMatrixTokenizer(ProteinTokenizer):
    def __init__(
        self,
        tokenizer_ckpt_path: str | Path,
        structure_ckpt_path: str | Path,
        map_location: Union[str, torch.device] = "cpu",
    ) -> None:
        from ..dist_utils import DiscreteTokenizer, StructurePredictionModel
        super().__init__()
        self.tokenizer_ckpt_path = Path(tokenizer_ckpt_path).resolve()
        self.structure_ckpt_path = Path(structure_ckpt_path).resolve()
        self.map_location = map_location

        print(f"Loading tokenizer checkpoint from: {self.tokenizer_ckpt_path}")
        tokenizer_checkpoint = torch.load(self.tokenizer_ckpt_path, map_location=map_location, weights_only=False)
        self.config = tokenizer_checkpoint["hyper_parameters"]["config"]

        self.model = DiscreteTokenizer(
            in_channels=self.config.model.in_channels,
            embed_dim=self.config.model.embed_dim,
            patch_size=self.config.model.patch_size,
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            mlp_ratio=self.config.model.mlp_ratio,
            dropout=self.config.model.dropout,
            z_channels=self.config.model.z_channels,
            double_z=self.config.model.double_z,
            quantizer_type=self.config.quantizer._target_,
            quantizer_kwargs=dict(self.config.quantizer.params),
        )

        tokenizer_state_prefix = "model."
        tokenizer_state = {
            k[len(tokenizer_state_prefix) :]: v
            for k, v in tokenizer_checkpoint["state_dict"].items()
            if k.startswith(tokenizer_state_prefix)
        }
        self.model.load_state_dict(tokenizer_state)

        print(f"Loading structure checkpoint from: {self.structure_ckpt_path}")
        structure_checkpoint = torch.load(self.structure_ckpt_path, map_location=map_location, weights_only=False)
        self.structure_config = structure_checkpoint["hyper_parameters"]["config"]
        self.use_frame_coordinates = bool(
            getattr(self.structure_config.model, "use_frame_coordinates")
        )

        output_dim = 9 if self.use_frame_coordinates else 3

        self.structure_model = StructurePredictionModel(
            in_channels=self.structure_config.model.in_channels,
            embed_dim=self.structure_config.model.embed_dim,
            num_layers=self.structure_config.model.num_layers,
            num_heads=self.structure_config.model.num_heads,
            mlp_ratio=self.structure_config.model.mlp_ratio,
            dropout=self.structure_config.model.dropout,
            max_seq_len=self.structure_config.model.max_seq_len,
            use_sinusoidal_pos_embed=self.structure_config.model.use_sinusoidal_pos_embed,
            use_frame_coordinates=self.use_frame_coordinates,
            output_dim=output_dim,
        )

        structure_state_prefix = "model."
        structure_state = {
            k[len(structure_state_prefix) :]: v
            for k, v in structure_checkpoint["state_dict"].items()
            if k.startswith(structure_state_prefix)
        }
        self.structure_model.load_state_dict(structure_state)

        self.std_value = float(self.config.data.std_value)
        self.std_data = float(self.structure_config.data.std_data)

        self.model.eval()
        self.structure_model.eval()
    
    @classmethod
    def get_instance(cls):
        from ..dist_utils import version_discrete_tokenizer, version_structure_head
        return cls(
            tokenizer_ckpt_path=version_discrete_tokenizer,
            structure_ckpt_path=version_structure_head,
            map_location="cpu",
        )

    @torch.no_grad()
    def __call__(self, batch_proteins: list[OpenfoldProtein]) -> dict[str, torch.Tensor]:
        # support batch encode, thus padding_mask is returned
        indices_list = []
        for protein in batch_proteins:
            indices = self.encode(protein)
            indices_list.append(indices)
        
        batch_indices = torch.nn.utils.rnn.pad_sequence(
            indices_list, batch_first=True, padding_value=-100
        )
        padding_mask = (batch_indices == -100).long()

        return {
            "batch_token_ids": batch_indices,
            "batch_padding_mask": padding_mask,
        }

    @torch.no_grad()
    def batch_decode(self, batch_token_ids: torch.Tensor, **kwargs) -> list[OpenfoldProtein]:
        # support batch decode, thus padding_mask is expected(in kwargs)
        batch_lengths = kwargs.get('batch_lengths', None)
        protein_lengths = kwargs.get('protein_lengths', None)
        if batch_lengths is None or protein_lengths is None:
            raise ValueError("batch_lengths and protein_lengths must be provided for batch decoding.")
        batch_size = batch_token_ids.shape[0]
        decoded_proteins = []
        for i in range(batch_size):
            batch_length = batch_lengths[i]
            token_ids = batch_token_ids[i][:batch_length]
            protein_length = protein_lengths[i]
            protein = self.decode(token_ids, protein_length=protein_length)
            decoded_proteins.append(protein)
        return decoded_proteins

    @torch.no_grad()
    def encode(self, protein: OpenfoldProtein) -> torch.Tensor:
        patch_size = self.config.model.patch_size
        ca_index = atom_order["CA"]

        ca_coords = protein.residue_atom37_coord[:, ca_index, :].to(self.device, dtype=torch.float32)

        distance_tensor = (ca_coords[:, None, :] - ca_coords[None, :, :]) / self.std_value
        seq_len = distance_tensor.size(0)
        padded_len = math.ceil(seq_len / patch_size) * patch_size

        if padded_len != seq_len:
            pad_amount = padded_len - seq_len
            distance_tensor = F.pad(distance_tensor, (0, 0, 0, pad_amount, 0, pad_amount))

        model_input = distance_tensor.unsqueeze(0)
        quantized, indices, _ = self.model.encode(model_input)
        indices = indices.reshape(indices.shape[0], -1)
        return indices.squeeze(0).contiguous()

    @torch.no_grad()
    def decode(self, token_ids: torch.Tensor, **kwargs) -> OpenfoldProtein:
        # HINT: for patch-based methods, there are both patch-padding and batch_padding
        # thus we need to know the original protein length for correct decoding
        
        if 'protein_length' not in kwargs:
            # PLUGIN: this 'leakage' function is to conveniently evaluate structure prediction
            assert 'ref' in kwargs, "Expect 'protein_length' or 'ref' in kwargs for decoding."
            kwargs['protein_length'] = len(kwargs.pop('ref'))
        
        protein_length = kwargs.get("protein_length")
        if protein_length is None:
            raise ValueError("protein_length must be provided for decoding.")
        patch_size = self.config.model.patch_size
        z_channels = self.config.model.z_channels

        token_ids = token_ids.to(self.device)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids.to(torch.int32)

        padded_len = math.ceil(protein_length / patch_size) * patch_size
        h_patches = padded_len // patch_size
        w_patches = h_patches
        expected_tokens = h_patches * w_patches
        if token_ids.shape[-1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} tokens for protein length {protein_length}, "
                f"but received {token_ids.shape[-1]} tokens."
            )

        quantizer = self.model.quantizer
        codes = quantizer.indices_to_codes(token_ids)
        codes = codes.to(self.device, dtype=torch.float32)

        batch_size = codes.shape[0]
        quantized = codes.view(batch_size, h_patches, w_patches, z_channels)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        reconstructed = self.model.decode(quantized)
        distance_matrix = reconstructed[:, : protein_length, : protein_length, :]

        residue_atom37_coord = torch.zeros((protein_length, 37, 3), device=self.device)
        residue_atom37_mask = torch.zeros((protein_length, 37), device=self.device)

        if self.use_frame_coordinates:
            predicted_frames = self.structure_model(distance_matrix) * self.std_data
            predicted_frames = predicted_frames.squeeze(0)  # [L, 3, 3]
            frame_indices = [atom_order["N"], atom_order["CA"], atom_order["C"]]
            for atom_idx, frame_index in enumerate(frame_indices):
                residue_atom37_coord[:, frame_index, :] = predicted_frames[:, atom_idx, :]
                residue_atom37_mask[:, frame_index] = 1.0
        else:
            predicted_ca = self.structure_model(distance_matrix) * self.std_data
            ca_index = atom_order["CA"]
            residue_atom37_coord[:, ca_index, :] = predicted_ca
            residue_atom37_mask[:, ca_index] = 1.0
        backbone = OpenfoldBackbone.from_dict(
            dict(
                residue_atom37_coord=residue_atom37_coord,
                residue_atom37_mask=residue_atom37_mask,
            )
        )
        protein = OpenfoldProtein.from_backbone(backbone)
        return protein
    
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def vsz(self) -> int:
        quantizer = self.model.quantizer
        return quantizer.codebook_size
