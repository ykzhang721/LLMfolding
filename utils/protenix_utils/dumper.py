from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import biotite.structure as struc
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile

from utils.protenix_utils import constants


STD_ELEMS: list[str] = constants.get_all_elems()
STD_RESTYPES: list[str] = list(constants.STD_RESIDUES.keys()) + ['gap']


__all__ = [
    'ProtenixBiotiteEntity'
]

class ProtenixBiotiteEntity:
    
    def __init__(self,
        input_dict: Dict[str, torch.Tensor],
        coordinates: Optional[torch.Tensor] = None,
        orig_atom_array: Optional[AtomArray] = None,
        orig_entity_poly_type: Optional[dict[int, str]] = None,
    ):
        atom_list = []
        for idx, (
            atom_char,
            atom_element,
            atom_restype,
            atom_uid,
            atom_res_id,
            atom_asym_id,
            atom_entity_id
        ) in enumerate(zip(
            input_dict['ref_atom_name_chars'],
            input_dict['ref_element'],
            input_dict['restype'],
            input_dict['ref_space_uid'],
            input_dict['residue_index'],
            input_dict['entity_id'],
            input_dict['asym_id'],
        )):
            atom_obj = struc.Atom(
                [0.0, 0.0, 0.0],
                res_id=int(atom_res_id.item()),
                chain_id=int(atom_asym_id.item()) + 1,
                label_asym_id=int(atom_asym_id.item()) + 1,
                label_entity_id=int(atom_entity_id.item()) + 1,
                res_name=self.atom_restype(atom_restype),
                atom_name=self.atom_name(atom_char),
                element=self.atom_element(atom_element),
                hetero=False,
            )
            atom_list.append(atom_obj)
        self.atom_array = struc.array(atom_list)
        
        # TODO `add_bond` interface here

        if coordinates is not None:
            self.set_coordinates(coordinates)

    
    @staticmethod
    def atom_name(
        atom_char: torch.Tensor
    ) -> str:
        # atom name [N_chars=4, N_vocab=64], one-hot
        # COMMENT: seems redundant, why not atom14/atom37 ?
        atom_name = ''
        for i in atom_char:
            atom_name_idx = int(torch.where(i == 1)[0].item())
            # COMMENT: unicode(A) = 65
            atom_name += chr(atom_name_idx + 32)
        return atom_name.strip()
    
    @staticmethod
    def atom_element(
        atom_element: torch.Tensor
    ) -> str:
        # atom element [N_vocab=128], one-hot
        element_idx = int(torch.where(atom_element == 1)[0].item())
        return STD_ELEMS[element_idx]
    
    @staticmethod
    def atom_restype(
        atom_restype: torch.Tensor,
        atom_uid: Optional[torch.Tensor] = None,
        orig_atom_array: Optional[AtomArray] = None
    ) -> str: 
        # atom restype [N_vocab=32], one-hot
        restype_idx = torch.where(atom_restype == 1)[0][0]
        restype_name3 = STD_RESTYPES[restype_idx]
        if restype_name3 == 'UNK' and orig_atom_array is not None:
            assert atom_uid is not None # COMMENT: because of possible permutation
            uid = int(atom_uid.item())
            restype_name3 = orig_atom_array.res_name[orig_atom_array.ref_space_uid == uid][0] # type: ignore
        return restype_name3
     
    @staticmethod
    def atomwise(input_feature_dict: dict):
        atomwise_input_feature_dict = {}
        for key in [
            'token_index',
            'residue_index',
            'entity_id',
            'asym_id',
            'restype',
        ]:
            # convert token-level features to atom level
            atomwise_input_feature_dict[key] = input_feature_dict[key][input_feature_dict['atom_to_token_idx']]
        for key in [
            'ref_atom_name_chars',
            'ref_element',
            'ref_space_uid',
        ]:
            # pick-up useful atom features
            atomwise_input_feature_dict[key] = input_feature_dict[key]
        return atomwise_input_feature_dict
    
    def set_coordinates(self, coordinates: torch.Tensor) -> AtomArray:
        self.atom_array.coord = coordinates.cpu().numpy()
        return self.atom_array
    
    def to_cif(self, path: Path, entry_id: Optional[str] = None, include_bonds: bool = False):
        if entry_id is None:
            entry_id = path.stem
        
        # TODO handle non-polymer here
        entity_poly_type = None
        
        block_dict = {'entry': pdbx.CIFCategory({'id': entry_id})}
        block = pdbx.CIFBlock(block_dict)
        cif = pdbx.CIFFile({path.stem: block})
        pdbx.set_structure(cif, self.atom_array, include_bonds=include_bonds)
        block = cif.block
        atom_site = block['atom_site']
        if (occ := atom_site.get('occupancy')) is None:
            atom_site['occupancy'] = np.ones(len(self.atom_array), dtype=float)
        atom_site['label_entity_id'] = self.atom_array.label_entity_id
        cif.write(path)