import torch
import gemmi
import warnings
import numpy as np
import tmtools
from pathlib import Path
from typing import Any, List, Tuple, Union, Optional, Union, Dict
from ..protenix_utils import rmsd_globally_aligned


__all__ = ['OpenfoldBackbone', 'OpenfoldProtein']



# To facilitate usage, we cloned AF2.residue_constants to this file:
# arom-wise vocabulary
atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.
atom_types2element = {k:k[0] for k in atom_types}

# residue-wise vocabulary
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}
restype_num_with_x = len(restypes_with_x)  # := 21.

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}
# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}


class ChainId2ChainName:
    CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    BASE = len(CHARS)

    def __getitem__(self, idx: int) -> str:
        # 1-char
        if idx < self.BASE:
            return self.CHARS[idx]
        # 2-char
        idx -= self.BASE
        if idx < self.BASE**2:
            return self.CHARS[idx // self.BASE] + self.CHARS[idx % self.BASE]
        # 3-char
        idx -= self.BASE**2
        if idx < self.BASE**3:
            i = idx // (self.BASE**2)
            j = (idx // self.BASE) % self.BASE
            k = idx % self.BASE
            return self.CHARS[i] + self.CHARS[j] + self.CHARS[k]
        else:
            raise NotImplementedError()

class ChainName2ChainId:
    CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    BASE = len(CHARS)
    
    def __getitem__(self, name: str) -> int:
        name = name.upper()
        L = len(name)
        if L == 0:
            return 0
        elif L == 1:
            return self.CHARS.index(name)
        elif L == 2:
            return self.BASE + self.BASE * self.CHARS.index(name[0]) + self.CHARS.index(name[1])
        elif L == 3:
            return self.BASE + self.BASE**2 + self.BASE**2 * self.CHARS.index(name[0]) \
                   + self.BASE * self.CHARS.index(name[1]) + self.CHARS.index(name[2])
        else:
            raise NotImplementedError()


pdb_chain_ids = ChainId2ChainName()
pdb_chain_order = ChainName2ChainId()

dtype_template: Dict[str, torch.dtype] = {
    'residue_atom37_coord': torch.float32,
    'residue_atom37_mask':  torch.float32,
    'residue_atom37_bfactor': torch.float32,
    'residue_mask':         torch.float32,
    'residue_aatype':       torch.int32,
    'residue_index':        torch.int32,
    'residue_chain_index':  torch.int32,
}

gemmi_default_protocol = {
    'drop_water':   True,
    'drop_ligand':  True,
    'drop_na':      True,
    'drop_nonstd':  False,
    'aggregate':    True,
}

gemmi_checker = {
    'is_aa': lambda residue: (
                (lambda r: r is not None and r.is_amino_acid())
                (gemmi.find_tabulated_residue(residue.name))
            ),
    'is_na': lambda residue: (
                (lambda r: r is not None and r.is_nucleic_acid())
                (gemmi.find_tabulated_residue(residue.name))
            ),
    'is_ligand': lambda residue: (
                (lambda r: r is not None and not (r.is_amino_acid() or r.is_nucleic_acid() or r.is_water()))
                (gemmi.find_tabulated_residue(residue.name))
            ),
    'is_water': lambda residue: (
                (lambda r: r is not None and r.is_water())
                (gemmi.find_tabulated_residue(residue.name))
            ),
    'is_standard': lambda residue: (
                (lambda r: r is not None and r.is_standard())
                (gemmi.find_tabulated_residue(residue.name))
            ),
}


# TODO: if necessary, implement biopython, biotite parser
# WARN: gemmi takes auth_asym_id as name rather than label_asym_id
def _gemmi_parser(
    p: Path,
    protocol: Dict[str, bool] = gemmi_default_protocol,
    verbose: bool = True,
) -> dict:
    # one can input: 7dz2.cif, 7dz2_C.cif, AF-<ID>.cif.gz
    # assumption: @ indicates auth_chain_id, % indicates label_chain_id
    p_copy = p
    stem = p.name.strip('.gz').strip('.cif')
    if '%' in stem:
        subchains_flag = 'label'
        subchains: List[str] | None = stem.split('%')
        if len(subchains) > 1:
            pure_name = subchains[0]
            subchains = subchains[1:]
            p = p.parent / (pure_name + ''.join(p.suffixes))
        else:
            subchains = None
    elif '@' in stem:
        subchains_flag = 'auth'
        subchains: List[str] | None = stem.split('@')
        if len(subchains) > 1:
            pure_name = subchains[0]
            subchains = subchains[1:]
            p = p.parent / (pure_name + ''.join(p.suffixes))
        else:
            subchains = None
    else:
        subchains_flag = 'auth'
        subchains = None
    
    # Most possibly due to non-standrd .cif format
    try:
        # p or p_copy
        structure = gemmi.read_structure(str(p))
    except Exception as e:
        if verbose:
            raise ValueError(f'Error reading structure from {p}: {e}. Returns empty dict.')
        else:
            print(f'Error parsing structure from {p}: {e}. Returns empty dict.')
            return {}

    structure.remove_alternative_conformations()
    (it, it_name) = {
        'auth': (structure[0], [c.name for c in structure[0]]),
        'label': (structure[0].subchains(), sorted([c.subchain_id() for c in structure[0].subchains()])),
    }[subchains_flag]

    chain2feature: Dict[str, Dict[str, torch.Tensor]] = {}
    for chain, name in zip(it, it_name):
        if subchains is not None and name not in subchains:
            continue
        feature_template = {
            'residue_atom37_coord': [],         # [L, 37, 3]
            'residue_atom37_mask':  [],         # [L, 37]
            'residue_atom37_bfactor': [],       # [L, 37]
            'residue_mask':         [],         # [L]
            'residue_aatype':       [],         # [L]
            'residue_index':        [],         # [L],
            'residue_chain_index':  [],         # [L]
        }
        
        for residue in chain:
            # Default behavior: keep std & non-std aa
            if protocol['drop_water'] and gemmi_checker['is_water'](residue): continue
            if protocol['drop_ligand'] and gemmi_checker['is_ligand'](residue): continue
            if protocol['drop_na'] and gemmi_checker['is_na'](residue): continue
            if protocol['drop_nonstd'] and (not gemmi_checker['is_standard'](residue)): continue
            
            atom37_coord = torch.zeros((atom_type_num, 3), dtype=dtype_template['residue_atom37_coord'])    # [37, 3]
            atom37_mask = torch.zeros((atom_type_num), dtype=dtype_template['residue_atom37_mask'])         # [37]
            atom37_bfactor = torch.zeros((atom_type_num), dtype=dtype_template['residue_atom37_bfactor'])   # [37]
            for atom in residue:
                if atom.name not in atom_types: continue
                atom37_coord[atom_order[atom.name]] = torch.tensor((atom.pos.x, atom.pos.y, atom.pos.z), dtype=dtype_template['residue_atom37_coord'])
                atom37_mask[atom_order[atom.name]] = 1.0
                atom37_bfactor[atom_order[atom.name]] = atom.b_iso
            
            # atom-wise
            feature_template['residue_atom37_coord'].append(atom37_coord)
            feature_template['residue_atom37_mask'].append(atom37_mask)
            feature_template['residue_atom37_bfactor'].append(atom37_bfactor)
            
            # residue-wise
            restype_idx = restype_order_with_x[restype_3to1.get(residue.name, 'X')]
            feature_template['residue_mask'].append(atom37_mask[1].to(dtype_template['residue_mask']))
            feature_template['residue_aatype'].append(
                torch.tensor(restype_idx, dtype=dtype_template['residue_aatype'])
            )
            feature_template['residue_index'].append(
                torch.tensor(residue.seqid.num, dtype=dtype_template['residue_index'])
            )
            feature_template['residue_chain_index'].append(
                torch.tensor(pdb_chain_order[name], dtype=dtype_template['residue_chain_index'])
            )
        
        if feature_template['residue_atom37_coord'] != []:
            chain2feature[name] = {k: torch.stack(v, dim=0) for k, v in feature_template.items()}
    
    # TODO we will handle non-protein strcutures in the future
    if chain2feature == {}:
        if verbose:
            raise ValueError(f'No residue left after filtering under current protocol. Returns empty dict.')
        else:
            print(f'No residue left for {p.stem} after filtering under current protocol. Returns empty dict.')
            return {}
    
    # Concat different chains
    if protocol['aggregate']:
        feature_names = list(next(iter(chain2feature.values())).keys())
        return {
            feature_name: torch.cat([
                chain2feature[chain_id][feature_name] for chain_id in chain2feature
            ], dim=0)
            for feature_name in feature_names
        }
    else:
        return chain2feature


class OpenfoldBackbone:
    
    # represents a concatenated bakcbone chains
    entry: str
    residue_atom37_coord:   torch.Tensor
    residue_atom37_mask:    torch.Tensor
    
    def __init__(self):
        pass
    
    @classmethod
    def from_file(cls, path: str | Path, verbose: bool = True):
        # FEAT: support .pdb file and .cif file
        # FEAT: now we can support format like 7dz2.C.cif, 7dz2_C.cif to sepcify chain
        if isinstance(path, str): path = Path(path)
        assert path.suffix.lower() in ['.cif', '.mmcif', '.pdb', '.gz'], f'Unsupported file type: {path.suffix}'
        instance = cls()
        gemmi_out = _gemmi_parser(path, verbose=verbose)
        if gemmi_out == {}:
            instance.entry = 'empty'
            return instance
        
        instance.entry = path.stem
        instance.residue_atom37_coord = gemmi_out['residue_atom37_coord']
        instance.residue_atom37_mask = gemmi_out['residue_atom37_mask']
        return instance
    
    @classmethod
    def from_dict(cls, feature_in: Dict[str, Any]):
        instance = cls()
        instance.entry = feature_in.get('entry', 'unknown')
        instance.residue_atom37_coord = feature_in['residue_atom37_coord']
        instance.residue_atom37_mask = feature_in['residue_atom37_mask']
        return instance
    
    def __len__(self) -> int:
        return self.residue_atom37_coord.size(0)
    
    def to(self, device: str | torch.device):
        self.residue_atom37_coord = self.residue_atom37_coord.to(device)
        self.residue_atom37_mask = self.residue_atom37_mask.to(device)
        return self
    
    def to_pdb(self, path: str | Path):
        if isinstance(path, str): path = Path(path)
        raise NotImplementedError('Incomplete feature. Cast to OpenfoldProtein first !')
        
    @property
    def device(self) -> torch.device:
        return self.residue_atom37_coord.device
    

class OpenfoldProtein:
    
    # represents a complete protein with additional metadata
    entry: str
    residue_atom37_coord:   torch.Tensor
    residue_atom37_mask:    torch.Tensor
    residue_mask:           torch.Tensor
    residue_aatype:         torch.Tensor
    residue_index:          torch.Tensor
    residue_chain_index:    torch.Tensor
    residue_atom37_bfactor: torch.Tensor
    
    def __init__(self):
        super().__init__()

    @classmethod
    def from_file(cls, path: str | Path, verbose: bool = True):
        if isinstance(path, str): path = Path(path)
        assert path.suffix.lower() in ['.cif', '.mmcif', '.pdb', '.gz'], f'Unsupported file type: {path.suffix}'
        stem = path.name.strip('.gz').strip('.cif').strip('.pdb')
        instance = cls()
        gemmi_out = _gemmi_parser(path, verbose=verbose)
        if gemmi_out == {}:
            instance.entry = 'empty'
            return instance
        
        instance.entry = stem
        instance.residue_atom37_coord   = gemmi_out['residue_atom37_coord']
        instance.residue_atom37_mask    = gemmi_out['residue_atom37_mask']
        instance.residue_mask           = gemmi_out['residue_mask']
        instance.residue_aatype         = gemmi_out['residue_aatype']
        instance.residue_index          = gemmi_out['residue_index']
        instance.residue_chain_index    = gemmi_out['residue_chain_index']
        instance.residue_atom37_bfactor = gemmi_out['residue_atom37_bfactor']
        return instance

    @classmethod
    def from_dict(cls, feature_in: Dict[str, Any]):
        instance = cls()
        instance.entry = feature_in.get('entry', 'unknown')
        fn = lambda k: feature_in[k] \
            if isinstance(feature_in[k], torch.Tensor) \
            else torch.from_numpy(feature_in[k])
        instance.residue_atom37_coord = fn('residue_atom37_coord')
        instance.residue_atom37_mask = fn('residue_atom37_mask')
        instance.residue_mask = fn('residue_mask')
        instance.residue_aatype = fn('residue_aatype')
        instance.residue_index = fn('residue_index')
        instance.residue_chain_index = fn('residue_chain_index')
        instance.residue_atom37_bfactor = fn('residue_atom37_bfactor')
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry': self.entry,
            'residue_atom37_coord': self.residue_atom37_coord.cpu().numpy(),
            'residue_atom37_mask': self.residue_atom37_mask.cpu().numpy(),
            'residue_mask': self.residue_mask.cpu().numpy(),
            'residue_aatype': self.residue_aatype.cpu().numpy(),
            'residue_index': self.residue_index.cpu().numpy(),
            'residue_chain_index': self.residue_chain_index.cpu().numpy(),
            'residue_atom37_bfactor': self.residue_atom37_bfactor.cpu().numpy(),
        }
    
    @classmethod
    def from_backbone(cls, backbone: OpenfoldBackbone):
        instance = cls()
        instance.entry = backbone.entry
        instance.residue_atom37_coord = backbone.residue_atom37_coord
        instance.residue_atom37_mask = backbone.residue_atom37_mask
        L, device = len(backbone), backbone.residue_atom37_coord.device
        instance.residue_mask = backbone.residue_atom37_mask[:, 1].to(dtype_template['residue_mask'])
        instance.residue_aatype = torch.zeros(L, device=device, dtype=dtype_template['residue_aatype'])
        instance.residue_index = torch.arange(L, device=device, dtype=dtype_template['residue_index'])
        instance.residue_chain_index = torch.zeros(L, device=device, dtype=dtype_template['residue_chain_index'])
        instance.residue_atom37_bfactor = torch.zeros(L, atom_type_num, device=device, dtype=dtype_template['residue_atom37_bfactor'])
        return instance
    
    def __len__(self) -> int:
        return self.residue_atom37_coord.size(0)
    
    def __str__(self) -> str:
        res_shortname = np.fromiter(
            map(lambda i: restypes_with_x[i], self.residue_aatype.cpu().numpy()),
            dtype='U1'
        )
        s = ''.join(res_shortname)
        return s
    
    def split(self) -> List['OpenfoldProtein']:
        # split a multi-chain protein into single-chain proteins
        unique_chain_idx = torch.unique(self.residue_chain_index)
        protein_list = []
        for cidx in unique_chain_idx:
            mask = self.residue_chain_index == cidx
            chain_name = pdb_chain_ids[int(cidx.item())]
            protein = OpenfoldProtein.from_dict(
                {
                    'entry': f'{self.entry}@{chain_name}',
                    'residue_atom37_coord': self.residue_atom37_coord[mask],
                    'residue_atom37_mask': self.residue_atom37_mask[mask],
                    'residue_mask': self.residue_mask[mask],
                    'residue_aatype': self.residue_aatype[mask],
                    'residue_index': self.residue_index[mask],
                    'residue_chain_index': self.residue_chain_index[mask],
                    'residue_atom37_bfactor': self.residue_atom37_bfactor[mask],
                }
            )
            protein_list.append(protein)
        return protein_list

    def to(self, device: str | torch.device):
        self.residue_atom37_coord = self.residue_atom37_coord.to(device)
        self.residue_atom37_mask = self.residue_atom37_mask.to(device)
        self.residue_mask = self.residue_mask.to(device)
        self.residue_aatype = self.residue_aatype.to(device)
        self.residue_index = self.residue_index.to(device)
        self.residue_chain_index = self.residue_chain_index.to(device)
        self.residue_atom37_bfactor = self.residue_atom37_bfactor.to(device)
        return self
    
    def to_pdb(self, path: str | Path): 
        if isinstance(path, str): path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdb_string = self.to_pdb_string()
        path.write_text(pdb_string)
        
    def to_pdb_string(self, model: int = 1, add_end: bool = True) -> str:
        res_1to3 = lambda r: restype_1to3.get(restypes_with_x[r], 'UNK')
        atom37_types = atom_types
        # pdb is a column format
        pdb_lines: list[str] = []
        atom_positions = self.residue_atom37_coord.cpu().numpy()
        atom_mask = self.residue_atom37_mask.cpu().numpy()
        residue_index = self.residue_index.cpu().numpy()
        chain_index = self.residue_chain_index.cpu().numpy()
        aatype = self.residue_aatype.cpu().numpy()
        b_factors = self.residue_atom37_bfactor.cpu().numpy()
        
        # simple format check  
        if np.any(aatype > restype_num):
            raise ValueError('Invalid aatypes.')
        chain_ids = {}
        for i in np.unique(chain_index):  # np.unique gives sorted output
            chain_ids[i] = pdb_chain_ids[i]

        # start with MODEL
        pdb_lines.append(f'MODEL     {model}')
        atom_index = 1
        last_chain_index = chain_index[0]
        for i in range(len(self)): # for each residue
            # Close the previous chain if in a multichain pdb
            if last_chain_index != chain_index[i]:
                chain_end = 'TER'
                pdb_lines.append(
                    f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[i - 1]):>3} '
                    f'{chain_ids[chain_index[i - 1]]:>1}{residue_index[i - 1]:>4}'
                )
                last_chain_index = chain_index[i]
                atom_index += 1  # Atom index increases at the TER symbol.

            res_name_3 = res_1to3(aatype[i])
            for atom37_name, pos, mask, b_factor in zip(atom37_types, atom_positions[i], atom_mask[i], b_factors[i]): # for atom37
                if mask < 0.5: continue
                record_type = 'ATOM'
                name = atom37_name if len(atom37_name) == 4 else f' {atom37_name}'
                alt_loc = ''
                insertion_code = ''
                occupancy = 1.00
                element = atom_types2element[atom37_name]
                charge = ''
                atom_line = (
                    f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                    f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                    f'{residue_index[i]:>4}{insertion_code:>1}   '
                    f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                    f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                    f'{element:>2}{charge:>2}'
                )
                pdb_lines.append(atom_line)
                atom_index += 1

        # close the final chain
        chain_end = 'TER'
        pdb_lines.append(
            f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[-1]):>3} '
            f'{chain_ids[chain_index[-1]]:>1}{residue_index[-1]:>4}'
        )
        pdb_lines.append('ENDMDL')
        if add_end: pdb_lines.append('END')
        # pad all lines to 80 characters.
        pdb_lines = [line.ljust(80) for line in pdb_lines]
        pdb_string = '\n'.join(pdb_lines) + '\n'  # add terminating newline
        return pdb_string

    def inherit(self, other: 'OpenfoldProtein'):
        # remain coordinate and mask, inherit other metadata, usually called after from_backbone()
        self.entry = other.entry
        # clear variable to release memory immediately
        self.residue_aatype = other.residue_aatype.clone().to(self.device)
        self.residue_index = other.residue_index.clone().to(self.device)
        self.residue_chain_index = other.residue_chain_index.clone().to(self.device)
        self.residue_atom37_bfactor = other.residue_atom37_bfactor.clone().to(self.device)

    def align_with(self, other: 'OpenfoldProtein', chain_wise: bool = False) -> Tuple[float, float, float]:
        assert len(self) == len(other), 'Length mismatch.'
        src_array = self.calpha.cpu().numpy()
        dst_array = other.calpha.cpu().numpy()
        # if chain-wise, split according to chain_index and calculate separately
        if chain_wise:
            assert torch.equal(self.residue_chain_index, other.residue_chain_index), 'Chain index mismatch. Call inherit() first ?'
            unique_chain_idx = torch.unique(self.residue_chain_index)
            tmscore_list, rmsd_l_list, rmsd_g_list = [], [], []
            for cidx in unique_chain_idx:
                cidx_mask = (self.residue_chain_index == cidx).cpu().numpy()
                src_c = src_array[cidx_mask]
                dst_c = dst_array[cidx_mask]
                # local alignment
                result = tmtools.tm_align(src_c, dst_c, 'A' * sum(cidx_mask), 'A' * sum(cidx_mask))
                tmscore_list.append(result.tm_norm_chain1)
                rmsd_l_list.append(result.rmsd)
                # global alignment
                rmsd_g_list.append(
                    rmsd_globally_aligned(
                        torch.tensor(src_c, device=self.device), torch.tensor(dst_c, device=self.device), atom_mask=self.residue_mask[cidx_mask]
                    )[0].item()
                )
            return float(np.mean(tmscore_list)), float(np.mean(rmsd_l_list)), float(np.mean(rmsd_g_list))
        else:
            result = tmtools.tm_align(src_array, dst_array, str(self), str(other))
            rmsd_g = rmsd_globally_aligned(
                torch.tensor(src_array, device=self.device), torch.tensor(dst_array, device=self.device), atom_mask=self.residue_mask
            )[0].item()
            return result.tm_norm_chain1, result.rmsd, rmsd_g
        
    def distance_matrix(self) -> torch.Tensor:
        return self.calpha[:, None, :] - self.calpha[None, :, :]
    
    def empty(self):
        return not hasattr(self, 'residue_atom37_coord')
    
    @property
    def num_chain(self) -> int:
        return torch.unique(self.residue_chain_index).numel()

    @property
    def device(self) -> torch.device:
        return self.residue_atom37_coord.device
    
    @property
    def calpha(self) -> torch.Tensor:
        ca_mask = self.residue_atom37_mask[:, atom_order['CA']]
        ca_coord = self.residue_atom37_coord[:, atom_order['CA'], :]
        impute_coord = ca_coord.new_zeros(ca_coord.shape)
        return ca_coord * ca_mask[:, None] + impute_coord * (1 - ca_mask[:, None])

    @property
    def cbeta(self) -> torch.Tensor:
        cb_mask = self.residue_atom37_mask[:, atom_order['CB']]
        cb_coord = self.residue_atom37_coord[:, atom_order['CB'], :]
        impute_coord = self.calpha
        return cb_coord * cb_mask[:, None] + impute_coord * (1 - cb_mask[:, None])
    
    @property
    def plddt(self) -> float:
        if not self.entry.startswith('AF'): return 0.0
        else:
            plddt_residue = self.residue_atom37_bfactor.sum(dim=1) / self.residue_atom37_mask.sum(dim=1) # [L]
            return plddt_residue.mean(dim=0).item()