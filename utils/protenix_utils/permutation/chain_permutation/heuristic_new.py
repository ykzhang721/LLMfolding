# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under Creative Commons Attribution-NonCommercial 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

# from protenix.metrics.lddt_metrics import LDDT
from utils.protenix_utils import (
    rmsd_globally_aligned,
    get_logger
)

from utils.protenix_utils.logger import get_logger
from utils.protenix_utils.permutation.chain_permutation.heuristic import (
    MultiChainPermutation as MultiChainPermutationBase,
)
from utils.protenix_utils.permutation.chain_permutation.utils import (
    apply_transform,
    get_optimal_transform,
    num_unique_matches,
)
from utils.protenix_utils.permutation.utils import Checker

logger = get_logger(__name__)

ExtraLabelKeys = [
    "pocket_mask",
    "interested_ligand_mask",
    "chain_1_mask",
    "chain_2_mask",
    "entity_mol_id",
    "mol_id",
    "mol_atom_index",
    "pae_rep_atom_mask",
]


def correct_symmetric_chains(
    pred_dict: dict,
    label_full_dict: dict,
    extra_label_keys: list[str] = ExtraLabelKeys,
    max_num_chains: int = 20,
    permute_label: bool = True,
    **kwargs,
):
    """Inputs

    Args:
        pred_dict (Dict[str, torch.Tensor]): A dictionary containing:
            - coordinate: pred_dict["coordinate"]
                shape = [N_cropped_atom, 3] or [Batch, N_cropped_atom, 3].
            - other keys: entity_mol_id, mol_id, mol_atom_index, pae_rep_atom_mask, is_ligand.
                shape = [N_cropped_atom]
        label_full_dict (Dict[str, torch.Tensor]): A dictionary containing
            - coordinate: label_full_dict["coordinate"] and label_full_dict["coordinate_mask"]
                shape = [N_atom, 3] and [N_atom] (for coordinate_mask)
            - other keys: entity_mol_id, mol_id, mol_atom_index, pae_rep_atom_mask.
                shape = [N_atom]
            - extra keys: keys specified by extra_feature_keys.
        extra_label_keys (list[str]):
            - Additional features in label_full_dict that should be returned along with the permuted coordinates.
        max_num_chains (int): if the number of chains is more than this number, than skip permutation to
            avoid expensive computations.
        permute_label (bool): if true, permute the groundtruth chains, otherwise premute the prediction chains

    Return:
        output_dict:
            If permute_label=True, this is a dictionary containing
            - coordinate
            - coordinate_mask
            - features specified by extra_label_keys.
            If permute_label=False, this is a dictionary containing
            - coordinate.

        log_dict: statistics.

        permute_pred_indices:
        permute_label_indices:
            If batch_mode, this is a list of LongTensor. Otherwise, this is a LongTensor.
            The LongTensor gives the indices to permute either prediction or label.
    """

    assert pred_dict["coordinate"].dim() in [2, 3]
    batch_mode = pred_dict["coordinate"].dim() > 2

    if not batch_mode:
        (
            best_match,
            permute_pred_indices,
            permute_label_indices,
            output_dict,
            log_dict,
        ) = _correct_symmetric_chains_for_one_sample(
            pred_dict,
            label_full_dict,
            max_num_chains,
            permute_label,
            extra_label_keys=extra_label_keys,
            **kwargs,
        )
        return output_dict, log_dict, permute_pred_indices, permute_label_indices
    else:
        assert not permute_label, "Only supports prediction permutations in batch mode."
        pred_coord = []
        log_dict = {}
        best_matches = []
        permute_pred_indices = []
        permute_label_indices = []
        # Loop over all samples to find best matches one by one
        for i, pred_coord_i in enumerate(pred_dict["coordinate"]):
            (
                best_match_i,
                permute_pred_indices_i,
                permute_label_indices_i,
                pred_dict_i,
                log_dict_i,
            ) = _correct_symmetric_chains_for_one_sample(
                {**pred_dict, "coordinate": pred_coord_i},
                label_full_dict,
                max_num_chains,
                permute_label=False,
                extra_label_keys=[],
                **kwargs,
            )

            best_matches.append(best_match_i)
            permute_pred_indices.append(permute_pred_indices_i)
            permute_label_indices.append(permute_label_indices_i)
            pred_coord.append(pred_dict_i["coordinate"])
            for key, value in log_dict_i.items():
                log_dict.setdefault(key, []).append(value)

        output_dict = {"coordinate": torch.stack(pred_coord, dim=0)}

        log_dict = {key: sum(value) / len(value) for key, value in log_dict.items()}
        log_dict["N_unique_perm"] = num_unique_matches(best_matches)

        return output_dict, log_dict, permute_pred_indices, permute_label_indices


def _correct_symmetric_chains_for_one_sample(
    pred_dict: dict,
    label_full_dict: dict,
    max_num_chains: int = 20,
    permute_label: bool = False,
    extra_label_keys: list[str] = [],
    **kwargs,
):

    if not permute_label:
        # Permutation will act on the predicted coordinate.
        # In this case, predicted structures and true structure need to have
        # the same number of atoms.
        assert pred_dict["coordinate"].size(-2) == label_full_dict["coordinate"].size(
            -2
        )

    with torch.no_grad():
        # Do not compute gradient while optimizing the permutation
        (
            best_match,
            permute_pred_indices,
            permute_label_indices,
            log_dict,
        ) = MultiChainPermutation(**kwargs)(
            pred_dict=pred_dict,
            label_full_dict=label_full_dict,
            max_num_chains=max_num_chains,
        )

    if permute_label:
        # Permute groundtruth coord and coord mask with indices, along the first dimension.
        indices = permute_label_indices.tolist()
        output_dict = {
            "coordinate": label_full_dict["coordinate"][indices, :],
            "coordinate_mask": label_full_dict["coordinate_mask"][indices],
        }
        # Permute extra label features, along the last dimension.
        output_dict.update(
            {
                k: label_full_dict[k][..., indices]
                for k in extra_label_keys
                if k in label_full_dict
            }
        )

    else:
        # Permute the predicted coord with permuted_indices
        indices = permute_pred_indices.tolist()
        output_dict = {
            "coordinate": pred_dict["coordinate"][indices, :],
        }

    return (
        best_match,
        permute_pred_indices,
        permute_label_indices,
        output_dict,
        log_dict,
    )


class MultiChainPermutation(MultiChainPermutationBase):

    def __init__(self, *args, **kwargs):
        self.selection_metric = kwargs.get("selection_metric", "aligned_rmsd")
        self.prioritize_polymer_anchor = kwargs.get("prioritize_polymer_anchor", False)
        self.verbose = kwargs.get("verbose", False)

        super(MultiChainPermutation, self).__init__(*args, **kwargs)

    @staticmethod
    def build_indices_from_match(
        pred_mol_id: torch.Tensor,
        pred_mol_atom_index: torch.Tensor,
        label_mol_id: torch.Tensor,
        label_mol_atom_index: torch.Tensor,
        pred_to_label_match: dict[int, int],
    ):
        """
        Args:
            pred_to_label_match (Dict[int, int]): {pred_mol_id: label_mol_id} a match between pred asym chains and label asym chains

        """

        N_pred_atom = pred_mol_id.size(0)
        N_label_atom = label_mol_id.size(0)

        indices_to_permute_label = pred_mol_id.new_zeros(size=(N_pred_atom,)).long()
        raw_label_indices = torch.arange(N_label_atom, device=label_mol_id.device)

        for p_mol_id, l_mol_id in pred_to_label_match.items():

            # Find common atoms in these two chains
            p_mask = pred_mol_id == p_mol_id
            l_mask = label_mol_id == l_mol_id

            p_mol_atom_index = pred_mol_atom_index[p_mask]
            l_mol_atom_index = label_mol_atom_index[l_mask]

            mask_1, mask_2 = MultiChainPermutation._get_common_atom_masks(
                p_mol_atom_index, l_mol_atom_index
            )
            p_mask[p_mask.clone()] *= mask_1
            l_mask[l_mask.clone()] *= mask_2

            indices_to_permute_label[p_mask] = raw_label_indices[l_mask].clone()

        assert len(torch.unique(indices_to_permute_label)) == len(
            indices_to_permute_label
        )
        if N_pred_atom == N_label_atom:
            indices_to_permute_pred = torch.argsort(indices_to_permute_label)
        else:
            indices_to_permute_pred = None  # Not well-defined
        return indices_to_permute_label, indices_to_permute_pred

    @staticmethod
    def _get_asym_ids_with_at_least_4_tokens(per_asym_dict: dict) -> list[int]:
        """
        Filtering Rules:
        1. The chain must contain at least 4 tokens.
        2. For groundtruth structure, the chain must contain at least 4 resolved tokens.

        Return:
            valid_asyms (list[int]): List of asym_id
        """
        valid_asyms = []
        for asym_id, asym_dict in per_asym_dict.items():
            if "coordinate_mask" not in asym_dict:
                if len(asym_dict["coordinate"]) >= 4:
                    valid_asyms.append(asym_id)
            else:
                if asym_dict["coordinate_mask"].sum().item() >= 4:
                    valid_asyms.append(asym_id)

        return valid_asyms

    @staticmethod
    def _asym_list_to_entity_dict(asym_id_list: list, asym_to_entity: dict):
        out = {}
        for asym_id in asym_id_list:
            entity_id = asym_to_entity[asym_id]
            if isinstance(entity_id, torch.Tensor):
                entity_id = entity_id.item()
            out.setdefault(entity_id, []).append(asym_id)
        return out

    def _prepare_anchor_pairs(self):
        """Filtering Rules:
        1. The paired chains have the same entity_id
        2. Both chains in the pair have at least 4 (resolved) tokens
        3. TODO:

        Returns:
            pred_anchor_entity_to_asym (Dict[int, list]):
                {entity_id_1: [asym_id_1, ..., asym_id_n],
                 entity_id_2: [asym_id_1, ..., asym_id_m], ...}
            label_anchor_entity_to_asym (Dict[int, list]):
                {entity_id_1: [asym_id_1, ..., asym_id_n],
                 entity_id_2: [asym_id_1, ..., asym_id_m], ...}
        """

        # Get asyms with at least 4 (resolved) tokens
        pred_anchor_asym_ids = self._get_asym_ids_with_at_least_4_tokens(
            self.pred_asym_dict
        )
        label_anchor_asym_ids = self._get_asym_ids_with_at_least_4_tokens(
            self.label_asym_dict
        )

        # Group by entity id
        pred_anchor_entity_to_asym = self._asym_list_to_entity_dict(
            pred_anchor_asym_ids, self.pred_token_dict["asym_to_entity"]
        )
        label_anchor_entity_to_asym = self._asym_list_to_entity_dict(
            label_anchor_asym_ids, self.label_token_dict["asym_to_entity"]
        )

        # Drop entity id if exists only at one side
        anchor_entity_ids = set(pred_anchor_entity_to_asym.keys()).intersection(
            set(label_anchor_entity_to_asym.keys())
        )

        if self.prioritize_polymer_anchor:
            # Find polymer entities
            polymer_entity_id = []
            for ent_id in anchor_entity_ids:
                mask = self.pred_token_dict["entity_mol_id"] == ent_id
                is_ligand = self.pred_token_dict["is_ligand"][mask]
                if torch.sum(is_ligand) <= is_ligand.shape[0] / 2:
                    polymer_entity_id.append(ent_id)
            if len(polymer_entity_id) > 0:  # Otherwise use ligands
                anchor_entity_ids = set(polymer_entity_id)

        for key in set(pred_anchor_entity_to_asym.keys()) - anchor_entity_ids:
            pred_anchor_entity_to_asym.pop(key)
        for key in set(label_anchor_entity_to_asym.keys()) - anchor_entity_ids:
            label_anchor_entity_to_asym.pop(key)

        # TODO: here add filterings for selecting a single pred anchor or label anchor

        return pred_anchor_entity_to_asym, label_anchor_entity_to_asym

    @staticmethod
    def _get_common_atom_masks(mol_atom_index_1, mol_atom_index_2):
        mask_1 = torch.isin(mol_atom_index_1, mol_atom_index_2)
        mask_2 = torch.isin(mol_atom_index_2, mol_atom_index_1)
        assert (
            mol_atom_index_1[mask_1] == mol_atom_index_2[mask_2]
        ).all()  # ensure the common atoms have the same order

        return mask_1, mask_2

    def get_best_chain_match_given_an_anchor_pair(
        self, pred_anchor_id, label_anchor_id
    ):
        anchor_i_dict = self.pred_asym_dict[pred_anchor_id]
        anchor_j_dict = self.label_asym_dict[label_anchor_id]

        """
        Align two anchor chains
        """
        # Restrict to atoms appeared in both chains
        # Use 'mol_atom_index' to find common atoms

        mask_i, mask_j = self._get_common_atom_masks(
            mol_atom_index_1=anchor_i_dict["mol_atom_index"],
            mol_atom_index_2=anchor_j_dict["mol_atom_index"],
        )
        coord_i = anchor_i_dict["coordinate"][mask_i]
        coord_j = anchor_j_dict["coordinate"][mask_j]
        resolved_mask = coord_j.new_ones(size=coord_j.shape[:-1]).bool()
        if "coordinate_mask" in anchor_i_dict:
            resolved_mask *= anchor_i_dict["coordinate_mask"][mask_i].bool()
        if "coordinate_mask" in anchor_j_dict:
            resolved_mask *= anchor_j_dict["coordinate_mask"][mask_j].bool()

        # Align label anchor (j) to pred anchor (i)
        if resolved_mask.sum() < 3:
            # Skip this pair since alignment is not well-defined for <3 atoms
            return None
        rot, trans = get_optimal_transform(
            coord_j[resolved_mask], coord_i[resolved_mask]
        )

        # Transform the full label structure according to alignment results
        aligned_coordinate = apply_transform(
            self.label_token_dict["coordinate"], rot, trans
        )
        for asym_id in self.label_asym_dict:
            self.label_asym_dict[asym_id]["aligned_coordinate"] = aligned_coordinate[
                self.label_token_dict["mol_id"] == asym_id
            ]

        """
        Greedily match remaining chains
        """
        matched = {pred_anchor_id: label_anchor_id}
        p_chains = [k for k in self.pred_asym_dict if k != pred_anchor_id]
        l_chains = [k for k in self.label_asym_dict if k != label_anchor_id]

        # Sort the chains by their length, so that longer chain chooses its match first.
        p_chains = sorted(
            p_chains,
            key=lambda k: -self.pred_asym_dict[k]["coordinate"].size(-2),
        )

        while len(p_chains) > 0:
            p_asym_id = p_chains.pop(0)
            p_ent_id = self.pred_token_dict["asym_to_entity"][p_asym_id]
            l_asym_ids = set(self.label_token_dict["entity_to_asym"][p_ent_id].tolist())
            l_asym_ids = l_asym_ids.intersection(set(l_chains))

            l_asym_id_matched, _ = self.match_pred_asym_to_gt_asym(
                p_asym_id, list(l_asym_ids)
            )
            matched[p_asym_id] = l_asym_id_matched
            l_chains.remove(l_asym_id_matched)

        assert len(matched) == len(self.pred_asym_dict)
        return matched

    @staticmethod
    def calc_aligned_rmsd(
        pred_coord, true_coord, coord_mask, reduce: bool = True, eps: float = 1e-8
    ):

        with torch.cuda.amp.autocast(enabled=False):
            aligned_rmsd, _, _, _ = rmsd_globally_aligned(
                pred_pose=pred_coord.to(torch.float32),
                true_pose=true_coord.to(torch.float32),
                atom_mask=coord_mask,
                allowing_reflection=False,
                reduce=reduce,
                eps=eps,
            )
        return aligned_rmsd.item()

    def error_of_one_match(self, pred_to_label_match) -> float:
        """
        Args:
            pred_to_label_match (Dict[int, int]): {pred_mol_id: label_mol_id} a match between pred asym chains and label asym chains
        """
        # use aligned rmsd of rep atoms
        indices_to_permute_label, _ = self.build_indices_from_match(
            pred_mol_id=self.pred_token_dict["mol_id"],
            pred_mol_atom_index=self.pred_token_dict["mol_atom_index"],
            label_mol_id=self.label_token_dict["mol_id"],
            label_mol_atom_index=self.label_token_dict["mol_atom_index"],
            pred_to_label_match=pred_to_label_match,
        )
        ErrorFunc = {
            "aligned_rmsd": Metrics.aligned_rmsd,
            "lddt": Metrics.lddt_neg,
        }
        error = ErrorFunc[self.selection_metric](
            pred_coord=self.pred_token_dict["coordinate"],
            true_coord=self.label_token_dict["coordinate"][indices_to_permute_label, :],
            coord_mask=self.label_token_dict["coordinate_mask"][
                indices_to_permute_label
            ],
            reduce=True,
        ).item()
        return error

    def compute_best_match_heuristic(self):

        # Prepare anchor pairs for alignment
        pred_anchor_entity_to_asym, label_anchor_entity_to_asym = (
            self._prepare_anchor_pairs()
        )

        # Enumerate all anchor pairs and get candidate matches
        matches = []
        for entity_id in pred_anchor_entity_to_asym:
            pred_asyms = pred_anchor_entity_to_asym[entity_id]
            label_asyms = label_anchor_entity_to_asym[entity_id]
            for anchor_i in pred_asyms:
                for anchor_j in label_asyms:

                    matched_ij = self.get_best_chain_match_given_an_anchor_pair(
                        pred_anchor_id=anchor_i, label_anchor_id=anchor_j
                    )

                    if self.verbose:
                        print(
                            f"entity={entity_id}, p_anchor={anchor_i}, l_anchor={anchor_j}"
                        )
                        print("matched: ", matched_ij)
                        print("\n")

                    if matched_ij:
                        matches.append(matched_ij)

        # Evaluate all matches and find the best
        errors = [self.error_of_one_match(match) for match in matches]
        best_idx = torch.argsort(torch.tensor(errors), stable=True)[0].item()
        best_match = matches[best_idx]
        return best_match


class Metrics:
    @staticmethod
    def aligned_rmsd(
        pred_coord, true_coord, coord_mask, reduce: bool = True, eps: float = 1e-8
    ):
        with torch.cuda.amp.autocast(enabled=False):
            rmsd, _, _, _ = rmsd_globally_aligned(
                pred_pose=pred_coord.to(torch.float32),
                true_pose=true_coord.to(torch.float32),
                atom_mask=coord_mask,
                allowing_reflection=False,
                reduce=reduce,
                eps=eps,
            )
        return rmsd

    # @staticmethod
    # def lddt(pred_coord, true_coord, coord_mask, reduce: bool = True):
    #     lddt_base = LDDT()
    #     lddt_mask = lddt_base.compute_lddt_mask(true_coord, coord_mask)
    #     lddt_val = lddt_base.forward(
    #         pred_coordinate=pred_coord, true_coordinate=true_coord, lddt_mask=lddt_mask
    #     )
    #     if reduce:
    #         return lddt_val.mean()
    #     else:
    #         return lddt_val

    @staticmethod
    def lddt_neg(*args, **kwargs):
        return -Metrics.lddt(*args, **kwargs)
