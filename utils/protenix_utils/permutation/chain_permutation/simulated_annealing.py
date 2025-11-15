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

import itertools
import math
import random
from functools import partial

import torch

# from protenix.metrics.lddt_metrics import LDDT
# from protenix.model.loss import compute_lddt_mask
from utils.protenix_utils.logger import get_logger
from utils.protenix_utils.permutation.chain_permutation.utils import num_unique_matches
from utils.protenix_utils.permutation.utils import Checker, save_permutation_error
# from utils.torch_utils import cdist

from .heuristic import MultiChainPermutation as HeuristicMultiChainPermutation

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
    exhaustive_threshold: int = -1,
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
        exhaustive_threshold (int): threshold under which will use exhaustive search for chain permutation, otherwise will use heuristic search methods in AFMultimer.

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
            exhaustive_threshold=exhaustive_threshold,
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
                exhaustive_threshold=exhaustive_threshold,
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
    exhaustive_threshold: int = -1,
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
        ) = SAMultiChainPermutation()(
            pred_dict=pred_dict,
            label_full_dict=label_full_dict,
            max_num_chains=max_num_chains,
            exhaustive_threshold=exhaustive_threshold,
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


def _linear_annealing_schedule(current_iter, initial_temperature):
    return initial_temperature / (current_iter + 1)


def _exponential_annealing_schedule(current_iter, initial_temperature, alpha=0.9):
    return initial_temperature * (alpha**current_iter)


class SimulatedAnnealingChainPermutation:
    def __init__(
        self,
        lddt_base,
        initial_temp: int = 1000,
        max_iters: int = 1000,
        perm_groups: list = [],
        chain_perm_list: list = [],
        temp_schedule_configs: dict = {},
    ):
        self.lddt_base = lddt_base
        self.initial_temp = initial_temp
        self.max_iters = max_iters
        self.perm_groups = perm_groups
        self.chain_perm_list = chain_perm_list
        self.init_annealing_schedule(temp_schedule_configs)

    def init_annealing_schedule(self, temp_schedule_configs):
        annealing_schedule = temp_schedule_configs.pop("annealing_schedule", "linear")
        temp_schedule_configs.update({"initial_temperature": self.initial_temp})

        if annealing_schedule == "linear":
            self.annealing_schedule = partial(
                _linear_annealing_schedule, **temp_schedule_configs
            )
        elif annealing_schedule == "exponential":
            self.annealing_schedule = partial(
                _exponential_annealing_schedule, **temp_schedule_configs
            )
        else:
            raise NotImplementedError

    def acceptance_probability(self, delta_lddt, temp):
        """
        Acceptance probability of the current step
        """
        return torch.exp(torch.tensor(delta_lddt / temp)).item()

    def perturb_permutation(self, current_perm):
        """
        In the context of multi-chain permutation, a random pertrubation corresponds to randomly swap two chain assignments
        within a randomly-sampled permutation group.
        See Kim, Jin, Sakti Pramanik, and Moon Jung Chung. "Multiple sequence alignment using simulated annealing."
        Bioinformatics 10.4 (1994): 419-426. for a more comprehensive setting.
        """
        random_perm_group = random.sample(self.perm_groups, 1)[
            0
        ]  # [N_chain in the permutation group]
        assert (
            len(random_perm_group) >= 2
        ), "a permutation group should contain at least two elements"

        perm_key_1, perm_key_2 = random.sample(random_perm_group, 2)

        # Swap the chain values of the two keys
        new_chain_perm = current_perm.copy()
        new_chain_perm[perm_key_1], new_chain_perm[perm_key_2] = (
            current_perm[perm_key_2],
            current_perm[perm_key_1],
        )
        return new_chain_perm

    def __call__(
        self,
        pred_dict,
        label_full_dict,
        lddt_mask,
    ):
        # Initialize with a random permutation
        current_perm = random.choice(self.chain_perm_list)
        best_perm = current_perm
        perm_indices = SAMultiChainPermutation.build_permuted_indice(
            pred_dict, label_full_dict, best_perm
        )
        best_lddt = self.lddt_base(
            pred_coordinate=pred_dict["coordinate"],
            true_coordinate=label_full_dict["coordinate"][perm_indices],
            lddt_mask=lddt_mask[perm_indices][:, perm_indices],
        )
        temp = self.initial_temp

        # Simulated annealing loop
        for current_iter in range(self.max_iters):
            new_perm = self.perturb_permutation(current_perm)
            perm_indices = SAMultiChainPermutation.build_permuted_indice(
                pred_dict, label_full_dict, new_perm
            )
            new_lddt = self.lddt_base(
                pred_coordinate=pred_dict["coordinate"],
                true_coordinate=label_full_dict["coordinate"][perm_indices],
                lddt_mask=lddt_mask[perm_indices][:, perm_indices],
            )

            delta_lddt = new_lddt - best_lddt

            acceptance_prob = self.acceptance_probability(delta_lddt, temp)

            # Accept new solution based on probability
            if torch.rand(1).item() < acceptance_prob:
                current_perm = new_perm
                if new_lddt > best_lddt:
                    best_lddt = new_lddt
                    best_perm = new_perm

            # Anneal the temperature
            temp = self.annealing_schedule(current_iter)
        print(f"sa best lddt: {best_lddt}")

        return best_perm


class SAMultiChainPermutation(HeuristicMultiChainPermutation):
    """Exhaustive or simulated annealing method.
    Find the best match that maps predicted chains to chains in the true complex based on LDDT.
    Here we assume the predicted chains are uncropped.
    """

    def calc_total_perm(self):
        """
        Calculate the total number of asym_id permutations.

        Args:
            exhaustive_threshold (int): The threshold for the number of permutations. If the number of permutations exceeds this threshold,
            it will not proceed with computing the actual permutation indices.

        Returns:
            n_perm (int): The total number of permutations.
        """
        entity_to_asym_gt = self.label_token_dict["entity_to_asym"]
        asym_values = list(entity_to_asym_gt.values())
        n_perm = math.prod([math.factorial(torch.numel(val)) for val in asym_values])
        if n_perm > 1e4:
            print("warning: over 10000 chain permutations for this data!")
        perm_groups = [
            list(itertools.permutations(val.tolist())) for val in asym_values
        ]
        cartesian_permutations = itertools.product(*perm_groups)
        chain_perm_combination = [
            torch.tensor(list(itertools.chain(*perm_set)), device=asym_values[0].device)
            for perm_set in cartesian_permutations
        ]
        assert (
            chain_perm_combination
        ), "chain permutation should contain at least one permutation"

        perm_keys = chain_perm_combination[0]  # use the first permutation as the keys
        self.chain_perm_combination = [
            {
                perm_key.item(): perm_val.item()
                for perm_key, perm_val in zip(perm_keys, perm_vals)
            }
            for perm_vals in chain_perm_combination
        ]
        self.perm_groups = [
            perm_group[0] for perm_group in perm_groups if len(perm_group) > 1
        ]  # save perm groups for simulated_annealing perturbation steps
        return n_perm

    def compute_lddt_mask(self, pred_dict, label_full_dict):

        distance_mask = (
            label_full_dict["coordinate_mask"][..., None]
            * label_full_dict["coordinate_mask"][..., None, :]
        )

        distance = (
            cdist(label_full_dict["coordinate"], label_full_dict["coordinate"])
            * distance_mask
        ).to(
            label_full_dict["coordinate"].dtype
        )  # [..., N_atom, N_atom]

        lddt_mask = compute_lddt_mask(
            true_distance=distance,
            distance_mask=distance_mask,
            is_nucleotide=label_full_dict["is_rna"].bool()
            + label_full_dict["is_dna"].bool(),
            is_nucleotide_threshold=30.0,
            is_not_nucleotide_threshold=15.0,
        )

        del distance
        """
        TODO: precompute sparse pairwise dist? but it requires mapping the sparse permutation indices
        lddt_indices = torch.nonzero(lddt_mask, as_tuple=True)
        l_index = lddt_indices[0]
        m_index = lddt_indices[1]
        pred_distance_sparse_lm, true_distance_sparse_lm = (
            self.lddt_base._calc_sparse_dist(
                pred_dict["coordinate"], label_full_dict["coordinate"], l_index, m_index
            )
        )
        
        return pred_distance_sparse_lm, true_distance_sparse_lm, lddt_mask
        """
        return lddt_mask

    def compute_best_match_simulated_annealing(self, pred_dict, label_full_dict):

        # Compute LDDT masks
        lddt_mask = self.compute_lddt_mask(pred_dict, label_full_dict)
        best_match = SimulatedAnnealingChainPermutation(
            lddt_base=self.lddt_base,
            initial_temp=10,
            max_iters=256,
            perm_groups=self.perm_groups,
            chain_perm_list=self.chain_perm_combination,
            temp_schedule_configs={
                "annealing_schedule": "linear",
            },
        )(
            pred_dict=pred_dict,
            label_full_dict=label_full_dict,
            lddt_mask=lddt_mask,
        )

        return best_match

    def compute_best_match_exhaustive(self, pred_dict, label_full_dict):
        # Compute LDDT masks
        lddt_mask = self.compute_lddt_mask(pred_dict, label_full_dict)

        # Find best match based on lddt
        best_lddt = -torch.inf
        best_match = None
        for chain_perm in self.chain_perm_combination:
            perm_indices = self.build_permuted_indice(
                pred_dict, label_full_dict, chain_perm
            )
            curr_lddt = self.lddt_base(
                pred_coordinate=pred_dict["coordinate"],
                true_coordinate=label_full_dict["coordinate"][perm_indices],
                lddt_mask=lddt_mask[perm_indices][:, perm_indices],
            )
            if curr_lddt > best_lddt:
                best_lddt = curr_lddt
                best_match = chain_perm
        print(f"exhaustive best lddt: {best_lddt}")
        assert best_match is not None
        return best_match

    def __call__(
        self,
        pred_dict: dict[str, torch.Tensor],
        label_full_dict: dict[str, torch.Tensor],
        max_num_chains: int = 20,
        exhaustive_threshold: int = -1,
    ):
        match, has_sym_chain = self.process_input(
            pred_dict, label_full_dict, max_num_chains
        )

        if match is not None:
            # either the structure does not contain symmetric chains, or
            # there are too many chains so that the algorithm gives up.
            indices = self.build_permuted_indice(pred_dict, label_full_dict, match)
            pred_indices = torch.argsort(indices)
            return match, pred_indices, indices, {"has_sym_chain": False}

        # Core step: get best mol_id match
        n_gt_perm = self.calc_total_perm()
        self.lddt_base = LDDT()
        if n_gt_perm <= exhaustive_threshold:
            best_match = self.compute_best_match_exhaustive(pred_dict, label_full_dict)
        else:
            # Printing lddt for both exhaustive and sa methods
            best_match = self.compute_best_match_exhaustive(pred_dict, label_full_dict)
            best_match = self.compute_best_match_simulated_annealing(
                pred_dict, label_full_dict
            )

        permuted_indices = self.build_permuted_indice(
            pred_dict, label_full_dict, best_match
        )

        log_dict = {
            "has_sym_chain": True,
            "is_permuted": num_unique_matches([best_match, self.unpermuted_match]) > 1,
            "algo:no_permute": num_unique_matches([best_match, self.unpermuted_match])
            == 1,
            "n_gt_perm": n_gt_perm,
        }

        if log_dict["algo:no_permute"]:
            # return now
            pred_indices = torch.argsort(permuted_indices)
            return best_match, pred_indices, permuted_indices, log_dict

        # Compare rmsd before/after permutation
        unpermuted_indices = self.build_permuted_indice(
            pred_dict, label_full_dict, self.unpermuted_match
        )

        permuted_rmsd = self.aligned_rmsd(pred_dict, label_full_dict, permuted_indices)
        unpermuted_rmsd = self.aligned_rmsd(
            pred_dict, label_full_dict, unpermuted_indices
        )
        improved_rmsd = unpermuted_rmsd - permuted_rmsd
        if improved_rmsd >= 1e-12:
            # better
            log_dict.update(
                {
                    "algo:equivalent_permute": False,
                    "algo:worse_permute": False,
                    "algo:better_permute": True,
                    "algo:better_rmsd": improved_rmsd,
                }
            )
        elif improved_rmsd < 0:
            # worse
            log_dict.update(
                {
                    "algo:equivalent_permute": False,
                    "algo:worse_permute": True,
                    "algo:better_permute": False,
                    "algo:worse_rmsd": -improved_rmsd,
                }
            )
        elif not log_dict["algo:no_permute"]:
            # equivalent
            log_dict.update(
                {
                    "algo:equivalent_permute": True,
                    "algo:worse_permute": False,
                    "algo:better_permute": False,
                }
            )
        else:
            # no permute
            log_dict["debug:zero_rmsd"] = improved_rmsd

        # Revert worse/equivalent permute to original chain assignment
        if log_dict["algo:equivalent_permute"] or log_dict["algo:worse_permute"]:
            # Revert to original chain assignment
            best_match = self.unpermuted_match
            permuted_indices = unpermuted_indices
            log_dict["is_permuted"] = False

        if pred_dict["coordinate"].size(-2) == label_full_dict["coordinate"].size(-2):
            Checker.is_permutation(permuted_indices)  # indices to permute/crop label
            permute_pred_indices = torch.argsort(
                permuted_indices
            )  # indices to permute pred
        else:
            # hard to `define` permute_pred_indices in this case
            permute_pred_indices = None

        return best_match, permute_pred_indices, permuted_indices, log_dict
