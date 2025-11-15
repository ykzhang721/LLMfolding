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

import traceback
from typing import Any, Optional, Union

import torch

from utils.protenix_utils.logger import get_logger
from utils.protenix_utils.permutation.utils import save_permutation_error

from .heuristic import (
    correct_symmetric_chains as correct_symmetric_chains_heuristic_old,
)
from .heuristic_new import (
    correct_symmetric_chains as correct_symmetric_chains_heuristic,
)
from .pocket_based_permutation import permute_pred_to_optimize_pocket_aligned_rmsd

logger = get_logger(__name__)


def run(
    pred_coord: torch.Tensor,
    input_feature_dict: dict[str, Any],
    label_full_dict: dict[str, Any],
    max_num_chains: int = -1,
    permute_label: bool = True,
    permute_by_pocket: bool = False,
    error_dir: Optional[str] = None,
    **kwargs,
):

    enumerate_all_anchor_pairs = kwargs.pop("enumerate_all_anchor_pairs", False)

    if pred_coord.dim() > 2:
        assert (
            permute_label is False
        ), "Only supports prediction permutations in batch mode."

    try:
        if permute_by_pocket:
            """optimize the chain assignment on pocket-ligand interface"""
            assert not permute_label

            if label_full_dict["pocket_mask"].dim() == 2:
                # first pocket is the `main` pocket
                pocket_mask = label_full_dict["pocket_mask"][0]
                ligand_mask = label_full_dict["interested_ligand_mask"][0]
            else:
                pocket_mask = label_full_dict["pocket_mask"]
                ligand_mask = label_full_dict["interested_ligand_mask"]

            permute_pred_indices, permuted_aligned_pred_coord, log_dict = (
                permute_pred_to_optimize_pocket_aligned_rmsd(
                    pred_coord=pred_coord,
                    true_coord=label_full_dict["coordinate"],
                    true_coord_mask=label_full_dict["coordinate_mask"],
                    true_pocket_mask=pocket_mask,
                    true_ligand_mask=ligand_mask,
                    atom_entity_id=input_feature_dict["entity_mol_id"],
                    atom_asym_id=input_feature_dict["mol_id"],
                    mol_atom_index=input_feature_dict["mol_atom_index"],
                    use_center_rmsd=kwargs.get("use_center_rmsd", False),
                )
            )
            output_dict = {"coordinate": permuted_aligned_pred_coord}
            permute_label_indices = []

        else:
            """optimize the chain assignment on all chains"""
            if enumerate_all_anchor_pairs:
                heuristic_func = correct_symmetric_chains_heuristic
            else:
                heuristic_func = correct_symmetric_chains_heuristic_old
            output_dict, log_dict, permute_pred_indices, permute_label_indices = (
                heuristic_func(
                    pred_dict={**input_feature_dict, "coordinate": pred_coord},
                    label_full_dict=label_full_dict,
                    max_num_chains=max_num_chains,
                    permute_label=permute_label,
                    **kwargs,
                )
            )

    except Exception as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        logger.warning(error_message)
        save_permutation_error(
            data={
                "error_message": error_message,
                "pred_dict": {**input_feature_dict, "coordinate": pred_coord},
                "label_full_dict": label_full_dict,
                "max_num_chains": max_num_chains,
                "permute_label": permute_label,
                "dataset_name": input_feature_dict.get("dataset_name", None),
                "pdb_id": input_feature_dict.get("pdb_id", None),
            },
            error_dir=error_dir,
        )
        output_dict, log_dict, permute_pred_indices, permute_label_indices = (
            {},
            {},
            [],
            [],
        )

    return output_dict, log_dict, permute_pred_indices, permute_label_indices
