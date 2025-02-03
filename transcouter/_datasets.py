import torch
from torch.utils.data import Dataset
import numpy as np
import anndata as ad
from ._utils import to_np_array


class BalancedDataset(Dataset):
    def __init__(
        self,
        adata: ad.AnnData,
        ctrl_value: str,
        key_pert: str,
        key_embd_idx: str,
        key_stratify: str,
        shuffle: bool = True,
        seed: int = 24,
    ):
        """
        Initialize the Dataset where the number of control cells matches with that of perturbed cells.

        Parameters:
        adata:
            The adata object that contains all cells (e.g. control, and perturbed)
        ctrl_value:
            The value of the control condition
        key_pert:
            The column name of `adata.obs` that contains the perturbation condition
        key_embd_idx:
            The column name of `adata.obs` that contains the index of perturbed genes in the gene embedding matrix
        key_stratify:
            The column name of `adata.obs` that contains the stratification information, where sampling is performed within each group.
        shuffle:
            Whether to shuffle control cells and perturbed cells. Default is to True.
        seed : int, optional
            Random seed for reproducibility. Default is 24.
        """
        np.random.seed(seed)

        # Initialize attributes to hold aggregated data
        self.ctrl_expr = []
        self.pert_expr = []
        self.pert_idx = []
        self.ctrl_bcode = []
        self.pert_bcode = []

        # Stratify the data
        stratification = adata.obs[key_stratify].agg("_".join, axis=1)
        strat_rows = (
            stratification.groupby(stratification)
            .apply(lambda group: group.index.tolist())
            .to_dict()
        )

        # Iterate over each group
        for group, row_names in strat_rows.items():
            sub_adata = adata[row_names].copy()

            ctrl_adata = sub_adata[sub_adata.obs[key_pert] == ctrl_value]
            pert_adata = sub_adata[sub_adata.obs[key_pert] != ctrl_value]

            if len(pert_adata) == 0 or len(ctrl_adata) == 0:
                continue

            n_pert_cells = len(pert_adata)
            n_ctrl_cells = len(ctrl_adata)
            ctrl_idx = np.random.choice(np.arange(n_ctrl_cells), size=n_pert_cells, replace=True)  # fmt: skip
            sampled_ctrl_adata = ctrl_adata[ctrl_idx]

            ctrl_expr = to_np_array(sampled_ctrl_adata.X)
            pert_expr = to_np_array(pert_adata.X)
            pert_idx = np.vstack(pert_adata.obs[key_embd_idx].values)
            ctrl_bcode = sampled_ctrl_adata.obs.index.tolist()
            pert_bcode = pert_adata.obs.index.tolist()

            # Append to aggregated lists
            self.ctrl_expr.append(ctrl_expr)
            self.pert_expr.append(pert_expr)
            self.pert_idx.append(pert_idx)
            self.ctrl_bcode.extend(ctrl_bcode)
            self.pert_bcode.extend(pert_bcode)

        # Convert aggregated lists into numpy arrays for consistency
        self.ctrl_expr = np.vstack(self.ctrl_expr)
        self.pert_expr = np.vstack(self.pert_expr)
        self.pert_idx = np.vstack(self.pert_idx)
        self.n_pairs = len(self.pert_bcode)

        self.idx = np.arange(self.n_pairs)

        if shuffle:
            np.random.shuffle(self.idx)

        if len(self.pert_bcode) != len(self.ctrl_bcode):
            raise ValueError(
                "Number of Perturbed cells and Control cells must be equal."
            )

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        ctrl_expr = torch.tensor(self.ctrl_expr[self.idx[idx]], dtype=torch.float32)
        pert_expr = torch.tensor(self.pert_expr[self.idx[idx]], dtype=torch.float32)
        pert_idx = torch.tensor(self.pert_idx[self.idx[idx]], dtype=torch.long)
        ctrl_bcode = self.ctrl_bcode[self.idx[idx]]
        pert_bcode = self.pert_bcode[self.idx[idx]]
        return ctrl_expr, pert_expr, pert_idx, ctrl_bcode, pert_bcode
