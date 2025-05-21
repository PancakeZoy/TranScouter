import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import warnings
from pandas.errors import PerformanceWarning
from tqdm import tqdm
from ._utils import get_degs


class ScouterData:
    """
    Class for loading and processing perturbation data

    Attributes
    ----------
    adata: anndata.AnnData
        AnnData object containing all cells
    embd: pandas.DataFrame
        pandas dataframe containing gene embeddings
    key_pert: str
        The column name of `adata.obs` that corresponds to perturbation conditions
    key_cov: str | None = None
        The column name of `adata.obs` that corresponds to covariates (e.g. cell type, stimulation, etc.).
        If not provided, all cells are assumed to be perturbed under the same condition.
    key_cov_pert: str, optional
        The column name of `adata.obs` that corresponds to the combined covariate and perturbation condition. Default is 'cov_pert'.
    key_embd_idx: str
        The column name of `adata.obs` that corresponds to gene index in embedding matrix.
    key_var_gnames: str
        The column name of `adata.var` that corresponds to gene names.
    key_uns_DegNames: str, optional
        The key used to store the name of ranked genes in `adata.uns`. Default is 'top_degs_names'.
    key_uns_DegIdx: str, optional
        The key used to store the index of ranked genes in `adata.uns`. Default is 'top_degs_idx'.
    key_uns_NonzeroIdx: str, optional
        The key used to store the index of non-zero genes for each condition in `adata.uns`. Default is 'nonzero_gene_idx'.
    ctrl_value: str
        The value of `adata.obs[key_pert]` that corresponds to control cells.
    matched_genes: list
        A list of matched genes between adata and embd
    train_conds: list
        List of perturbation conditions in the train split.
    train_adata: anndata.AnnData
        AnnData object for the train split
    val_conds: list
        List of perturbation conditions in the validation split.
    val_adata: anndata.AnnData
        AnnData object for the validation split
    test_conds: list
        List of perturbation conditions in the test split.
    test_adata: anndata.AnnData
        AnnData object for the test split
    """

    def __init__(
        self,
        adata: ad.AnnData,
        embd: pd.DataFrame,
        key_pert: str,
        key_var_gnames: str,
        key_cov: str | None = None,
        ctrl_value: str = "ctrl",
    ):
        """
        Initialize the ScouterData object.

        Parameters:
        ----------
        adata: anndata.AnnData
            Annotated data object. `adata.obs` must contain a column `key_pert` with required format:
                `'ctrl'` for control cells
                `'geneA+geneB'` or `'geneA+ctrl'` to denote the name of gene(s) perturbed.
        embd: pandas.DataFrame
            Gene embedding matrix, with gene names as row names.
        key_pert: str
            The column name of `adata.obs` that corresponds to perturbation conditions.
        key_var_gnames: str
            The column name of `adata.var` that corresponds to gene names.
        key_cov: str | None = None
            The column name of `adata.obs` that corresponds to covariates (e.g. cell type, stimulation, etc.).
            If not provided, all cells are assumed to be perturbed under the same condition.
        ctrl_value: str, optional
            The value of `adata.obs[key_pert]` that corresponds to control cells. Default is 'ctrl'.
        """

        if not isinstance(adata, ad.AnnData):
            raise TypeError("adata must be an AnnData object")
        if not isinstance(embd, pd.DataFrame):
            raise TypeError("embd must be an pandas DataFrame")

        self.adata = adata.copy()
        self.embd = embd.copy()
        self.key_pert = key_pert
        self.key_var_gnames = key_var_gnames
        self.key_cov = key_cov
        self.ctrl_value = ctrl_value

    def setup_ad(self, key_embd_idx: str = "embd_idx", slim: bool = True):
        """
        Setup `adata` and `embd`.
        `embd` will be filtered to only contain the matched genes.
        `adata` will drop the perturbation conditions not covered by matched genes.
        A new column `key_embd_idx` will be added to `adata.obs`, denoting the index of perturbed genes in `embd`.

        Parameters:
        ----------
        key_embd_idx: str, optional
            The column name of `adata.obs` that corresponds to gene index in the embedding matrix. Default is 'embd_idx'.
        slim: bool, optional
            Whether to filter the embedding matrix to only contain perturbed genes. Default is True.
        """
        self.key_embd_idx = key_embd_idx

        # Two gene name variables: 1. Embedding matrix rownames; 2. Notation of perturbed genes
        gene_name_embd = self.embd.index.tolist()
        uniq_conds = self.adata.obs[self.key_pert].unique().tolist()
        if self.ctrl_value not in uniq_conds:
            raise TypeError("Provided annData does not have control cells")
        gene_name_pert = np.unique(sum([p.split("+") for p in uniq_conds], []))

        # Find the matched genes between 1 and 2.
        matched_genes = sorted(list(np.intersect1d(gene_name_embd, gene_name_pert)))
        if self.ctrl_value not in matched_genes:
            print(f"Control condition '{self.ctrl_value}' not found in gene embedding matrix. Adding a row of zeros for control.")  # fmt: skip
            ctrl_row = pd.DataFrame(
                [np.zeros(self.embd.shape[1])],
                columns=self.embd.columns,
                index=[self.ctrl_value],
            )
            self.embd = pd.concat([ctrl_row, self.embd])
            matched_genes = [self.ctrl_value] + matched_genes

        # slim embedding matrix to only contain matched genes if 'slim' is True
        if slim:
            self.embd = self.embd.loc[matched_genes]

        # Detect cells with perturbed genes not in matched genes
        unmatched_genes = gene_name_pert[
            [p not in matched_genes for p in gene_name_pert]
        ]
        if len(unmatched_genes) > 0:
            print(
                f"{len(unmatched_genes)} perturbed genes are not found in the gene embedding matrix: \n{unmatched_genes}. \nHence they are deleted. Please check if this is because of different gene synonyms. "  # fmt: skip
            )
            # Filter the DataFrame by excluding rows where the condition contains unmatched genes
            delete = (
                self.adata.obs[self.key_pert]
                .str.split("+")
                .apply(lambda x: any(i in unmatched_genes for i in x))
            )
            print(f"Please check if the deletion of following conditions are correct: \n{sorted(list(self.adata[delete].obs[self.key_pert].unique()))}")  # fmt: skip
            self.adata = self.adata[~delete].copy()
            uniq_conds = self.adata.obs[self.key_pert].unique().tolist()
            gene_name_pert = np.unique(sum([p.split("+") for p in uniq_conds], []))
        else:
            print(f"All {len(gene_name_pert)} perturbed genes are found in the gene embedding matrix!")  # fmt: skip

        # Create a new column that contains the index of perturbed genes in embd matrix
        embd_names = self.embd.index.tolist()
        gene_ind_dic = {g: embd_names.index(g) for g in gene_name_pert}
        cond_ind_dic = {
            cond: [gene_ind_dic[gene] for gene in cond.split("+")]
            for cond in uniq_conds
        }
        if self.adata.is_view:
            self.adata = self.adata.copy()
        self.adata.obs[key_embd_idx] = (
            self.adata.obs[self.key_pert].astype(str).map(cond_ind_dic)
        )
        self.matched_genes = matched_genes
        self.unmatched_genes = unmatched_genes

    def split_Train_Val_Test(
        self,
        train_conds=None,
        val_conds=None,
        test_conds=None,
    ):
        """
        Splits the annotated data into training, validation, and testing sets.
        Please pass in the covariate x perturbation combination as the `val_conds` and `test_conds`
        Do not include the control conditions.

        Parameters:
        ----------
        val_conds: list or None, optional
            List of perturbation conditions to be the validation set. If None, conditions are selected randomly based on `val_ratio`. Default is None.
        val_ratio: float, optional
            The proportion of the validation split compared to the training split. Default is 0.1.
        if_test: bool, optional
            Whether to generate a split for testing. Default is True.
        test_conds: list or None, optional
            List of perturbation conditions to be the test set. If None, conditions are selected randomly based on `test_ratio`. Default is None.
        test_ratio: float, optional
            The proportion of the test split compared to the rest. Default is 0.2.
        seed: int, optional
            Random seed for reproducibility. Default is 24.
        """

        self.train_conds = np.array(train_conds)
        self.val_conds = np.array(val_conds)
        self.test_conds = np.array(test_conds)

        self.val_adata = self.adata[
            (self.adata.obs[self.key_cov_pert].isin(val_conds))
            | (self.adata.obs[self.key_pert] == self.ctrl_value)
        ]
        self.test_adata = self.adata[
            (self.adata.obs[self.key_cov_pert].isin(test_conds))
            | (self.adata.obs[self.key_pert] == self.ctrl_value)
        ]
        self.train_adata = self.adata[
            (self.adata.obs[self.key_cov_pert].isin(self.train_conds))
            | (self.adata.obs[self.key_pert] == self.ctrl_value)
        ]

    def gene_ranks(
        self,
        key_cov_pert: str = "cov_pert",
        delim: str = "_",
        genes_keep: int = 25,
        pval_cutoff: float = 0.05,
        log2fc_min: float = 1.5,
        log2fc_max: float = -1.5,
        rankby_abs: bool = True,
        method: str = "wilcoxon",
        key_uns_DegNames: str = "top_degs_names",
        key_uns_DegIdx: str = "top_degs_idx",
        **kwargs,
    ):
        """
        Rank genes for each perturbation group, controlled by `self.key_cov`. Saved as a dictionary in `adata.uns[key_uns_DegNames]`.

        Parameters:
        ----------
        key_cov_pert: str, optional
            The column name of `adata.obs` that corresponds to the combined covariate and perturbation condition. Default is 'cov_pert'.
        delim: str, optional
            The delimiter used to concatenate the covariate and perturbation condition. Default is '_'.
        genes_keep: int, optional
            The number of genes to return for each group. Default is 25.
        pval_cutoff: float, optional
            P-value threshold for statistical significance. Default is 0.05.
        log2fc_min: float, optional
            Minimum log2 fold change threshold for up-regulated genes. Default is 1.5.
        log2fc_max: float, optional
            Maximum log2 fold change threshold for down-regulated genes. Default is -1.5.
        rankby_abs: bool, optional
            Rank genes by the absolute value of the score, not by the score. The returned scores are never the absolute values. Default is True.
        method: str, optional
            The method used to rank genes. Default is 'wilcoxon'.
        key_uns_DegNames: str, optional
            The key used to store the name of ranked genes in `adata.uns`. Default is 'top_degs_names'.
        key_uns_DegIdx: str, optional
            The key used to store the index of ranked genes in `adata.uns`. Default is 'top_degs_idx'.
        kwargs: dict, optional
            All additional keyword arguments are passed to the `scanpy.tl.rank_genes_groups` call.
        """
        self.key_cov_pert = key_cov_pert
        self.key_uns_DegNames = key_uns_DegNames
        self.key_uns_DegIdx = key_uns_DegIdx

        deg_names = {}
        if self.key_cov is None:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=PerformanceWarning)
                temp_adata = self.adata.copy()
                sc.tl.rank_genes_groups(
                    temp_adata,
                    groupby=self.key_pert,
                    reference=self.ctrl_value,
                    rankby_abs=rankby_abs,
                    method=method,
                    **kwargs,
                )
            degs = get_degs(temp_adata, pval_cutoff=pval_cutoff, log2fc_min=log2fc_min, log2fc_max=log2fc_max, genes_keep=genes_keep)  # fmt: skip
            deg_names.update(degs)
            self.adata.obs[self.key_cov_pert] = self.adata.obs[self.key_pert]
        else:
            for unq_cov in tqdm(self.adata.obs[self.key_cov].unique()):
                subset = self.adata[self.adata.obs[self.key_cov] == unq_cov].copy()
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=PerformanceWarning)
                    sc.tl.rank_genes_groups(
                        subset,
                        groupby=self.key_pert,
                        reference=self.ctrl_value,
                        rankby_abs=rankby_abs,
                        method=method,
                        **kwargs,
                    )
                degs = get_degs(subset, pval_cutoff=pval_cutoff, log2fc_min=log2fc_min, log2fc_max=log2fc_max, genes_keep=genes_keep)  # fmt: skip
                degs = {f"{unq_cov}_{key}": value for key, value in degs.items()}
                deg_names.update(degs)
            self.adata.obs[self.key_cov_pert] = self.adata.obs[[self.key_cov, self.key_pert]].agg(delim.join, axis=1)  # fmt: skip

        deg_indices = {
            cov_pert: [self.adata.var_names.get_loc(gene) for gene in genes]
            for cov_pert, genes in deg_names.items()
        }
        self.adata.uns[self.key_uns_DegNames] = deg_names
        self.adata.uns[self.key_uns_DegIdx] = deg_indices

    def get_nonzero_genes(
        self,
        key_added: str = "nonzero_gene_idx",
        delim: str = "_",
        strict: bool = True,
        threshold: int = 5,
    ):
        """
        Compute the non-zero genes for each perturbation condition.

        Parameters:
        ----------
        key_added: str, optional
            The key used to store the index of non-zero genes in `adata.uns`. Default is 'nonzero_gene_idx'.
        delim: str, optional
            The delimiter used to concatenate the covariate and perturbation condition. Default is '_'.
        strict: bool, optional
            Whether to consider only the non-zero genes. Default is False.
        threshold: int, optional
            The threshold for defining non-zero genes. Default is being detected in at least 5 cells.
        """
        self.key_uns_NonzeroIdx = key_added
        non_zeros_gene_idx = {}
        covs_pert = self.adata.obs[self.key_cov_pert]
        covs_pert = covs_pert[~covs_pert.str.contains(self.ctrl_value)].unique()

        for cov_pert in tqdm(covs_pert):
            cov, pert = cov_pert.split(delim)
            subset = self.adata[self.adata.obs[self.key_cov] == cov].copy()

            X = subset[subset.obs[self.key_pert].isin([pert, self.ctrl_value])].X
            X_nonzero = np.array(np.sum(X != 0, axis=0)).squeeze()
            nonzero_idx = np.where(X_nonzero > (0 if strict else threshold))[0]

            non_zeros_gene_idx[cov_pert] = nonzero_idx

        self.adata.uns[self.key_uns_NonzeroIdx] = non_zeros_gene_idx
