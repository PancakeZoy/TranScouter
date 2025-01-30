from ._utils import test_Dataset, total_loss
from ._model import TranScouterModel
from ._datasets import BalancedDataset
from .scouterdata import ScouterData
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr


class TranScouter:
    """
    TranScouter model class

    Attributes
    ----------
    all_adata: anndata.AnnData
        AnnData object for the entire dataset
    train_adata: anndata.AnnData
        AnnData object for the train split
    val_adata: anndata.AnnData
        AnnData object for the validation split
    test_adata: anndata.AnnData
        AnnData object for the test split
    ctrl_adata: anndata.AnnData
        AnnData object for the control cells
    embd_tensor: torch.tensor
        torch.tensor object of the gene embedding matrix
    key_pert: str
        The column name of `adata.obs` that corresponds to perturbation conditions
    key_cov: str | None = None
        The column name(s) of `adata.obs` that corresponds to covariates (e.g. cell type, stimulation, etc.).
        If not provided, all cells are assumed to be perturbed under the same condition.
    key_cov_pert: str, optional
        The column name of `adata.obs` that corresponds to the combined covariate and perturbation condition. Default is 'cov_pert'.
    key_stratify : str
        Stores the value of `key_stratify` for use in stratified pairing.
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
    n_genes: int
        Number of genes in the cell expression
    network: TranScouterModel
        The model achieves minimal validation loss after training
    loss_history: dict
        Dictionary containing the loss history on both train split and validation split
    """

    def __init__(self, pertdata: ScouterData, device: str = "auto"):
        """
        Parameters
        ----------
        - pertdata:
            An ScouterData Object containing cell expression anndata and gene embedding matrix
        - device:
            Device to run the model on. Default: 'auto'
        """

        if not isinstance(pertdata, ScouterData):
            raise TypeError("`pertdata` must be an ScouterData object")

        self.embd_idx_dict = {gene: i for i, gene in enumerate(pertdata.embd.index)}
        self.embd_tensor = torch.tensor(pertdata.embd.values, dtype=torch.float32)

        self.key_pert = pertdata.key_pert
        self.key_cov = pertdata.key_cov
        self.key_cov_pert = pertdata.key_cov_pert
        self.key_embd_idx = pertdata.key_embd_idx
        self.key_var_gnames = pertdata.key_var_gnames
        self.key_uns_DegNames = pertdata.key_uns_DegNames
        self.key_uns_DegIdx = pertdata.key_uns_DegIdx
        self.key_uns_NonzeroIdx = pertdata.key_uns_NonzeroIdx
        self.ctrl_value = pertdata.ctrl_value
        self.n_genes = pertdata.train_adata.shape[1]

        self.all_adata = pertdata.adata
        self.train_adata = pertdata.train_adata
        self.val_adata = pertdata.val_adata
        self.test_adata = pertdata.test_adata
        self.ctrl_adata = pertdata.adata[pertdata.adata.obs[self.key_pert] == self.ctrl_value]  # fmt: skip

        # Determine the device
        if device == "auto":
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                self.device = torch.device("cuda:" + str(current_device))
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def model_init(
        self,
        hidden_enc_embd=(512, 256),
        hidden_enc_ctrl=(2048, 1024),
        bottle_dim=512,
        hidden_dec=(1024, 2048),
        use_batch_norm=True,
        use_layer_norm=False,
        dropout_rate=0.0,
        use_sampling=False,
        condition="embedding",
    ):
        """
        Initialize the TranScouterModel.

        Parameters:
        ----------
        - hidden_enc_embd:
            Tuple specifying the hidden layer sizes for the gene embedding encoder.
        - hidden_enc_ctrl:
            Tuple specifying the hidden layer sizes for the condition (control cell expression) encoder.
        - bottle_dim:
            Size of the bottleneck space.
        - hidden_dec:
            Tuple specifying the hidden layer sizes for the decoder.
        - use_batch_norm:
            Whether to use batch normalization.
        - use_layer_norm:
            Whether to use layer normalization.
        - dropout_rate:
            Dropout rate.
        - use_sampling:
            Whether to use the reparameterization trick (i.e., VAE sampling) or skip it.
        - condition:
            Whether to condition on the control expression or the gene embedding.

        Attributes Created
        ------------------
        network : TranScouterModel
            The initialized model for perturbation prediction.
        best_val_loss : float
            Tracks the best validation loss during training. Initially set to infinity.
        loss_history : dict
            Stores training and validation loss history in `train_loss` and `val_loss` lists.
        """
        self.network = TranScouterModel(
            ctrl_dim=self.n_genes,
            gene_embd=self.embd_tensor,
            hidden_enc_embd=hidden_enc_embd,
            hidden_enc_ctrl=hidden_enc_ctrl,
            bottle_dim=bottle_dim,
            hidden_dec=hidden_dec,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            use_sampling=use_sampling,
            condition=condition,
        ).to(self.device)
        self.best_val_loss = np.inf
        self.loss_history = {"train_loss": [], "val_loss": []}

    def data_init(self, key_stratify: str, pairing_seed: int = 24):
        """
        Initializes BalancedDataset objects for the train, validation, and test splits.

        This function also sets the `key_stratify` attribute in the class instance,
        which is used for stratified pairing of perturbed cells with control cells.

        Parameters
        ----------
        key_stratify : str
            The column name in `adata.obs` used for random pairing of perturbed cells with control cells.
            Pairing is performed within each group specified by this column. This value is also stored
            as the `key_stratify` attribute in the class instance.
        pairing_seed : int, optional
            The random seed for reproducibility in the pairing process. Defaults to 24.

        Attributes
        ----------
        key_stratify : str
            Stores the value of `key_stratify` for use in stratified pairing.
        train_dataset : BalancedDataset
            The dataset for training with stratified pairing.
        val_dataset : BalancedDataset
            The dataset for validation with stratified pairing.
        test_dataset : BalancedDataset
            The dataset for testing with stratified pairing.

        Raises
        ------
        ValueError
            If the BalancedDataset objects do not pass the test.
        """
        self.key_stratify = key_stratify
        self.train_dataset = BalancedDataset(
            adata=self.train_adata,
            ctrl_value=self.ctrl_value,
            key_pert=self.key_pert,
            key_embd_idx=self.key_embd_idx,
            key_stratify=self.key_stratify,
            shuffle=True,
            seed=pairing_seed,
        )
        self.val_dataset = BalancedDataset(
            adata=self.val_adata,
            ctrl_value=self.ctrl_value,
            key_pert=self.key_pert,
            key_embd_idx=self.key_embd_idx,
            key_stratify=self.key_stratify,
            shuffle=False,
            seed=pairing_seed,
        )
        self.test_dataset = BalancedDataset(
            adata=self.test_adata,
            ctrl_value=self.ctrl_value,
            key_pert=self.key_pert,
            key_embd_idx=self.key_embd_idx,
            key_stratify=self.key_stratify,
            shuffle=False,
            seed=pairing_seed,
        )

        if all(
            [
                test_Dataset(self.train_dataset, self.train_adata),
                test_Dataset(self.val_dataset, self.val_adata),
                test_Dataset(self.test_dataset, self.test_adata),
            ]
        ):
            print("BalancedDataset passed the test")
        else:
            raise ValueError("BalancedDataset did not pass the test :)")

    def train(
        self,
        batch_size: int = 512,
        w_norm: float = 0.0,
        w_direction: float = 0.5,
        w_kld: float = 0.05,
        lr: float = 0.001,
        sched_gamma: float = 0.9,
        n_epochs: int = 40,
        patience: int = 5,
        if_nonzero: bool = True,
    ):
        """
        Trains the model with the given parameters.

        Note: you must call `data_init(...)` first to initialize the train and validation datasets.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples per batch, by default 256.
        w_norm : float, optional
            Scaling factor for loss sensitivity to large expression changes, by default 0.0.
        w_direction : float, optional
            Weight of the direction-aware loss in the total loss, by default 0.5.
        w_kld : float, optional
            Weight of the KL divergence loss, by default 0.05.
        lr : float, optional
            Learning rate, by default 0.001.
        sched_gamma : float, optional
            Multiplicative factor of learning rate decay in ExponentialLR, by default 0.9.
        n_epochs : int, optional
            Maximum number of epochs to train, by default 40.
        patience : int, optional
            Number of epochs with no improvement on validation loss before early stopping, by default 5.
        if_nonzero : bool, optional
            Whether to compute loss only for non-zero genes. If False, loss will be computed for all genes.

        Attributes Updated
        ------------------
        network : TranScouterModel
            Updated model parameters after training.
        loss_history : dict
            Stores loss values for `train_loss` and `val_loss` per epoch.
        best_val_loss : float
            Stores the best validation loss observed during training.
        """
        if if_nonzero:
            nonzero_idx_dict = self.all_adata.uns[self.key_uns_NonzeroIdx]
        else:
            nonzero_idx_dict = None

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )
        val_loader = (
            DataLoader(self.val_dataset, batch_size=batch_size, drop_last=False)
            if len(self.val_dataset) > 0
            else None
        )

        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_gamma)

        epochs_no_improve = 0
        best_model_state_dict = None

        for epoch in range(n_epochs):
            # Training phase
            self.network.train()
            train_loss = 0.0
            train_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{n_epochs} - Training Batches",
                leave=True,
                unit="batch",
            )
            for ctrl_expr, pert_expr, pert_idx, ctrl_bcode, pert_bcode in train_progress:  # fmt: skip
                ctrl_expr, pert_expr, pert_idx = ctrl_expr.to(self.device), pert_expr.to(self.device), pert_idx.to(self.device)  # fmt: off
                group = (self.train_adata[pert_bcode, :].obs[self.key_cov_pert].values.tolist())  # fmt: skip
                optimizer.zero_grad()
                pred_expr, mu, logvar = self.network(pert_idx, ctrl_expr)
                loss = total_loss(
                    pred_expr=pred_expr,
                    true_expr=pert_expr,
                    ctrl_expr=ctrl_expr,
                    group=group,
                    nonzero_idx_dict=nonzero_idx_dict,
                    w_norm=w_norm,
                    w_direction=w_direction,
                    mu=mu,
                    logvar=logvar,
                    w_kld=w_kld,
                    use_sampling=self.network.use_sampling,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            if val_loader:
                # Validation phase
                self.network.eval()
                val_loss = 0.0
                val_progress = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{n_epochs} - Validation Batches",
                    leave=True,
                    unit="batch",
                )
                with torch.no_grad():
                    for ctrl_expr, pert_expr, pert_idx, ctrl_bcode, pert_bcode in val_progress:  # fmt: skip
                        ctrl_expr, pert_expr, pert_idx = ctrl_expr.to(self.device), pert_expr.to(self.device), pert_idx.to(self.device)  # fmt: off
                        group = (self.val_adata[pert_bcode, :].obs[self.key_cov_pert].values.tolist())  # fmt: skip
                        pred_expr, mu, logvar = self.network(pert_idx, ctrl_expr)
                        loss = total_loss(
                            pred_expr=pred_expr,
                            true_expr=pert_expr,
                            ctrl_expr=ctrl_expr,
                            group=group,
                            nonzero_idx_dict=nonzero_idx_dict,
                            w_norm=w_norm,
                            w_direction=w_direction,
                            mu=mu,
                            logvar=logvar,
                            w_kld=w_kld,
                            use_sampling=self.network.use_sampling,
                        )
                        val_loss += loss.item()
                val_loss /= len(val_loader)
            else:
                val_loss = None

            # Store the loss in the history and print
            self.loss_history["train_loss"].append(train_loss)
            if val_loss is not None:
                self.loss_history["val_loss"].append(val_loss)
                print(
                    f"Epoch {epoch + 1}/{n_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{n_epochs}, Training Loss: {train_loss:.4f}")

            # Step the learning rate scheduler
            scheduler.step()

            # Early stopping logic
            if val_loss is not None:
                improvement = self.best_val_loss - val_loss
                if (improvement > 0.001):  # fmt: skip
                    self.best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state_dict = self.network.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break

        # Load the best model
        if best_model_state_dict is None:
            print("No improvement observed, keeping the original model")
        else:
            self.network.load_state_dict(best_model_state_dict)

    def pred(self, pert_names, ctrl_expr, covariate=None):
        """
        Transcriptional prediction given the list of perturbations, and corresponding control expression.

        Parameters:
        ----------
        - pert_names [str]:
            A list of perturbation gene names for prediction
        - ctrl_expr:
            A cell by gene matrix of control cell's expression for prediction.
        - covariate (str, optional):
            The description of the covariate of each control cell.

        If provided, it should satisfy that len(pert_names) = nrow(ctrl_expr) = len(covariate)
        Returns:
        ----------
        - pandas.DataFrame:
            A dictionary where keys are the perturbations and values are predicted transcriptional responses.
        """

        # Examine if there is any input gene not in the embedding matrix
        unique_inputs = np.unique(sum([p.split("+") for p in pert_names], []))
        unique_not_embd = [p not in self.embd_idx_dict for p in unique_inputs]
        not_found = unique_inputs[unique_not_embd]
        if len(not_found) > 0:
            raise ValueError(
                f"{len(not_found)} gene(s) are not found in the gene embedding matrix: {not_found}"
            )

        # Prepare the input for network
        pert_idx_list = [[self.embd_idx_dict[g] for g in p.split("+")] for p in pert_names]  # fmt: skip
        pert_idx = torch.tensor(pert_idx_list, dtype=torch.long).to(self.device)
        ctrl_expr = torch.tensor(ctrl_expr, dtype=torch.float32).to(self.device)

        # Inference
        self.network.eval()
        with torch.no_grad():
            pred_expr, mu, logvar = self.network(pert_idx, ctrl_expr)

        pert_return = pd.DataFrame(
            pred_expr.cpu().numpy(),
            index=["_".join([c, p]) for c, p in zip(covariate, pert_names)]
            if covariate
            else pert_names,
            columns=self.all_adata.var_names,
        )
        return pert_return

    def barplot(self, cov_pert, genes_shown="degs"):
        """
        For a provided `cov_pert` in `self.all_adata.obs[self.key_cov_pert]`, generate a bar plot showing the corresponding expression of selected genes on
        control cells, predicted cells, and true cells, respectively.

        Parameters:
        ----------
        - cov_pert:
            The covariate and perturbation that the barplot is based on. It has to be in `self.all_adata.obs[self.key_cov_pert]`
        - genes_shown (str or list, optional):
            - If 'degs' (default), the function selects and visualizes the top differentially expressed genes (DEGs) for the given `cov_pert`.
            - If a list of gene names is provided, the function plots the expression of only those specified genes.
        """
        if cov_pert not in self.all_adata.obs[self.key_cov_pert].unique():
            raise ValueError(f"{cov_pert} is not found in `self.all_adata.obs[self.key_cov_pert]`.")  # fmt: skip

        if genes_shown == "degs":
            genes_shown = self.all_adata.uns[self.key_uns_DegNames][cov_pert]

        covariate, pert_names = cov_pert.split("_")
        subset = self.all_adata[
            (self.all_adata.obs[self.key_cov] == covariate)
            & (self.all_adata.obs[self.key_pert].isin([pert_names, self.ctrl_value]))
        ].copy()

        plot_DS = BalancedDataset(
            adata=subset,
            ctrl_value=self.ctrl_value,
            key_pert=self.key_pert,
            key_embd_idx=self.key_embd_idx,
            key_stratify=self.key_stratify,
            shuffle=False,
        )
        ctrl_expr, pert_expr = plot_DS.ctrl_expr, plot_DS.pert_expr

        pred = self.pred([pert_names] * len(ctrl_expr), ctrl_expr, [covariate] * len(ctrl_expr))[genes_shown].assign(Group="Pred")  # fmt: skip
        ctrl = pd.DataFrame(ctrl_expr, columns=self.all_adata.var_names, index=pred.index)[genes_shown].assign(Group="Ctrl")  # fmt: skip
        true = pd.DataFrame(pert_expr, columns=self.all_adata.var_names, index=pred.index)[genes_shown].assign(Group="True")  # fmt: skip

        df_combined = pd.concat([ctrl, pred, true], axis=0)
        df_melted = df_combined.melt(id_vars="Group", var_name="Gene", value_name="Value")  # fmt: skip

        # Create the boxplot
        plt.figure(figsize=(15, 8))
        sns.boxplot(x="Gene", y="Value", hue="Group", data=df_melted, showfliers=False)
        plt.xticks(rotation=90)
        plt.title(f"Expression of top 20 genes for perturbation on {cov_pert}")
        plt.show()

    def evaluate(self):
        """
        Evaluates the model's performance on the test dataset using various metrics.

        Returns:
        - pandas.DataFrame:
            A DataFrame containing the evaluation metrics (Normalized MSE, Pearson correlation, Spearman correlation) for each test condition.

        Notes:
        ----------
        - NormMSE:
            Normalized Mean Squared Error (MSE) of the predictions compared to the true values, normalized by the MSE of the control values.
        - Pearson:
            Pearson correlation coefficient between (true-ctrl) and (pred-ctrl) values.
        - Spearman:
            Spearman rank correlation coefficient between (true-ctrl) and (pred-ctrl) values.
        """
        metric = {
            # ALL
            "MSEPred_all": {},
            "MSECtrl_all": {},
            "NormMSE_all": {},
            "Pearson_all": {},
            "NormPearson_all": {},
            "Spearman_all": {},
            "NormSpearman_all": {},
            "PredCosine_all": {},
            "CtrlCosine_all": {},
            "NormCosine_all": {},
            "SameDirect_all": {},
            "DiffDirect_all": {},
            # DEGS
            "MSEPred_degs": {},
            "MSECtrl_degs": {},
            "NormMSE_degs": {},
            "Pearson_degs": {},
            "NormPearson_degs": {},
            "Spearman_degs": {},
            "NormSpearman_degs": {},
            "PredCosine_degs": {},
            "CtrlCosine_degs": {},
            "NormCosine_degs": {},
            "SameDirect_degs": {},
            "DiffDirect_degs": {},
        }
        test_conds = list(self.test_adata.obs[self.key_cov_pert].unique())

        # fmt: off
        for wholeset in DataLoader(self.test_dataset, batch_size=len(self.test_dataset), drop_last=False):
            ctrl_expr, pert_expr, pert_idx, ctrl_bcode, pert_bcode = wholeset

        pert_names = self.test_adata[list(pert_bcode)].obs[self.key_pert].values.tolist()
        covariate = self.test_adata[list(pert_bcode)].obs[self.key_cov].values.tolist()
        cov_pert = self.test_adata[list(pert_bcode)].obs[self.key_cov_pert].values.tolist()

        pred_all = self.pred(pert_names, ctrl_expr.numpy(), covariate).assign(cov_pert=cov_pert)
        pred_all.index = pert_bcode
        ctrl_all = pd.DataFrame(ctrl_expr, columns=self.all_adata.var_names, index=ctrl_bcode).assign(cov_pert=cov_pert)
        true_all = pd.DataFrame(pert_expr, columns=self.all_adata.var_names, index=pert_bcode).assign(cov_pert=cov_pert)

        test_conds = self.test_adata.obs[self.key_cov_pert].unique()
        test_conds = [i for i in test_conds if self.ctrl_value not in i]

        for cond in test_conds:
            degs = self.all_adata.uns[self.key_uns_DegNames][cond]

            pred = pred_all[pred_all.cov_pert == cond].drop(columns="cov_pert").mean(axis=0)
            ctrl = ctrl_all[ctrl_all.cov_pert == cond].drop(columns="cov_pert").mean(axis=0)
            true = true_all[true_all.cov_pert == cond].drop(columns="cov_pert").mean(axis=0)

            diff_true_ctrl = true - ctrl
            diff_pred_ctrl = pred - ctrl

            metric["MSEPred_all"][cond] = mse(true, pred)
            metric["MSECtrl_all"][cond] = mse(true, ctrl)
            metric["NormMSE_all"][cond] = mse(true, pred)/mse(true, ctrl)
            metric["Pearson_all"][cond] = pearsonr(true, pred)[0]
            metric["NormPearson_all"][cond] = pearsonr(diff_true_ctrl, diff_pred_ctrl)[0]
            metric["Spearman_all"][cond] = spearmanr(true, pred)[0]
            metric["NormSpearman_all"][cond] = spearmanr(diff_true_ctrl, diff_pred_ctrl)[0]
            metric["PredCosine_all"][cond] = 1-cosine(true, pred)
            metric["CtrlCosine_all"][cond] = 1-cosine(true, ctrl)
            metric["NormCosine_all"][cond] = (1-cosine(true, pred)) / (1-cosine(true, ctrl))
            metric["SameDirect_all"][cond] = sum((diff_true_ctrl*diff_pred_ctrl) > 0)
            metric["DiffDirect_all"][cond] = sum((diff_true_ctrl)*(diff_pred_ctrl) < 0)

            metric["MSEPred_degs"][cond] = mse(true[degs], pred[degs]) if len(degs) > 0 else None
            metric["MSECtrl_degs"][cond] = mse(true[degs], ctrl[degs]) if len(degs) > 0 else None
            metric["NormMSE_degs"][cond] = mse(true[degs], pred[degs])/mse(true[degs], ctrl[degs]) if len(degs) > 0 else None
            metric["Pearson_degs"][cond] = pearsonr(true[degs], pred[degs])[0] if len(degs) > 1 else None
            metric["NormPearson_degs"][cond] = pearsonr(diff_true_ctrl[degs], diff_pred_ctrl[degs])[0] if len(degs) > 1 else None
            metric["Spearman_degs"][cond] = spearmanr(true[degs], pred[degs])[0] if len(degs) > 1 else None
            metric["NormSpearman_degs"][cond] = spearmanr(diff_true_ctrl[degs], diff_pred_ctrl[degs])[0] if len(degs) > 1 else None
            metric["PredCosine_degs"][cond] = 1-cosine(true[degs], pred[degs]) if len(degs) > 1 else None
            metric["CtrlCosine_degs"][cond] = 1-cosine(true[degs], ctrl[degs]) if len(degs) > 1 else None
            metric["NormCosine_degs"][cond] = (1-cosine(true[degs], pred[degs])) / (1-cosine(true[degs], ctrl[degs])) if len(degs) > 1 else None
            metric["SameDirect_degs"][cond] = sum((diff_true_ctrl*diff_pred_ctrl)[degs] > 0) if len(degs) > 0 else None
            metric["DiffDirect_degs"][cond] = sum((diff_true_ctrl)*(diff_pred_ctrl)[degs] < 0) if len(degs) > 0 else None

        metric_df = pd.DataFrame(metric)
        result_dict = {'Prediction': pred_all, 'Control': ctrl_all, 'True': true_all}
        return metric_df, result_dict
