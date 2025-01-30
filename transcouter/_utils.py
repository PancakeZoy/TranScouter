import torch
import numpy as np
import os
import gzip
import shutil
import requests
from tqdm import tqdm
import anndata as ad
import pandas as pd
from scipy.sparse import issparse
import scanpy as sc


def gears_loss(
    pred_expr,
    true_expr,
    ctrl_expr,
    group,
    nonzero_idx_dict,
    w_norm,
    w_direction,
):
    # Autofocus loss calculation
    autofocus_loss = (true_expr - pred_expr).abs() ** (2 + w_norm)
    # Direction-aware loss calculation
    direction_aware_loss = (
        torch.sign(true_expr - ctrl_expr) - torch.sign(pred_expr - ctrl_expr)
    ) ** 2
    # Total loss
    total_loss = autofocus_loss + w_direction * direction_aware_loss
    unique_cond, cells_cond_index = np.unique(group, return_inverse=True)

    idx_dict = {val: [] for val in unique_cond}
    for idx_cell, idx_cond in enumerate(cells_cond_index):
        cond = unique_cond[idx_cond]
        idx_dict[cond].append(idx_cell)

    loss_scalar = torch.tensor(0.0, device=pred_expr.device)
    for p in unique_cond:
        cell_idx = idx_dict[p]
        if nonzero_idx_dict is not None:
            retain_gene_idx = list(nonzero_idx_dict[p])
            loss_scalar += total_loss[cell_idx][:, retain_gene_idx].mean()
        else:
            loss_scalar += total_loss[cell_idx].mean()

    return loss_scalar / (len(unique_cond))


def kld_loss(mu, logvar):
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def total_loss(
    pred_expr,
    true_expr,
    ctrl_expr,
    group,
    nonzero_idx_dict,
    w_norm,
    w_direction,
    mu,
    logvar,
    w_kld,
    use_sampling,
):
    recon_loss = gears_loss(
        pred_expr, true_expr, ctrl_expr, group, nonzero_idx_dict, w_norm, w_direction
    )
    kld = kld_loss(mu, logvar)
    total_loss = recon_loss + (w_kld * kld if use_sampling else 0)
    return total_loss


def download_and_extract(url, save_dir, filename):
    """
    Download and unzip a file if it's compressed (.gz).

    Parameters
    ----------
    url : str
        URL of the file to download.
    save_dir : str
        Directory where the file will be saved.
    filename : str
        Name of the file to be saved in `save_dir`.

    Returns
    -------
    str
        Path to the downloaded (and unzipped if applicable) file.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    uncompressed_path = file_path.rstrip(".gz")

    # Check if the uncompressed file already exists
    if not os.path.exists(uncompressed_path):
        # Download the compressed file if it hasn't been downloaded
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024
            progress_bar = tqdm(
                total=total_size_in_bytes, unit="B", unit_scale=True, unit_divisor=1024
            )
            with open(file_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print(
                    "ERROR: Something went wrong during the download. The downloaded file might be corrupted."
                )

        # Unzip the .gz file and delete it
        if file_path.endswith(".gz"):
            print(f"Unzipping {filename}...")
            with gzip.open(file_path, "rb") as f_in:
                with open(uncompressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)
            print("Unzipped and removed the compressed file.")
    else:
        print(f"{filename} already available and uncompressed.")

    return uncompressed_path


def adamson_small(save_dir="./data"):
    """
    Download, unzip, and load a small version of Adamson dataset as a demo dataset.

    Returns
    -------
    anndata.AnnData
        The example dataset loaded into an AnnData object.
    """
    url = "https://github.com/PancakeZoy/scouter_misc/raw/main/data/Data_Demo/Adamson_small.h5ad.gz"
    dataset_filename = "Adamson_small.h5ad.gz"
    file_path = download_and_extract(url, save_dir, dataset_filename)
    adata = ad.read_h5ad(file_path)
    return adata


def embedding_small(save_dir="./data"):
    """
    Download, unzip, and load a small version of Embedding dataset as a demo dataset.

    Returns
    -------
    pd.DataFrame
        The example dataset loaded into a pandas DataFrame.
    """
    url = "https://github.com/PancakeZoy/scouter_misc/raw/main/data/Data_Demo/Embedding_small.csv.gz"
    dataset_filename = "Embedding_small.csv.gz"
    file_path = download_and_extract(url, save_dir, dataset_filename)
    df = pd.read_csv(file_path, index_col=0)
    return df


def to_np_array(X):
    if issparse(X):
        return X.toarray()
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise TypeError("Input must be a scipy.sparse matrix or a NumPy array.")


def interleave_rows(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Interleave rows from df1 and df2, one by one.
    If one DataFrame is shorter, then after interleaving as far as possible,
    append the remaining rows of the longer DataFrame.
    """
    n1, n2 = len(df1), len(df2)
    nmin = min(n1, n2)

    if n1 == 0 and n2 == 0:
        return df1.copy()

    interleaved_rows = []
    for i in range(nmin):
        interleaved_rows.append(df1.iloc[[i]])
        interleaved_rows.append(df2.iloc[[i]])
    if n1 > nmin:
        interleaved_rows.append(df1.iloc[nmin:])
    if n2 > nmin:
        interleaved_rows.append(df2.iloc[nmin:])

    return pd.concat(interleaved_rows, ignore_index=True)


def get_degs(subset, pval_cutoff=0.05, log2fc_min=1.5, log2fc_max=-1.5, genes_keep=25):
    """
    Get differentially expressed genes (DEGs) for each group in the dataset.

    This function identifies both up-regulated and down-regulated genes based on
    p-value and log2 fold change thresholds, and interleaves them for each perturbation.

    Parameters
    ----------
    subset : AnnData
        Annotated data matrix containing gene expression data and metadata.
    pval_cutoff : float, optional (default=0.05)
        P-value threshold for statistical significance.
    log2fc_min : float, optional (default=1.5)
        Minimum log2 fold change threshold for up-regulated genes.
    log2fc_max : float, optional (default=-1.5)
        Maximum log2 fold change threshold for down-regulated genes.

    Returns
    -------
    dict
        Dictionary where keys are perturbation names and values are lists of
        differentially expressed gene names, alternating between up- and down-regulated genes.
    """
    degs = {}
    degs_pos = sc.get.rank_genes_groups_df(subset, group=None, pval_cutoff=pval_cutoff, log2fc_min=log2fc_min)  # fmt: skip
    degs_neg = sc.get.rank_genes_groups_df(subset, group=None, pval_cutoff=pval_cutoff, log2fc_max=log2fc_max)  # fmt: skip

    for pert in degs_pos["group"].cat.categories:
        pert_pos = degs_pos[degs_pos["group"] == pert]
        pert_neg = degs_neg[degs_neg["group"] == pert]
        degs_df = interleave_rows(pert_pos, pert_neg)
        degs[pert] = degs_df.names.tolist()[:genes_keep]

    return degs


def test_Dataset(dataset, ad):
    i = np.random.randint(0, len(dataset))
    ctrl_expr, pert_expr, pert_idx, ctrl_bcode, pert_bcode = dataset[i]

    ctrl_test = abs(ad[ctrl_bcode].X.sum().item() - ctrl_expr.sum().item()) < 0.001
    pert_test = abs(ad[pert_bcode].X.sum().item() - pert_expr.sum().item()) < 0.001
    embd_test = ad[pert_bcode].obs.embd_index.values.item() == pert_idx.tolist()

    return all([ctrl_test, pert_test, embd_test])
