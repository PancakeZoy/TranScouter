# A Structured Study of Cross-Condition Prediction of Transcriptional Responses to Gene Perturbations

TranScouter is a lightweight encoder-decoder model for predicting transcriptional responses to genetic perturbations across biological conditions. It represents perturbed genes with LLM-derived gene embeddings and represents the target biological condition with transcriptomic profiles of control cells from that condition.

The model is designed for cross-condition perturbation-response prediction, including both settings where the target perturbation has been observed in other training conditions and settings where the target perturbation is absent from all training perturbation-condition pairs.

Preprint: [A structured study of cross-condition prediction of transcriptional responses to gene perturbations](https://www.biorxiv.org/content/10.64898/2026.07.30.741892v1.abstract)

Code for reproducing manuscript results and figures: [TranScouter_misc](https://github.com/PancakeZoy/TranScouter_misc)

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/PancakeZoy/TranScouter.git
```

For local development:

```bash
git clone https://github.com/PancakeZoy/TranScouter.git
cd TranScouter
pip install -e .
```

## Main API

The example below shows the main TranScouter workflow. The required inputs are an AnnData object containing expression profiles and metadata, plus a gene-embedding table indexed by perturbation gene names.

```python
import pickle
import pandas as pd
import scanpy as sc

from transcouter import ScouterData, TranScouter

# Load expression data and gene embeddings.
adata = sc.read_h5ad("data/jiang/processed/aggregate_sum_hvg.h5ad")
with open("data/embeddings/scELMo.pickle", "rb") as f:
    embd = pd.DataFrame(pickle.load(f)).T
    embd.rename(index={"H1-0": "H1F0"}, inplace=True)

# Prepare the perturbation dataset.
scouterdata = ScouterData(
    adata=adata,
    embd=embd,
    key_pert="perturbation",
    key_cov="covariate",
    ctrl_value="control",
    key_var_gnames="gene_name",
)

scouterdata.setup_ad("embd_index")
scouterdata.gene_ranks(pval_cutoff=0.1)
scouterdata.get_nonzero_genes()

# Provide train, validation, and test perturbation-condition labels.
scouterdata.split_Train_Val_Test(
    train_conds=train_conds,
    val_conds=val_conds,
    test_conds=test_conds,
)

# Initialize and train TranScouter.
model = TranScouter(scouterdata)
model.data_init(key_stratify=["covariate", "bulk"])
model.model_init(
    hidden_enc_embd=(512, 256),
    hidden_enc_ctrl=(2048, 1024),
    bottle_dim=512,
    hidden_dec=(1024, 2048),
    use_batch_norm=True,
    use_layer_norm=False,
    dropout_rate=0.0,
    use_sampling=False,
    condition="control",
)

model.train(
    batch_size=512,
    w_norm=0.0,
    w_direction=0.25,
    w_kld=0.0,
    lr=1e-4,
    if_nonzero=True,
    n_epochs=50,
)

# Evaluate on the held-out perturbation-condition pairs.
metrics, predictions = model.evaluate()
```

## Input Format

TranScouter expects an AnnData object with:

- `adata.X`: expression matrix.
- `adata.obs[key_pert]`: perturbation labels, including a control label.
- `adata.obs[key_cov]`: biological condition labels.
- `adata.var[key_var_gnames]`: gene names matching the expression matrix columns.

The embedding input should be a pandas DataFrame whose index contains perturbation gene names and whose columns contain embedding dimensions.

## Model Overview

TranScouter combines two sources of information:

- A perturbation encoder that maps LLM-derived gene embeddings into a latent perturbation representation.
- A condition encoder that maps control-cell transcriptomic profiles from the target condition into a condition representation.

The perturbation and condition representations are combined through a bottleneck layer, and the condition representation is reintroduced before decoding to preserve target-condition information. The model is trained with reconstruction loss and an optional direction-aware loss that encourages the predicted expression change to match the observed perturbation direction.

## Reproducibility

This repository contains the TranScouter package implementation. Scripts, data-processing workflow, precomputed result organization, and figure notebooks for the manuscript are maintained separately in [TranScouter_misc](https://github.com/PancakeZoy/TranScouter_misc).

## Citation

If you use TranScouter, please cite:

```bibtex
@article{zhu2026transcouter,
  title = {A structured study of cross-condition prediction of transcriptional responses to gene perturbations},
  author = {Zhu, Ouyang and Li, Jun},
  year = {2026},
  doi = {10.64898/2026.07.30.741892},
  journal = {bioRxiv}
}
```
