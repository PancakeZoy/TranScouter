import torch
import torch.nn as nn


class TranScouterModel(nn.Module):
    def __init__(
        self,
        ctrl_dim: int,
        gene_embd: torch.Tensor,
        hidden_enc_embd: tuple,
        hidden_enc_ctrl: tuple,
        bottle_dim: int,
        hidden_dec: tuple,
        use_batch_norm: bool,
        use_layer_norm: bool,
        dropout_rate: float,
        use_sampling: bool = False,
        condition: str = "control",
    ):
        """
        Initialize the TranScouterModel.

        Parameters:
        ----------
        - ctrl_dim:
            Length of the control expression vector.
        - gene_embd:
            Tensor of gene embeddings.
        - hidden_enc_embd:
            Tuple specifying the hidden layer sizes for the encoder of gene embedding.
        - hidden_enc_ctrl:
            Tuple specifying the hidden layer sizes for the encoder of control cell's expression.
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
        """

        super(TranScouterModel, self).__init__()

        self.use_sampling = use_sampling
        self.condition = condition
        self.gene_embd = nn.Parameter(gene_embd.clone().detach(), requires_grad=False)
        n_embd = self.gene_embd.shape[1]

        # 1) Separate MLPs to encode g (gene embedding) and c (control expression)
        self.encoder_embd = self._build_mlp(
            input_dim=n_embd,
            hidden_dims=hidden_enc_embd,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
        )
        self.encoder_ctrl = self._build_mlp(
            input_dim=ctrl_dim,
            hidden_dims=hidden_enc_ctrl,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
        )

        # 2) Latent distribution parameters
        concat_size = hidden_enc_embd[-1] + hidden_enc_ctrl[-1]
        self.fc_mu = nn.Linear(concat_size, bottle_dim)
        self.fc_logvar = nn.Linear(concat_size, bottle_dim)

        # 3) Decoder
        if condition == "control":
            dec_input_dim = bottle_dim + hidden_enc_ctrl[-1]
        elif condition == "embedding":
            dec_input_dim = bottle_dim + hidden_enc_embd[-1]
        self.decoder = self._build_mlp(
            input_dim=dec_input_dim,
            hidden_dims=hidden_dec,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            output_dim=ctrl_dim,
        )

    def encode(self, embd, ctrl):
        """
        Encodes (embd, ctrl) into latent distribution parameters (mu, logvar).

        Returns:
            - mu: The mean of the latent distribution.
            - logvar: The log-variance of the latent distribution.
            - hc: The last layer of the condition encoder, which is reused in the decoder.

        Note:
            - If `use_sampling=True`, the model will sample from the latent distribution using (mu, logvar).
            - If `use_sampling=False`, only mu is used as the latent representation, bypassing stochastic sampling.
        """
        h_embd = self.encoder_embd(embd)
        h_ctrl = self.encoder_ctrl(ctrl)

        # Concatenate for latent distribution
        if self.condition == "control":
            h_concat = torch.cat([h_embd, h_ctrl], dim=-1)
            h_condition = h_ctrl
        elif self.condition == "embedding":
            h_concat = torch.cat([h_ctrl, h_embd], dim=-1)
            h_condition = h_embd
        mu = self.fc_mu(h_concat)
        logvar = self.fc_logvar(h_concat)

        return mu, logvar, h_condition

    def reparameterize(self, mu, logvar):
        """
        Samples z via the reparameterization trick:
        z = mu + std * epsilon,  epsilon ~ N(0,1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, h_condition):
        """
        Decodes the latent variable z and the encoded condition hc into a gene expression vector.
        """

        z_condition = torch.cat([z, h_condition], dim=-1)
        out = self.decoder(z_condition)
        return out

    def forward(self, pert_idx, ctrl_exp):
        """
        Forward pass of the TranScouterModel.

        Parameters:
        - pert_idx (torch.Tensor):
            Tensor containing the indices of perturbed genes. Shape: (batch_size, num_genes).
        - ctrl_exp (torch.Tensor):
            Tensor containing the control expression values. Shape: (batch_size, num_genes).
        """
        gene_embd = self.gene_embd[pert_idx].sum(axis=1)
        mu, logvar, h_condition = self.encode(gene_embd, ctrl_exp)
        if self.use_sampling:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        recon = self.decode(z, h_condition)
        return recon, mu, logvar

    def _build_mlp(
        self,
        input_dim,
        hidden_dims,
        use_batch_norm,
        use_layer_norm,
        dropout_rate,
        output_dim=None,
    ):
        """
        Builds a multi-layer perceptron (MLP) with the specified configurations.

        Parameters:
        - input_dim (int):
            The dimension of the input layer.
        - hidden_dims (tuple of int):
            A tuple specifying the size of each hidden layer.
        - output_dim (int):
            The dimension of the output layer.
        - use_batch_norm (bool):
            Whether to use batch normalization after each hidden layer.
        - use_layer_norm (bool):
            Whether to use layer normalization after each hidden layer.
        - dropout_rate (float):
            The dropout rate to use in Alpha Dropout layers. Set to 0.0 to disable dropout.

        Returns:
        - nn.Sequential:
            A sequential container with the constructed MLP layers.

        Example:
        mlp = model._build_mlp(input_dim=128, hidden_dims=(64, 32), output_dim=10,
                               use_batch_norm=True, use_layer_norm=False, dropout_rate=0.1)
        """
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.SELU())
            if dropout_rate > 0.0:
                layers.append(nn.AlphaDropout(dropout_rate))
            input_dim = h_dim
        if output_dim is not None:
            layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)
