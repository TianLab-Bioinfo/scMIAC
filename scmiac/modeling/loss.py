import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, x_rec, x, mu, logvar):
        """
        Compute VAE loss
        Parameters:
        - x_rec: Reconstructed input data
        - x: Original input data
        - mu: Latent space mean
        - logvar: Log variance of the latent space
        Returns:
        - loss: Total loss
        - recon_loss: Reconstruction loss
        - kl_loss: KL divergence loss
        """
        # MSE loss
        recon_loss = F.mse_loss(x_rec, x, reduction='sum') / x.size(0)
        # KL loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss, kl_loss


# vae_loss_fn = VAELoss()
# recon_loss, kl_loss = vae_loss_fn(x_rec, x, mu, logvar)




class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, verbose=False):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.verbose = verbose

    def forward(self, z_i, z_j):
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        batch_size = z_i.size(0)
        
        if self.verbose:
            print(f"z_i.shape: {z_i.shape}")
            print(f"z_j.shape: {z_j.shape}")

        similarity_matrix = torch.mm(z_i, z_j.t()) / self.temperature
        if self.verbose:
            print(f"similarity_matrix: {similarity_matrix}")
            print(f"similarity_matrix.shape: {similarity_matrix.shape}")

        positives = torch.diag(similarity_matrix)

        # Log-Sum-Exp
        max_sim, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        exp_sim_matrix = torch.exp(similarity_matrix - max_sim)
        exp_sum = exp_sim_matrix.sum(dim=1)
        if self.verbose:
            print(f"exp_sim_matrix: {exp_sim_matrix}")
            print(f"exp_sim_matrix.shape: {exp_sim_matrix.shape}")
            print(f"exp_sum: {exp_sum}")

        log_sum_exp = max_sim.squeeze() + torch.log(exp_sum)
        if self.verbose:
            print(f"log_sum_exp: {log_sum_exp}")
            print(f"log_sum_exp.shape: {log_sum_exp.shape}")

        loss = -positives + log_sum_exp
        return loss.mean()
    

# infonce_loss_fn = NTXentLoss(temperature=0.5, verbose=True)
# loss = infonce_loss_fn(z_i, z_j)
# z_i, z_j are paired samples






class NTXentLoss2(nn.Module):
    def __init__(self, temperature=0.5):
        """
        Initialize the NTXentLoss2 (Normalized Temperature-scaled Cross Entropy Loss).

        Parameters:
        - temperature (float): Temperature scaling factor for the softmax function, controlling contrastive sharpness.
        """
        super(NTXentLoss2, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute the NT-Xent loss between two sets of embeddings.

        Parameters:
        - z_i (Tensor): Normalized embedding vectors from one modality or view.
        - z_j (Tensor): Normalized embedding vectors from another modality or view.

        Returns:
        - loss (Tensor): The NT-Xent loss, averaged over the batch.
        """
        
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # Step 1: i -> j 
        similarity_matrix_ij = torch.mm(z_i, z_j.t()) / self.temperature
        positives_ij = torch.diag(similarity_matrix_ij)
        max_sim_ij, _ = torch.max(similarity_matrix_ij, dim=1, keepdim=True)
        exp_sim_matrix_ij = torch.exp(similarity_matrix_ij - max_sim_ij)
        exp_sum_ij = exp_sim_matrix_ij.sum(dim=1)
        log_sum_exp_ij = max_sim_ij.squeeze() + torch.log(exp_sum_ij)
        loss_ij = -positives_ij + log_sum_exp_ij
        loss_ij = loss_ij.mean()

        # Step 2: j -> i 
        similarity_matrix_ji = torch.mm(z_j, z_i.t()) / self.temperature
        positives_ji = torch.diag(similarity_matrix_ji)
        max_sim_ji, _ = torch.max(similarity_matrix_ji, dim=1, keepdim=True)
        exp_sim_matrix_ji = torch.exp(similarity_matrix_ji - max_sim_ji)
        exp_sum_ji = exp_sim_matrix_ji.sum(dim=1)
        log_sum_exp_ji = max_sim_ji.squeeze() + torch.log(exp_sum_ji)
        loss_ji = -positives_ji + log_sum_exp_ji
        loss_ji = loss_ji.mean()

        # Step 3: mean
        loss = 0.5 * (loss_ij + loss_ji)
        return loss


class AnchorMSELoss(nn.Module):
    def __init__(self):
        """
        Initialize the AnchorMSELoss for anchor cell pairs alignment.
        This loss directly minimizes the mean squared error between anchor pairs,
        serving as an alternative to contrastive learning in ablation studies.
        """
        super(AnchorMSELoss, self).__init__()

    def forward(self, z_i, z_j):
        """
        Compute MSE loss between anchor pairs from two modalities.

        Parameters:
        - z_i (Tensor): Latent representations from modality 1 (e.g., RNA).
        - z_j (Tensor): Latent representations from modality 2 (e.g., ATAC).

        Returns:
        - loss (Tensor): Mean squared error between paired representations.
        """
        return F.mse_loss(z_i, z_j, reduction='mean')









class MMDLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(MMDLoss, self).__init__()
        self.gamma = gamma

    def rbf_kernel(self, x, y):
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        yy = torch.sum(y ** 2, dim=1, keepdim=True)
        dist = xx + yy.t() - 2.0 * torch.mm(x, y.t())
        return torch.exp(-self.gamma * dist)

    def forward(self, x, y):
        Kxx = self.rbf_kernel(x, x) 
        Kyy = self.rbf_kernel(y, y) 
        Kxy = self.rbf_kernel(x, y) 

        mmd = torch.mean(Kxx) + torch.mean(Kyy) - 2 * torch.mean(Kxy)
        return mmd





def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance



def unbalanced_ot(tran, mu1, mu2, device, Couple, reg=0.1, reg_m=1.0):
    '''
    Calculate a unbalanced optimal transport matrix between batches.

    Parameters
    ----------
    tran
        transport matrix between the two batches sampling from the global OT matrix. 
    mu1
        mean vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    reg
        Entropy regularization parameter in OT. Default: 0.1
    reg_m
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    '''

    ns = mu1.size(0)
    nt = mu2.size(0)

    cost_pp = distance_matrix(mu1, mu2)
    if Couple is not None:
        Couple = torch.tensor(Couple, dtype=torch.float).to(device)

    p_s = torch.ones(ns, 1) / ns
    p_t = torch.ones(nt, 1) / nt
    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if tran is None:
        tran = torch.ones(ns, nt) / (ns * nt)
        tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f = reg_m / (reg_m + reg)

    for m in range(10):
        if Couple is not None:
            #print(cost_pp)
            #print(Couple)
            cost = cost_pp*Couple
        else:
            cost = cost_pp

        kernel = torch.exp(-cost / (reg*torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for i in range(10):
            dual =( p_s / (kernel @ b) )**f
            b = ( p_t / (torch.t(kernel) @ dual) )**f
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    d_fgw = (cost * tran.detach().data).sum()

    return d_fgw, tran.detach()