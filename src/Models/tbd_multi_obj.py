import numpy as np
import torch
from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel

import Models
from torchdiffeq import odeint
from torchsde import sdeint
import torch.nn.functional as F

class TBD(GenerativeModel):
    
    """
     Class for Trajectory-Based Diffusion
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params, device, doc):
        super().__init__(params, device, doc)
        trajectory = get(self.params, "trajectory", "linear_trajectory")
        try:
            self.trajectory =  getattr(Models.tbd, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")

        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"C is {self.C}")

        self.bayesian = get(self.params, "bayesian", 0)
        self.t_min = get(self.params, "t_min", 0)
        self.t_max = get(self.params, "t_max", 1)
        self.distribution = torch.distributions.uniform.Uniform(low=self.t_min, high=self.t_max)
        self.add_noise = get(self.params, "add_noise", False)
        self.gamma = get(self.params, "gamma", 1.e-4)


    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "Resnet")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        condition = input[1]
        weights = None
        return input[0], condition, weights

    # def batch_loss(self, x):
    #     """
    #     Calculate batch loss as described by Peter
    #     """
    #     # get input and conditions
    #     x, condition, weights = self.get_condition_and_input(x)
        
    #     if self.latent:
    #         # encode x into autoencoder latent space
    #         x = self.ae.encode(x, condition)
    #         if self.ae.kl:
    #             x = self.ae.reparameterize(x[0], x[1])

    #     # t = self.distribution.sample((x.size(0),1)).to(x.device)
    #     t = self.distribution.sample([x.shape[0]] + [1]*(x.dim() - 1)).to(x.device)
    #     x_0 = torch.randn_like(x)
    #     if self.add_noise:
    #         x = x + self.gamma * torch.randn_like(x, device=x.device, dtype=x.dtype)
    #     x_t, x_t_dot = self.trajectory(x_0, x, t)
    #     self.net.kl = 0
    #     drift = self.net(x_t, t.view(-1, 1), condition)

    #     loss = torch.mean((drift - x_t_dot) ** 2)#* torch.exp(self.t_factor * t)) ?
    #     # self.regular_loss.append(loss.detach().cpu().numpy())
    #     # if self.C != 0:
    #         # kl_loss = self.C*self.net.kl / self.n_traindata
    #         # self.kl_loss.append(kl_loss.detach().cpu().numpy())
    #         # loss = loss + kl_loss

    #     return loss
    def compute_sparsity_occupancy(
        self,
        real: torch.Tensor,
        gen: torch.Tensor,
    ):
        
        """
        Compute sparsity and soft-occupancy diagnostics in preprocessed space.
    
        Args:
            real: real voxel data x, shape (B, 1, 45, H, W)
            gen: generated voxel data x0_pred, same shape as real
    
        Returns:
            dict with scalar tensors:
                - sparsity_l1_real
                - sparsity_l1_gen
                - sparsity_match_loss
                - occ_real
                - occ_gen
                - occ_match_loss
        """
    
        device = real.device
        dtype  = real.dtype
    
        # -------- initialize outputs (safe defaults) --------
        out = {
            "sparsity_l1_real": torch.zeros((), device=device, dtype=dtype),
            "sparsity_l1_gen":  torch.zeros((), device=device, dtype=dtype),
            "sparsity_match_loss": torch.zeros((), device=device, dtype=dtype),
            "occ_real": torch.zeros((), device=device, dtype=dtype),
            "occ_gen":  torch.zeros((), device=device, dtype=dtype),
            "occ_match_loss": torch.zeros((), device=device, dtype=dtype),
        }
    
        # -------- L1 sparsity --------
        sparsity_l1_real = real.abs().mean()
        sparsity_l1_gen  = gen.abs().mean()
    
        out["sparsity_l1_real"] = sparsity_l1_real
        out["sparsity_l1_gen"]  = sparsity_l1_gen
        out["sparsity_match_loss"] = (sparsity_l1_gen - sparsity_l1_real).pow(2)
    
        # -------- soft occupancy --------
        tau  = float(getattr(self, "occ_tau", 0.0))
        temp = max(float(getattr(self, "occ_temp", 1.0)), 1e-6)
    
        act_real = torch.sigmoid((real.abs() - tau) / temp)
        act_gen  = torch.sigmoid((gen.abs()  - tau) / temp)
    
        occ_real = act_real.mean()
        occ_gen  = act_gen.mean()
    
        out["occ_real"] = occ_real
        out["occ_gen"]  = occ_gen
        out["occ_match_loss"] = (occ_gen - occ_real).pow(2)
    
        return out
    
    def compute_auxiliary_losses_log_space(self, x0_pred, x_real):
        """
        Compute ALL auxiliary losses in preprocessed (log) space,
        consistent with the diffusion loss scale.
        
        Args:
            x0_pred: (B, 1, 45, 16, 9) - predicted clean sample
            x_real:  (B, 1, 45, 16, 9) - real clean sample
            condition: (B, 46) - [E_inc, u_1, ..., u_45] or None
        
        Returns:
            aux_loss_tensors: dict of loss tensors (for gradient computation)
            aux_loss_scalars: dict of scalar values (for logging)
        """
        device = x_real.device
        dtype  = x_real.dtype

        aux_loss_tensors = {}
        aux_loss_scalars = {}

        # Squeeze channel dim: (B, 1, 45, 16, 9) → (B, 45, 16, 9)
        voxels_pred = x0_pred.squeeze(1)
        with torch.no_grad():
            voxels_true = x_real.squeeze(1)

        # ========================================
        # 1. MOMENT MATCHING LOSS
        # ========================================
        mu_real  = voxels_true.mean(dim=(-1, -2))
        mu_gen   = voxels_pred.mean(dim=(-1, -2))
        var_real = voxels_true.var(dim=(-1, -2), unbiased=False)
        var_gen  = voxels_pred.var(dim=(-1, -2), unbiased=False)

        moment_loss = (torch.mean((mu_gen  - mu_real) ** 2) +
                       torch.mean((var_gen - var_real) ** 2))

        aux_loss_tensors['moment_loss'] = moment_loss
        aux_loss_scalars['moment_loss'] = moment_loss.item()

        # ========================================
        # 2. SPARSITY & OCCUPANCY LOSS
        # ========================================
        sparsity_stats = self.compute_sparsity_occupancy(real=x_real, gen=x0_pred)
        sparsity_loss  = sparsity_stats['sparsity_match_loss']

        # ========================================
        # 3. VOXEL ENERGY LOSS (Poisson-Weighted Huber in preprocessed space)
        # ========================================
        epsilon = self.params.get('voxel_loss_epsilon', 1e-6)
        delta   = self.params.get('voxel_loss_delta',   1.0)
        
        residuals        = voxels_pred - voxels_true
        poisson_std      = torch.sqrt(torch.abs(voxels_true) + epsilon)
        normalized_resid = residuals / poisson_std
        
        voxel_energy_loss = F.huber_loss(
            normalized_resid,
            torch.zeros_like(normalized_resid),
            delta=delta,
            reduction='mean'
        )
        
        aux_loss_tensors['voxel_energy_loss'] = voxel_energy_loss
        aux_loss_scalars['voxel_energy_loss'] = voxel_energy_loss.item()

        # ========================================
        # Sparsity logging
        # ========================================
        aux_loss_tensors['sparsity_loss']       = sparsity_loss
        aux_loss_scalars['sparsity_l1_real']    = sparsity_stats['sparsity_l1_real'].item()
        aux_loss_scalars['sparsity_l1_gen']     = sparsity_stats['sparsity_l1_gen'].item()
        aux_loss_scalars['sparsity_match_loss'] = sparsity_stats['sparsity_match_loss'].item()
        aux_loss_scalars['occ_real']            = sparsity_stats['occ_real'].item()
        aux_loss_scalars['occ_gen']             = sparsity_stats['occ_gen'].item()
        aux_loss_scalars['occ_match_loss']      = sparsity_stats['occ_match_loss'].item()

        return aux_loss_tensors, aux_loss_scalars
        
    def batch_loss(self, x):
        x, condition, weights = self.get_condition_and_input(x)
    
        t = self.distribution.sample([x.shape[0]] + [1]*(x.dim() - 1)).to(x.device)
        x_0 = torch.randn_like(x)
        x_t, x_t_dot = self.trajectory(x_0, x, t)
    
        drift = self.net(x_t, t.view(-1, 1), condition)
    
        # ── Standard FM loss ──────────────────────────────────────────────
        fm_loss = torch.mean((drift - x_t_dot) ** 2)
    
        # ── x₁ estimate in log/preprocessed space ────────────────────────
        x_1_hat = x_t + (1 - t) * drift
    
        # ── Auxiliary losses ──────────────────────────────────────────────
        aux_tensors, aux_scalars = self.compute_auxiliary_losses_log_space(
            x0_pred=x_1_hat,
            x_real=x,
            
        )
        #aux_tensors['fm_loss'] = fm_loss
        aux_scalars['fm_loss'] = fm_loss.item()
    
        # ── t-gate scalar ─────────────────────────────────────────────────
        t_gate = (t.view(-1) > self.params.get('physics_loss_t_threshold', 0.3)).float().mean()
        t_gate_scalar = t_gate.item()
        # ── Weighted sum ──────────────────────────────────────────────────
        lambda_voxel    = self.params.get('lambda_voxel_energy_loss', 0.02)
        lambda_moment   = self.params.get('lambda_moment_loss',       0.0)
        lambda_sparsity = self.params.get('lambda_sparsity_loss',     0.0)
        # ── Log the weights alongside the losses ─────────────────────────
        aux_scalars['lambda_voxel']    = t_gate_scalar * lambda_voxel
        aux_scalars['lambda_moment']   = t_gate_scalar * lambda_moment
        aux_scalars['lambda_sparsity'] = t_gate_scalar * lambda_sparsity
        total_loss = (fm_loss
                      + t_gate * lambda_voxel    * aux_tensors['voxel_energy_loss']
                      + t_gate * lambda_moment   * aux_tensors['moment_loss']
                      + t_gate * lambda_sparsity * aux_tensors['sparsity_loss'])
    
        
        
    
        return total_loss, aux_scalars, aux_tensors
        
    @torch.inference_mode()
    def sample_batch(self, batch):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        dtype = batch.dtype
        device = batch.device

        x_T = torch.randn((batch.shape[0], *self.shape), dtype=dtype, device=device)

        def f(t, x_t):
            t_torch = t.repeat((x_t.shape[0],1)).to(self.device)
            return self.net(x_t, t_torch, batch)

        solver = sdeint if self.params.get("use_sde", False) else odeint
        function = SDE(self.net) if self.params.get("use_sde", False) else f

        sample = solver(
            function, x_T,
            torch.tensor([self.t_min, self.t_max], dtype=dtype, device=device),
            **self.params.get("solver_kwargs", {})
        )[-1]

        if self.latent:
            # decode the generated sample
            sample = self.ae.decode(sample, batch)
            
        return sample

    def invert_n(self, samples):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        if self.net.bayesian:
            self.net.map = get(self.params,"fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval()
        batch_size = get(self.params, "batch_size", 8192)
        n_samples = samples.shape[0]

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((-1, *self.shape)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.inference_mode():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.inference_mode():
            for i in range(int(n_samples / batch_size)):
                sol = solve_ivp(f, (1, 0), samples[batch_size * i: batch_size * (i + 1)].flatten())
                s = sol.y[:, -1].reshape(batch_size, *self.shape)
                events.append(s)
            sol = solve_ivp(f, (1, 0), samples[batch_size * (i+1):].flatten())
            s = sol.y[:, -1].reshape(-1, *self.shape)
            events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]

    def sample_n_evolution(self, n_samples):

        n_frames = get(self.params, "n_frames", 1000)
        t_frames = np.linspace(0, 1, n_frames)

        batch_size = get(self.params, "batch_size", 8192)
        x_T = np.random.randn(n_samples + batch_size, *self.shape)

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, *self.shape)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.inference_mode():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.inference_mode():
            for i in range(int(n_samples / batch_size) + 1):
                sol = solve_ivp(f, (0, 1), x_T[batch_size * i: batch_size * (i + 1)].flatten(), t_eval=t_frames)
                s = sol.y.reshape(batch_size, *self.shape, -1)
                events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]



def sine_cosine_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c * x_0 + s * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = c_dot * x_0 + s_dot * x_1
    return x_t, x_t_dot

def sine2_cosine2_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c**2 * x_0 + s**2 * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = 2 * c_dot * c * x_0 + 2 * s_dot * s * x_1
    return x_t, x_t_dot

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot

def vp_trajectory(x_0, x_1, t, a=19.9, b=0.1):

    e = -1./4. * a * (1-t)**2 - 1./2. * b * (1-t)
    alpha_t = torch.exp(e)
    beta_t = torch.sqrt(1-alpha_t**2)
    x_t = x_0 * alpha_t + x_1 * beta_t

    e_dot = 2 * a * (1-t) + 1./2. * b
    alpha_t_dot = e_dot * alpha_t
    beta_t_dot = -2 * alpha_t * alpha_t_dot / beta_t
    x_t_dot = x_0 * alpha_t_dot + x_1 * beta_t_dot
    return x_t, x_t_dot

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, net):
        super().__init__()
        self.net = net

    def f(self,t, x_t):
        t_torch = t * torch.ones_like(x_t[:, [0]])
        v = self.net(x_t, t_torch)

        return v
    def g(self,t,x_t):
        epsilon = 0.5 * torch.ones_like(x_t)
        return np.sqrt(2*epsilon)*x_t.shape[1]
