import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConditionalVariationalModule(nn.Module):
    """
    Conditional Variational Module for modeling temporally-dependent latent variables.
    This module learns the prior and posterior distributions for the latent space.
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 latent_dim: int = 128,
                 hidden_dim: int = 256):
        super(ConditionalVariationalModule, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Prior network (p(z_t | z_{<t}, h_t))
        self.prior_net = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and log_var
        )
        
        # Posterior network (q(z_t | x_t, z_{<t}, h_t))
        self.posterior_net = nn.Sequential(
            nn.Linear(input_dim + latent_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and log_var
        )
        
        # Initial latent state
        self.init_latent = nn.Parameter(torch.randn(latent_dim))
        
    def forward(self, 
                encoder_features: torch.Tensor,
                prev_latent: torch.Tensor = None,
                is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the conditional variational module.
        
        Args:
            encoder_features: Features from encoder [batch_size, seq_len, input_dim]
            prev_latent: Previous latent state [batch_size, latent_dim] or None
            is_training: Whether in training mode (affects sampling)
            
        Returns:
            latent_samples: Sampled latent variables [batch_size, seq_len, latent_dim]
            prior_mean: Prior means [batch_size, seq_len, latent_dim]
            prior_log_var: Prior log variances [batch_size, seq_len, latent_dim]
            posterior_mean: Posterior means [batch_size, seq_len, latent_dim]
            posterior_log_var: Posterior log variances [batch_size, seq_len, latent_dim]
        """
        batch_size, seq_len, _ = encoder_features.shape
        
        if prev_latent is None:
            # Initialize with learned initial latent state
            prev_latent = self.init_latent.unsqueeze(0).expand(batch_size, -1)
        
        latent_samples = []
        prior_means = []
        prior_log_vars = []
        posterior_means = []
        posterior_log_vars = []
        
        current_latent = prev_latent
        
        for t in range(seq_len):
            # Current encoder features at time t
            h_t = encoder_features[:, t]  # [batch_size, input_dim]
            
            # Prior distribution: p(z_t | z_{<t}, h_t)
            prior_input = torch.cat([h_t, current_latent], dim=-1)
            prior_output = self.prior_net(prior_input)
            prior_mean = prior_output[:, :self.latent_dim]
            prior_log_var = prior_output[:, self.latent_dim:]
            
            if is_training:
                # Posterior distribution: q(z_t | x_t, z_{<t}, h_t)
                # For training, we use the current observation x_t (same as h_t here)
                posterior_input = torch.cat([h_t, current_latent, h_t], dim=-1)
                posterior_output = self.posterior_net(posterior_input)
                posterior_mean = posterior_output[:, :self.latent_dim]
                posterior_log_var = posterior_output[:, self.latent_dim:]
                
                # Sample from posterior during training
                z_t = self._reparameterize(posterior_mean, posterior_log_var)
            else:
                # During inference, sample from prior
                z_t = self._reparameterize(prior_mean, prior_log_var)
                posterior_mean = prior_mean
                posterior_log_var = prior_log_var
            
            latent_samples.append(z_t)
            prior_means.append(prior_mean)
            prior_log_vars.append(prior_log_var)
            posterior_means.append(posterior_mean)
            posterior_log_vars.append(posterior_log_var)
            
            # Update current latent state for next time step
            current_latent = z_t
        
        # Stack all time steps
        latent_samples = torch.stack(latent_samples, dim=1)
        prior_means = torch.stack(prior_means, dim=1)
        prior_log_vars = torch.stack(prior_log_vars, dim=1)
        posterior_means = torch.stack(posterior_means, dim=1)
        posterior_log_vars = torch.stack(posterior_log_vars, dim=1)
        
        return latent_samples, prior_means, prior_log_vars, posterior_means, posterior_log_vars
    
    def _reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from the latent distribution.
        
        Args:
            mean: Mean of the distribution
            log_var: Log variance of the distribution
            
        Returns:
            Sample from the distribution
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def compute_kl_divergence(self, 
                             prior_mean: torch.Tensor, 
                             prior_log_var: torch.Tensor,
                             posterior_mean: torch.Tensor, 
                             posterior_log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior distributions.
        
        Args:
            prior_mean: Prior means
            prior_log_var: Prior log variances
            posterior_mean: Posterior means
            posterior_log_var: Posterior log variances
            
        Returns:
            KL divergence loss
        """
        kl_div = 0.5 * torch.sum(
            prior_log_var - posterior_log_var + 
            (posterior_log_var.exp() + (posterior_mean - prior_mean).pow(2)) / prior_log_var.exp() - 1,
            dim=-1
        )
        return kl_div.mean()


class TemporalVariationalModule(nn.Module):
    """
    Enhanced variational module with explicit temporal dependencies.
    Models the temporal evolution of latent variables more explicitly.
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 temporal_hidden_dim: int = 128):
        super(TemporalVariationalModule, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Temporal LSTM for modeling latent evolution
        self.temporal_lstm = nn.LSTM(
            input_size=input_dim,  # Use input_dim instead of latent_dim
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Prior network with temporal context
        self.prior_net = nn.Sequential(
            nn.Linear(input_dim + temporal_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Posterior network with temporal context
        self.posterior_net = nn.Sequential(
            nn.Linear(input_dim + temporal_hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Initial latent state
        self.init_latent = nn.Parameter(torch.randn(latent_dim))
        
    def forward(self, 
                encoder_features: torch.Tensor,
                is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with explicit temporal modeling.
        """
        batch_size, seq_len, _ = encoder_features.shape
        
        # Initialize latent sequence
        latent_samples = []
        prior_means = []
        prior_log_vars = []
        posterior_means = []
        posterior_log_vars = []
        
        # Process through temporal LSTM
        temporal_features, _ = self.temporal_lstm(encoder_features)
        
        for t in range(seq_len):
            h_t = encoder_features[:, t]
            temp_t = temporal_features[:, t]
            
            # Prior: p(z_t | z_{<t}, h_t, temp_t)
            prior_input = torch.cat([h_t, temp_t], dim=-1)
            prior_output = self.prior_net(prior_input)
            prior_mean = prior_output[:, :self.latent_dim]
            prior_log_var = prior_output[:, self.latent_dim:]
            
            if is_training:
                # Posterior: q(z_t | x_t, z_{<t}, h_t, temp_t)
                posterior_input = torch.cat([h_t, temp_t, h_t], dim=-1)
                posterior_output = self.posterior_net(posterior_input)
                posterior_mean = posterior_output[:, :self.latent_dim]
                posterior_log_var = posterior_output[:, self.latent_dim:]
                
                z_t = self._reparameterize(posterior_mean, posterior_log_var)
            else:
                z_t = self._reparameterize(prior_mean, prior_log_var)
                posterior_mean = prior_mean
                posterior_log_var = prior_log_var
            
            latent_samples.append(z_t)
            prior_means.append(prior_mean)
            prior_log_vars.append(prior_log_var)
            posterior_means.append(posterior_mean)
            posterior_log_vars.append(posterior_log_var)
        
        # Stack all time steps
        latent_samples = torch.stack(latent_samples, dim=1)
        prior_means = torch.stack(prior_means, dim=1)
        prior_log_vars = torch.stack(prior_log_vars, dim=1)
        posterior_means = torch.stack(posterior_means, dim=1)
        posterior_log_vars = torch.stack(posterior_log_vars, dim=1)
        
        return latent_samples, prior_means, prior_log_vars, posterior_means, posterior_log_vars
    
    def _reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def compute_kl_divergence(self, 
                             prior_mean: torch.Tensor, 
                             prior_log_var: torch.Tensor,
                             posterior_mean: torch.Tensor, 
                             posterior_log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence."""
        kl_div = 0.5 * torch.sum(
            prior_log_var - posterior_log_var + 
            (posterior_log_var.exp() + (posterior_mean - prior_mean).pow(2)) / prior_log_var.exp() - 1,
            dim=-1
        )
        return kl_div.mean()
