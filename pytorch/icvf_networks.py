from jaxrl_m.typing import *
from jaxrl_m.networks import MLP, get_latent, default_init, ensemblize

import flax.linen as nn
import jax.numpy as jnp

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class LayerNormMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activate_final=False):
        super().__init__()
        # add input_dim to build module
        layers = []
        
        act = nn.GELU
        dims = input_dim + list(hidden_dims)
        
        for i in range(len(dims) - 1):
            if i < len(dims) - 2 or activate_final:
                layers += [nn.Linear(dims[i], dims[i + 1]), act()]
                layers += [nn.LayerNorm(dims[i + 1])]

        self.fcs = nn.Sequential(*layers)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = F.gelu(x)
                x = F.layer_norm(x, x.size()[1:])
        return x

class ICVFWithEncoder(nn.Module):
    encoder: nn.Module
    vf: nn.Module

    def get_encoder_latent(self, observations):     
        return get_latent(self.encoder, observations)
    
    def get_phi(self, observations):
        latent = get_latent(self.encoder, observations)
        return self.vf.get_phi(latent)

    def __call__(self, observations, outcomes, intents):
        latent_s = get_latent(self.encoder, observations)
        latent_g = get_latent(self.encoder, outcomes)
        latent_z = get_latent(self.encoder, intents)
        return self.vf(latent_s, latent_g, latent_z)
    
    def get_info(self, observations, outcomes, intents):
        latent_s = get_latent(self.encoder, observations)
        latent_g = get_latent(self.encoder, outcomes)
        latent_z = get_latent(self.encoder, intents)
        return self.vf.get_info(latent_s, latent_g, latent_z)

def create_icvf(icvf_cls_or_name, encoder=None, ensemble=True, **kwargs):    
    if isinstance(icvf_cls_or_name, str):
        icvf_cls = icvfs[icvf_cls_or_name]
    else:
        icvf_cls = icvf_cls_or_name

    if ensemble:
        vf = ensemblize(icvf_cls, 2, methods=['__call__', 'get_info', 'get_phi'])(**kwargs)
    else:
        vf = icvf_cls(**kwargs)
    
    if encoder is None:
        return vf

    return ICVFWithEncoder(encoder, vf)



##
#
# Actual ICVF definitions below
##

class ICVFTemplate(nn.Module):

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Returns useful metrics
        raise NotImplementedError
    
    def get_phi(self, observations):
        # Returns phi(s) for downstream use
        raise NotImplementedError
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        # Returns V(s, g, z)
        raise NotImplementedError

class MonolithicVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        # todo: add input_dim to build module
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.net = network_cls((*self.hidden_dims, 1), activate_final=False)

    def get_info(self, observations, outcomes, z) -> Dict[str, Tensor]:
        x = torch.cat([observations, outcomes, z], dim = -1)
        v = self.net(x)
        return {
            'v': v.squeeze(-1),
            'psi': outcomes,
            'z': z,
            'phi': observations,
        }
    
    def get_phi(self, observations):
        print('Warning: StandardVF does not define a state representation phi(s). Returning phi(s) = s')
        return observations
    
    def forward(self, observations, outcomes, z):
        x = torch.cat([observations, outcomes, z], dim=-1)
        v = self.net(x)
        return v.squeeze(-1)

class MultilinearVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.phi_net = network_cls(self.hidden_dims, activate_final=True, name='phi')
        self.psi_net = network_cls(self.hidden_dims, activate_final=True, name='psi')

        self.T_net =  network_cls(self.hidden_dims, activate_final=True, name='T')

        # todo find the input dim
        self.matrix_a = nn.Linear(input_dim, self.hidden_dims[-1])
        self.matrix_b = nn.Linear(input_dim, self.hidden_dims[-1])
        
    
    def forward(self, observations, outcomes, intents):
        return self.get_info(observations, outcomes, intents)['v']

    def get_phi(self, observations):
        return self.phi_net(observations)

    def get_info(self, observations: Tensor, outcomes: Tensor, intents: Tensor) -> Dict[str, Tensor]:
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents)
        Tz = self.T_net(z)

        # T(z) should be a dxd matrix, but having a network output d^2 parameters is inefficient
        # So we'll make a low-rank approximation to T(z) = (diag(Tz) * A * B * diag(Tz))
        # where A and B are (fixed) dxd matrices and Tz is a d-dimensional parameter dependent on z

        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(axis=-1)

        return {
            'v': v,
            'phi': phi,
            'psi': psi,
            'Tz': Tz,
            'z': z,
            'phi_z': phi_z,
            'psi_z': psi_z,
        }

icvfs = {
    'multilinear': MultilinearVF,
    'monolithic': MonolithicVF,
}