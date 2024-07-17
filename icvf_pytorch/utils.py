from typing import Dict, Sequence
import torch
import torch.nn as nn

import numpy as np
import random

class LayerNormMLP(nn.Module):
    def __init__(self, hidden_dims: Sequence[int], activation=nn.GELU, activate_final=False):
        super(LayerNormMLP, self).__init__()
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i + 1 < len(hidden_dims) or activate_final:
                layers.append(activation())
                layers.append(nn.LayerNorm(hidden_dims[i+1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class NormalMLP(nn.Module):
    def __init__(self, hidden_dims, activation=nn.GELU, activate_final=False):
        super(LayerNormMLP, self).__init__()
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i + 1 < len(hidden_dims) or activate_final:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

class MultilinearVF(nn.Module):
    def __init__(self, hidden_dims: Sequence[int], use_layer_norm: bool = False):
        super(MultilinearVF, self).__init__()
        network_cls = LayerNormMLP if use_layer_norm else NormalMLP
        self.phi_net = network_cls(hidden_dims, activate_final=True)
        self.psi_net = network_cls(hidden_dims, activate_final=True)
        self.T_net = network_cls(hidden_dims, activate_final=True)

        self.matrix_a = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.matrix_b = nn.Linear(hidden_dims[-1], hidden_dims[-1])

    def forward(self, observations: torch.Tensor, outcomes: torch.Tensor, intents: torch.Tensor) -> torch.Tensor:
        return self.get_info(observations, outcomes, intents)['v']

    def get_phi(self, observations: torch.Tensor) -> torch.Tensor:
        return self.phi_net(observations)

    def get_info(self, observations: torch.Tensor, outcomes: torch.Tensor, intents: torch.Tensor) -> Dict[str, torch.Tensor]:
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents)
        Tz = self.T_net(z)

        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(dim=-1)

        return {
            'v': v,
            'phi': phi,
            'psi': psi,
            'Tz': Tz,
            'z': z,
            'phi_z': phi_z,
            'psi_z': psi_z,
        }

def create_icvf(icvf_cls_or_name, encoder=None, ensemble=True, **kwargs):
    
    vf = MultilinearVF(**kwargs)

    return vf

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)