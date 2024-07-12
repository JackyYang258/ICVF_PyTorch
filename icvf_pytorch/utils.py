from typing import Dict, Sequence
import torch
import torch.nn as nn


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
        network_cls = LayerNormMLP if use_layer_norm else MLP
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
class MultilinearVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        network_cls = LayerNormMLP if self.use_layer_norm else NormalMLP
        self.phi_net = network_cls(self.hidden_dims, activate_final=True, name='phi')
        self.psi_net = network_cls(self.hidden_dims, activate_final=True, name='psi')

        self.T_net =  network_cls(self.hidden_dims, activate_final=True, name='T')

        self.matrix_a = nn.Dense(self.hidden_dims[-1], name='matrix_a')
        self.matrix_b = nn.Dense(self.hidden_dims[-1], name='matrix_b')
        
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> jnp.ndarray:
        return self.get_info(observations, outcomes, intents)['v']
        

    def get_phi(self, observations):
        return self.phi_net(observations)

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> Dict[str, jnp.ndarray]:
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

def create_icvf(icvf_cls_or_name, encoder=None, ensemble=True, **kwargs):
    
    vf = MultilinearVF(**kwargs)

    return ICVFWithEncoder(encoder, vf)