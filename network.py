from typing import Dict, Sequence
import torch
import torch.nn as nn

# class LayerNormMLP(nn.Module):
#     def __init__(self, dims: Sequence[int], activation=nn.GELU, activate_final=False):
#         super(LayerNormMLP, self).__init__()
#         layers = []
#         for i in range(len(dims) - 1):
#             layers.append(nn.Linear(dims[i], dims[i+1]))
#             if i + 2 < len(dims) or activate_final:
#                 layers.append(activation())
#                 layers.append(nn.LayerNorm(dims[i+1]))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)
    
class NormalMLP(nn.Module):
    def __init__(self, hidden_dims, activation=nn.GELU, activate_final=False):
        super(NormalMLP, self).__init__()
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i + 1 < len(hidden_dims) or activate_final:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x) 

class MultilinearVF(nn.Module):
    def __init__(self, input_dim, hidden_dims: Sequence[int], use_layer_norm: bool = False):
        super(MultilinearVF, self).__init__()
        network_cls = NormalMLP
        dims = [input_dim] + hidden_dims
        T_dims = [hidden_dims[-1]] + hidden_dims
        self.phi_net = network_cls(dims, activate_final=True)
        self.psi_net = network_cls(dims, activate_final=True)
        self.T_net = network_cls(T_dims, activate_final=True)

        self.matrix_a = nn.Linear(dims[-1], dims[-1])
        self.matrix_b = nn.Linear(dims[-1], dims[-1])

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
        
class Ensemble(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Ensemble, self).__init__()
        self.model_1 = MultilinearVF(*args, **kwargs)
        self.model_2 = MultilinearVF(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model_1(*args, **kwargs), self.model_2(*args, **kwargs)

    def get_info(self, *args, **kwargs):
        return self.model_1.get_info(*args, **kwargs), self.model_2.get_info(*args, **kwargs)

    def get_phi(self, *args, **kwargs):
        return self.model_1.get_phi(*args, **kwargs), self.model_2.get_phi(*args, **kwargs)