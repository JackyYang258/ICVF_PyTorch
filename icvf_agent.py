from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ICVFAgent(nn.Module):
    def __init__(self, value_fn, optimizer_factory, config):
        super().__init__()
        self.value_fn = value_fn.to(device)
        self.target_value_fn = deepcopy(value_fn).requires_grad_(False).to(device)
        self.config = config

        self.value_optimizer = optimizer_factory(self.value_fn.parameters()) 
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=config.optim_kwargs.learning_rate, eps=config.optim_kwargs.eps)
        self.discount = config['discount']
        self.expectile = config['expectile']
        self.no_intent = config['no_intent']
        self.min_q = config['min_q']
        self.target_update_rate = config['target_update_rate']
        
        self.update_interval = 1
        self.update_counter = 0

    def update(self, batch):
        # Unpack batch
        obs = torch.tensor(batch['observations'], dtype=torch.float32).to(device)
        next_obs = torch.tensor(batch['next_observations'], dtype=torch.float32).to(device)
        goals = torch.tensor(batch['goals'], dtype=torch.float32).to(device)
        desired_goals = torch.tensor(batch['desired_goals'], dtype=torch.float32).to(device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(device)
        masks = torch.tensor(batch['masks'], dtype=torch.float32).to(device)
        desired_rewards = torch.tensor(batch['desired_rewards'], dtype=torch.float32).to(device)
        desired_masks = torch.tensor(batch['desired_masks'], dtype=torch.float32).to(device)

        if self.no_intent:
            desired_goals = torch.ones_like(desired_goals)

        # Compute target values
        with torch.no_grad():
            next_v1_gz, next_v2_gz = self.target_value_fn(next_obs, goals, desired_goals)
            q1_gz = rewards + self.discount * masks * next_v1_gz
            q2_gz = rewards + self.discount * masks * next_v2_gz

        # Compute current values
        v1_gz, v2_gz = self.value_fn(obs, goals, desired_goals)

        with torch.no_grad():
            next_v1_zz, next_v2_zz = self.target_value_fn(next_obs, desired_goals, desired_goals)
            next_v_zz = torch.min(next_v1_zz, next_v2_zz) if self.min_q else (next_v1_zz + next_v2_zz) / 2
            q_zz = desired_rewards + self.discount * desired_masks * next_v_zz
            v1_zz, v2_zz = self.target_value_fn(obs, desired_goals, desired_goals)
            v_zz = (v1_zz + v2_zz) / 2
            adv = q_zz - v_zz

        if self.no_intent:
            adv = torch.zeros_like(adv)

        # Compute losses
        def expectile_loss(adv, diff, expectile):
            weight = torch.where(adv >= 0, expectile, 1 - expectile)
            return torch.mean(weight * diff ** 2)

        value_loss1 = expectile_loss(adv, q1_gz - v1_gz, self.expectile)
        value_loss2 = expectile_loss(adv, q2_gz - v2_gz, self.expectile)
        value_loss = value_loss1 + value_loss2

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update target networks
        self.update_counter += 1
        if self.update_counter % self.update_interval == 0:
            self.update_target_network()
        
        def masked_mean(x, mask):
            mask = torch.tensor(mask, device=x.device, dtype=x.dtype)
            result = (x * mask).sum() / (1e-5 + mask.sum())
            return result.item()

        return {
            'value_loss': value_loss.item(),
            'v_gz max': v1_gz.max().item(),
            'v_gz min': v1_gz.min().item(),
            'v_zz': v_zz.mean().item(),
            'v_gz': v1_gz.mean().item(),
            'abs adv mean': adv.abs().mean().item(),
            'adv mean': adv.mean().item(),
            'adv max': adv.max().item(),
            'adv min': adv.min().item(),
            'accept prob': (adv >= 0).float().mean().item(),
            'reward mean': rewards.mean().item(),
            'mask mean': masks.mean().item(),
            'q_gz max': q1_gz.max().item(),
            'value_loss1': masked_mean((q1_gz-v1_gz)**2, batch['masks']), # Loss on s \neq s_+
            'value_loss2': masked_mean((q1_gz-v1_gz)**2, 1.0 - batch['masks']), # Loss on s = s_+
        }
    
    def update_target_network(self, soft_update = True):
        with torch.no_grad():
            if soft_update:
                for target_param, param in zip(self.target_value_fn.parameters(), self.value_fn.parameters()):
                    target_param.data.copy_(self.target_update_rate * param.data + (1.0 - self.target_update_rate) * target_param.data)
            else:
                self.target_value_fn.load_state_dict(self.value_fn.state_dict())

def optimizer_factory(params):
    return optim.Adam(params, lr=0.00005, eps=0.0003125)

def create_agent(seed, value_fn, config):
    return ICVFAgent(value_fn, optimizer_factory, config)
