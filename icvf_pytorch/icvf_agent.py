import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ICVFAgent(nn.Module):
    def __init__(self, value_fn, target_value_fn, optimizer_factory, config):
        super().__init__()
        self.value_fn = value_fn.to(DEFAULT_DEVICE)
        self.target_value_fn = copy.deepcopy(target_value_fn).requires_grad_(False).to(DEFAULT_DEVICE)
        self.config = config

        self.value_optimizer = optimizer_factory(self.value_fn.parameters())
        self.scheduler = CosineAnnealingLR(self.value_optimizer, self.config['max_steps'])
        self.alpha = config['alpha']
        self.discount = config['discount']
        self.expectile = config['expectile']
        self.no_intent = config['no_intent']
        self.min_q = config['min_q']

    def update(self, batch):
        # Unpack batch
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['goals']
        desired_goals = batch['desired_goals']
        rewards = batch['rewards']
        masks = batch['masks']
        desired_rewards = batch['desired_rewards']
        desired_masks = batch['desired_masks']

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
        self.scheduler.step()

        # Update target networks
        self.update_target_network()

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
        }

    def update_target_network(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_value_fn.parameters(), self.value_fn.parameters()):
                target_param.data.copy_(self.alpha * param.data + (1.0 - self.alpha) * target_param.data)

def optimizer_factory(params):
    return optim.Adam(params, lr=0.00005, eps=0.0003125)

def create_learner(seed, value_fn, config):
    torch.manual_seed(seed)
    value_fn = value_fn().to(DEFAULT_DEVICE)
    target_value_fn = copy.deepcopy(value_fn).requires_grad_(False).to(DEFAULT_DEVICE)
    return ICVFAgent(value_fn, target_value_fn, optimizer_factory, config)

def get_default_config():
    return {
        'discount': 0.99,
        'expectile': 0.9,
        'target_update_rate': 0.005,
        'no_intent': False,
        'min_q': True,
        'alpha': 0.005,
        'max_steps': 1000,
    }