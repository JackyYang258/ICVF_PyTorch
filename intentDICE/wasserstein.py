from collections import OrderedDict
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class FullyConnectedNet(nn.Module):
    def __init__(self, ch_in, ch_out, chs=(128, 256, 256)):
        super().__init__()
        current_ch = ch_in
        components = OrderedDict()
        for i, ch in enumerate(chs):
            components[f'fc{i}'] = nn.Linear(current_ch, ch)
            components[f'act{i}'] = nn.ReLU()
            current_ch = ch

        components['fc_out'] = nn.Linear(current_ch, ch_out)
        self.f = nn.Sequential(components)

    def forward(self, x):
        return self.f(x)


def network_weight_matrices(model, max_norm, eps=1e-8):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            spectral_norm = torch.linalg.matrix_norm(w, ord=2)
            denom = max(1, spectral_norm / max_norm)
            module.weight.data = w / (denom + eps)
    return model


def _get_batch(iterator, loader):
    try:
        samples = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        samples = next(iterator)

    samples = samples[0][:,None]
    return samples, iterator


def _eval_wasserstein_model(p_loader, q_loader, h, device):
    p_length = len(p_loader.dataset)
    q_length = len(q_loader.dataset)

    p_length_inv = torch.DoubleTensor([1.]).to(device) / p_length
    q_length_inv = torch.DoubleTensor([1.]).to(device) / q_length
    h_p_expectation = torch.DoubleTensor([0.])
    h_q_expectation = torch.DoubleTensor([0.])
    with torch.no_grad():
        for data in p_loader:
            data = data[0][:,None].to(device)
            h_p_expectation += torch.sum(h(data).double() * p_length_inv).item()

        for data in q_loader:
            data = data[0][:,None].to(device)
            h_q_expectation += torch.sum(h(data).double() * q_length_inv).item()

    return (h_p_expectation - h_q_expectation).item()


def estimate_wasserstein_kantorovich_rubinstein(samples_p, samples_q, device, eval_steps=1000,
                                                bs=1024, eval_bs=2**15, optim='adam', lr=0.001,
                                                schedule='none', steps=100_00):
    h = FullyConnectedNet(1, 1)
    h.to(device)

    h = network_weight_matrices(h, 1)

    if optim == 'sgd':
        optimizer = torch.optim.SGD(h.parameters(), lr=lr, momentum=0.9)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(h.parameters(), lr=lr)
    else:
        raise NotImplementedError()

    if schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    elif schedule == 'none' or schedule is None:
        scheduler = None
    else:
        raise NotImplementedError()

    #prepare the dataset
    obs = phi(observations)
    next_observations = phi(next_observations)

    for i in trange(steps):

        p_samples, q_samples = p_samples.to(device), q_samples.to(device)

        optimizer.zero_grad()
        output_p = h(p_samples)
        output_q = h(q_samples)

        #minus loss as we have to maximize
        loss = -(torch.mean(output_p) - torch.mean(output_q))
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        h = network_weight_matrices(h, 1)

        if (i % eval_steps) == 0:
            current_d_estimate = _eval_wasserstein_model(eval_p_loader, eval_q_loader, h, device)
            print(f'Step {i}: Wasserstein-1 estimate {current_d_estimate:.5f}')

    current_d_estimate = _eval_wasserstein_model(eval_p_loader, eval_q_loader, h, device)
    print(f'Final Wasserstein-1 estimate {current_d_estimate:.5f}')
    return current_d_estimate