import datetime
def time():
    # Get the current time
    current_time = datetime.datetime.now()

    # Format the current time
    formatted_time = current_time.strftime("%H:%M:%S")

    # Print the formatted time
    print("Current Time:", formatted_time)
time()
import os
from absl import app, flags
import numpy as np
from ml_collections import config_flags
from icecream import ic
import torch

import tqdm
import wandb

import sys
sys.path.append('/scratch/bdaw/kaiyan289/icvf_pytorch')
from network import Ensemble
from utils import set_seed
from d4rl_utils import make_env, get_dataset
from dataset import GCSDataset
from icvf_agent import create_agent
from wandb_utils import setup_wandb

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'maze2d-open-dense-v0', 'Environment name.')
flags.DEFINE_string('save_dir', f'experiment_output/', 'Logging dir.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 100, 'Metric logging interval.')
flags.DEFINE_integer('eval_interval', 25000, 'Visualization interval.')
flags.DEFINE_integer('save_interval', 100000, 'Save interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2e5), 'Number of training steps.')
flags.DEFINE_list('hidden_dims', [256, 256], 'Hidden sizes.')

from icvf_config import wandb_config, config, gcdataset_config

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', config, lock_config=False)
config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(_):
    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    setup_wandb(params_dict, **FLAGS.wandb)
    
    print(wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    env = make_env(FLAGS.env_name)
    dataset = get_dataset(env)
    #dataset: observations, actions, rewards, masks:1-terminals, dones_float:next_obs != obs[i+1] or terminal, next_observations
    set_seed(FLAGS.seed, env=env)
    
    gc_dataset = GCSDataset(dataset, **FLAGS.gcdataset.to_dict())
    state_dim = env.observation_space.shape[0]
    print(f'Input dim: {state_dim}')
    hidden_dims = [int(h) for h in FLAGS.hidden_dims]
    
    value_net = Ensemble(state_dim, hidden_dims=hidden_dims)
    agent = create_agent(FLAGS.seed, value_net, FLAGS.config)

    for i in range(1, FLAGS.max_steps + 1):
        batch = gc_dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            debug_statistics = get_debug_statistics(agent, batch)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            wandb.log(train_metrics, step=i)

        if i % FLAGS.save_interval == 0:
            #save the pytorch model of the agent.value_fn
            torch.save(agent.value_fn.state_dict(), os.path.join(FLAGS.save_dir, f'phi_{i}.pt'))
            print(f'Saved model at step {i} to {FLAGS.save_dir}')
            # save the model
            
            # save_dict = dict(
            #     agent=flax.serialization.to_state_dict(agent),
            #     config=FLAGS.config.to_dict()
            # )
            # fname = os.path.join(FLAGS.save_dir, f'params.pkl')
            # print(f'Saving to {fname}')
            # with open(fname, "wb") as f:
            #     pickle.dump(save_dict, f)

###################################################################################################
#
# Helper functions for visualization
#
###################################################################################################

def get_debug_statistics(agent, batch):
    value_fn = agent.value_fn

    def get_info(s, g, z):
        if agent.config['no_intent']:
            return value_fn.get_info(s, g, torch.ones_like(z).to(device))[0]
        else:
            return value_fn.get_info(s, g, z)[0]

    s = torch.tensor(batch['observations'], dtype=torch.float32, device=device)
    g = torch.tensor(batch['goals'], dtype=torch.float32, device=device)
    z = torch.tensor(batch['desired_goals'], dtype=torch.float32, device=device)

    info_ssz = get_info(s, s, z)
    info_szz = get_info(s, z, z)
    info_sgz = get_info(s, g, z)
    info_sgg = get_info(s, g, g)
    info_szg = get_info(s, z, g)

    if 'phi' in info_sgz:
        stats = {
            'phi_norm': torch.norm(info_sgz['phi'], dim=-1).mean().item(),
            'psi_norm': torch.norm(info_sgz['psi'], dim=-1).mean().item(),
        }
    else:
        stats = {}

    stats.update({
        'v_ssz': info_ssz['v'].mean().item(),
        'v_szz': info_szz['v'].mean().item(),
        'v_sgz': info_sgz['v'].mean().item(),
        'v_sgg': info_sgg['v'].mean().item(),
        'v_szg': info_szg['v'].mean().item(),
        'diff_szz_szg': (info_szz['v'] - info_szg['v']).mean().item(),
        'diff_sgg_sgz': (info_sgg['v'] - info_sgz['v']).mean().item(),
    })
    return stats

if __name__ == '__main__':
    app.run(main)

