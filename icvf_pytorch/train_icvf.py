import os
from absl import app, flags
import numpy as np
from ml_collections import config_flags
from icecream import ic

import tqdm
import wandb

from .network import MultilinearVF
from .utils import set_seed
from .d4rl import make_env, get_dataset
from .dataset import Dataset
from .icvf_agent import create_agent
from .wandb import setup_wandb

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'hopper-medium-v2', 'Environment name.')
flags.DEFINE_string('save_dir', f'experiment_output/', 'Logging dir.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Metric logging interval.')
flags.DEFINE_integer('eval_interval', 25000, 'Visualization interval.')
flags.DEFINE_integer('save_interval', 100000, 'Save interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(10), 'Number of training steps.')
flags.DEFINE_list('hidden_dims', [256, 256], 'Hidden sizes.')

from .config import wandb_config, config, gcdataset_config
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', config, lock_config=False)
config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)

def main(_):
    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    setup_wandb(params_dict, **FLAGS.wandb)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    env = make_env(FLAGS.env_name)
    dataset = get_dataset(env)
    #dataset: observations, actions, rewards, masks:1-terminals, dones_float:next_obs != obs[i+1] or terminal, next_observations
    set_seed(FLAGS.seed, env=env)
    
    gc_dataset = Dataset(dataset, **FLAGS.gcdataset.to_dict())
    input_dim = gc_dataset.observations.shape[1:]
    hidden_dims = [int(h) for h in FLAGS.hidden_dims]
    
    value_net = MultilinearVF(input_dim, hidden_dims=hidden_dims)
    agent = create_agent(FLAGS.seed, value_net, FLAGS.config.to_dict())

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        batch = gc_dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            debug_statistics = get_debug_statistics(agent, batch)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            wandb.log(train_metrics, step=i)

        if i % FLAGS.save_interval == 0:
            pass
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
            return value_fn.get_info(s, g, np.ones_like(z))
        else:
            return value_fn.get_info(s, g, z)

    s = batch['observations']
    g = batch['goals']
    z = batch['desired_goals']

    info_ssz = get_info(s, s, z)
    info_szz = get_info(s, z, z)
    info_sgz = get_info(s, g, z)
    info_sgg = get_info(s, g, g)
    info_szg = get_info(s, z, g)

    if 'phi' in info_sgz:
        stats = {
            'phi_norm': np.linalg.norm(info_sgz['phi'], axis=-1).mean(),
            'psi_norm': np.linalg.norm(info_sgz['psi'], axis=-1).mean(),
        }
    else:
        stats = {}

    stats.update({
        'v_ssz': info_ssz['v'].mean(),
        'v_szz': info_szz['v'].mean(),
        'v_sgz': info_sgz['v'].mean(),
        'v_sgg': info_sgg['v'].mean(),
        'v_szg': info_szg['v'].mean(),
        'diff_szz_szg': (info_szz['v'] - info_szg['v']).mean(),
        'diff_sgg_sgz': (info_sgg['v'] - info_sgz['v']).mean(),
    })
    return stats

if __name__ == '__main__':
    app.run(main)

