import os
from absl import app, flags
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import flax

from pathlib import Path

import gym
import d4rl
import numpy as np
import torch
from tqdm import trange

import ml_collections

import tqdm
import sys
sys.path.append('/home/ysq/project/RL/icvf_pytorch')
from src import icvf_learner as learner
from src.icvf_networks import icvfs, create_icvf
from icvf_envs.antmaze import d4rl_utils, d4rl_ant, ant_diagnostics, d4rl_pm
from src.gc_dataset import GCSDataset
from src import viz_utils

from jaxrl_m.wandb import setup_wandb, default_wandb_config
import wandb
from jaxrl_m.evaluation import supply_rng, evaluate, evaluate_with_trajectories

from ml_collections import config_flags
import pickle
from jaxrl_m.dataset import Dataset
from icecream import ic
from iql.src.iql import ImplicitQLearning
from iql.src.policy import GaussianPolicy, DeterministicPolicy
from iql.src.value_functions import TwinQ, ValueFunction
from iql.src.util import return_range, set_seed, Log, sample_batch, torchify

USINGICVF = True
FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'hopper-medium-v2', 'Environment name.')

flags.DEFINE_string('save_dir', f'experiment_output/', 'Logging dir.')

flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Metric logging interval.')
flags.DEFINE_integer('eval_interval', 25000, 'Visualization interval.')
flags.DEFINE_integer('save_interval', 100000, 'Save interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of training steps.')

flags.DEFINE_enum('icvf_type', 'multilinear', list(icvfs), 'Which model to use.')
flags.DEFINE_list('hidden_dims', [256, 256], 'Hidden sizes.')

flags.DEFINE_string('log_directory', 'iql_logs', 'Logging dir.')
# flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('discount', 0.99, 'Discount factor.')
flags.DEFINE_integer('hidden_dim', 256, 'Hidden dimension size.')
flags.DEFINE_integer('n_hidden', 2, 'Number of hidden layers.')
flags.DEFINE_integer('n_steps', 10**6, 'Number of training steps.')
# flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_float('alpha', 0.005, 'Alpha parameter.')
flags.DEFINE_float('tau', 0.7, 'Tau parameter.')
flags.DEFINE_float('beta', 3.0, 'Beta parameter.')
# flags.DEFINE_boolean('deterministic_policy', False, 'Use deterministic policy.')
flags.DEFINE_integer('deterministic_policy', 0, 'Use deterministic policy.')
flags.DEFINE_integer('eval_period', 5000, 'Evaluation period.')
flags.DEFINE_integer('n_eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('max_episode_steps', 1000, 'Maximum steps per episode.')

iql_config = ml_collections.ConfigDict({
        'env_name': 'hopper-medium-v2',
        'log_dir': 'logs',
        'seed': 0,
        'discount': 0.99,
        'hidden_dim': 256,
        'n_hidden': 2,
        'n_steps': 10**6,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'alpha': 0.005,
        'tau': 0.7,
        'beta': 3.0,
        'deterministic_policy': False,
        'eval_period': 5000,
        'n_eval_episodes': 10,
        'max_episode_steps': 1000,
    })

def update_dict(d, additional):
    d.update(additional)
    return d

wandb_config = update_dict(
    default_wandb_config(),
    {
        'project': 'icvf',
        'group': 'icvf',
        # 'name': '{icvf_type}_{env_name}',
        'name': 'iql_with_icvf',
        'mode': 'online',
    }
)

config = update_dict(
    learner.get_default_config(),
    # config = ml_collections.ConfigDict({
    #     'optim_kwargs': {
    #         'learning_rate': 0.00005,
    #         'eps': 0.0003125
    #     }, # LR for vision here. For FC, use standard 1e-3
    #     'discount': 0.99,
    #     'expectile': 0.9,  # The actual tau for expectiles.
    #     'target_update_rate': 0.005,  # For soft target updates.
    #     'no_intent': False,
    #     'min_q': True,
    #     'periodic_target_update': False,
    # })
    {
    'discount': 0.99, 
     'optim_kwargs': { # Standard Adam parameters for non-vision
            'learning_rate': 3e-4,
            'eps': 1e-8
        }
    }
)

gcdataset_config = GCSDataset.get_default_config()

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', config, lock_config=False)
config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)
config_flags.DEFINE_config_dict('iql', iql_config, lock_config=False)

visual = False

def get_env_and_dataset_for_downstream(log, env_name, max_episode_steps):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    for k, v in dataset.items():
        dataset[k] = torchify(v[:200000])

    return env, dataset

def main(_):
    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    setup_wandb(params_dict, **FLAGS.wandb)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    env = d4rl_utils.make_env(FLAGS.env_name)
    dataset = d4rl_utils.get_dataset(env)
    #dataset: observations, actions, rewards, masks:1-terminals, dones_float:next_obs != obs[i+1] or terminal, next_observations
    gc_dataset = GCSDataset(dataset, **FLAGS.gcdataset.to_dict())
    example_batch = gc_dataset.sample(1)
    hidden_dims = tuple([int(h) for h in FLAGS.hidden_dims])
    value_def = create_icvf(FLAGS.icvf_type, hidden_dims=hidden_dims)

    agent = learner.create_learner(FLAGS.seed,
                    example_batch['observations'],
                    value_def,
                    **FLAGS.config)

    if(visual):
        visualizer = DebugPlotGenerator(FLAGS.env_name, gc_dataset)

    if USINGICVF:
        for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                        smoothing=0.1,
                        dynamic_ncols=True):
            batch = gc_dataset.sample(FLAGS.batch_size)  
            agent, update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                debug_statistics = get_debug_statistics(agent, batch)
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
                wandb.log(train_metrics, step=i)

            if i % FLAGS.eval_interval == 0 and visual:
                visualizations = visualizer.generate_debug_plots(agent)
                eval_metrics = {f'visualizations/{k}': v for k, v in visualizations.items()}
                wandb.log(eval_metrics, step=i)

            if i % FLAGS.save_interval == 0:
                save_dict = dict(
                    agent=flax.serialization.to_state_dict(agent),
                    config=FLAGS.config.to_dict()
                )

                fname = os.path.join(FLAGS.save_dir, f'params.pkl')
                print(f'Saving to {fname}')
                with open(fname, "wb") as f:
                    pickle.dump(save_dict, f)
    # we can use get_latent(agent, obs) to get the latent state representation
    
    wandb.finish()
    setup_wandb(params_dict, **FLAGS.wandb)
    
    args = FLAGS.iql
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir)/args.env_name, vars(args))
    log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_dataset_for_downstream(log, args.env_name, args.max_episode_steps)
    if USINGICVF:
        print(dataset['observations'].shape)
        obs = get_latent(agent, jnp.array(np.array(dataset['observations'].cpu())))
        dataset['observations'] = torch.mean(torch.tensor(np.array(obs)).to('cuda:0'),dim=0)
        obs = get_latent(agent, jnp.array(np.array(dataset['next_observations'].cpu())))
        dataset['next_observations'] = torch.mean(torch.tensor(np.array(obs)).to('cuda:0'),dim=0)
        print(dataset['next_observations'].shape)
        print(dataset['observations'].device)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, policy, args.max_episode_steps) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })
        return eval_returns.mean(), eval_returns.std(), normalized_returns.mean(), normalized_returns.std()

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )
    def evaluate_policy(env, policy, max_episode_steps, deterministic=True):
        obs = env.reset()
        total_reward = 0.
        for _ in range(max_episode_steps):
            with torch.no_grad():
                if USINGICVF:
                    action = policy.act(
                        torch.mean(torch.tensor(np.array(get_latent(agent,jnp.array(obs)))).to('cuda:0'),dim=0),
                        deterministic=deterministic).cpu().numpy()
                else:
                    action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
            else:
                obs = next_obs
        return total_reward

    for step in trange(args.n_steps):
        iql.update(**sample_batch(dataset, args.batch_size))
        if (step+1) % args.eval_period == 0:
            mean_reward, return_std, norm_mean_reward, norm_return_std = eval_policy()
            wandb.log({"iql/mean_reward_iql":mean_reward}, step=step)
            wandb.log({"iql/return_std_iql":return_std}, step=step)
            wandb.log({"iql/norm_mean_reward_iql":norm_mean_reward}, step=step)
            wandb.log({"iql/norm_return_std_iql":norm_return_std}, step=step)
            

    torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()
    
    
###################################################################################################
#
# Creates wandb plots
#
###################################################################################################
class DebugPlotGenerator:
    def __init__(self, env_name, gc_dataset):
        if 'antmaze' in env_name:
            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
            viz_library = d4rl_ant
            self.viz_things = (viz_env, viz_dataset, viz_library, init_state)

        elif 'maze' in env_name:
            viz_env, viz_dataset = d4rl_pm.get_gcenv_and_dataset(env_name)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (3, 4)
            viz_library = d4rl_pm
            self.viz_things = (viz_env, viz_dataset, viz_library, init_state)
        else:
            raise NotImplementedError('Visualization not implemented for this environment')

        # intent_set_indx = np.random.default_rng(0).choice(dataset.size, FLAGS.config.n_intents, replace=False)
        # Chosen by hand for `antmaze-large-diverse-v2` to get a nice spread of goals, use the above line for random selection

        intent_set_indx = np.array([184588, 62200, 162996, 110214, 4086, 191369, 92549, 12946, 192021])
        self.intent_set_batch = gc_dataset.sample(9, indx=intent_set_indx)
        self.example_trajectory = gc_dataset.sample(50, indx=np.arange(1000, 1050))



    def generate_debug_plots(self, agent):
        example_trajectory = self.example_trajectory
        intents = self.intent_set_batch['observations']
        (viz_env, viz_dataset, viz_library, init_state) = self.viz_things

        visualizations = {}
        traj_metrics = get_traj_v(agent, example_trajectory)
        value_viz = viz_utils.make_visual_no_image(traj_metrics, 
            [
            partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()
                ]
        )
        visualizations['value_traj_viz'] = wandb.Image(value_viz)

        if 'maze' in FLAGS.env_name:
            print('Visualizing intent policies and values')
            # Policy visualization
            methods = [
                partial(viz_library.plot_policy, policy_fn=partial(get_policy, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            image = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_policies'] = wandb.Image(image)

            # Value visualization
            methods = [
                partial(viz_library.plot_value, value_fn=partial(get_values, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            image = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_values'] = wandb.Image(image)

            for idx in range(3):
                methods = [
                    partial(viz_library.plot_policy, policy_fn=partial(get_policy, agent, intent=intents[idx])),
                    partial(viz_library.plot_value, value_fn=partial(get_values, agent, intent=intents[idx]))
                ]
                image = viz_library.make_visual(viz_env, viz_dataset, methods)
                visualizations[f'intent{idx}'] = wandb.Image(image)

            image_zz = viz_library.gcvalue_image(
                viz_env,
                viz_dataset,
                partial(get_v_zz, agent),
            )
            image_gz = viz_library.gcvalue_image(
                viz_env,
                viz_dataset,
                partial(get_v_gz, agent, init_state),
            )
            visualizations['v_zz'] = wandb.Image(image_zz)
            visualizations['v_gz'] = wandb.Image(image_gz)
        return visualizations

###################################################################################################
#
# Helper functions for visualization
#
###################################################################################################
@jax.jit
def get_latent(agent, obs):
    def get_phi(s):
        return agent.value(s, method='get_phi')
    return get_phi(obs)

@jax.jit
def get_values(agent, observations, intent):
    def get_v(observations, intent):
        intent = intent.reshape(1, -1)
        intent_tiled = jnp.tile(intent, (observations.shape[0], 1))
        v1, v2 = agent.value(observations, intent_tiled, intent_tiled)
        return (v1 + v2) / 2    
    return get_v(observations, intent)

@jax.jit
def get_policy(agent, observations, intent):
    def v(observations):
        def get_v(observations, intent):
            intent = intent.reshape(1, -1)
            intent_tiled = jnp.tile(intent, (observations.shape[0], 1))
            v1, v2 = agent.value(observations, intent_tiled, intent_tiled)
            return (v1 + v2) / 2    
            
        return get_v(observations, intent).mean()

    grads = jax.grad(v)(observations)
    policy = grads[:, :2]
    return policy / jnp.linalg.norm(policy, axis=-1, keepdims=True)

@jax.jit
def get_debug_statistics(agent, batch):
    def get_info(s, g, z):
        if agent.config['no_intent']:
            return agent.value(s, g, jnp.ones_like(z), method='get_info')
        else:
            return agent.value(s, g, z, method='get_info')

    s = batch['observations']
    g = batch['goals']
    z = batch['desired_goals']
    
    def get_latent(s):
        return agent.value(s, method='get_phi')
    print(s)
    print(g)
    print(z)
    latent = get_latent(s)

    info_ssz = get_info(s, s, z)
    info_szz = get_info(s, z, z)
    info_sgz = get_info(s, g, z)
    info_sgg = get_info(s, g, g)
    info_szg = get_info(s, z, g)
    print(info_szz['v'])
    print(info_szz['v'].mean())
    if 'phi' in info_sgz:
        stats = {
            'phi_norm': jnp.linalg.norm(info_sgz['phi'], axis=-1).mean(),
            'psi_norm': jnp.linalg.norm(info_sgz['psi'], axis=-1).mean(),
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

@jax.jit
def get_gcvalue(agent, s, g, z):
    v_sgz_1, v_sgz_2 = agent.value(s, g, z)
    return (v_sgz_1 + v_sgz_2) / 2

def get_v_zz(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal, goal)

def get_v_gz(agent, initial_state, target_goal, observations):
    initial_state = jnp.tile(initial_state, (observations.shape[0], 1))
    target_goal = jnp.tile(target_goal, (observations.shape[0], 1))
    return get_gcvalue(agent, initial_state, observations, target_goal)

@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        return agent.value(s[None], g[None], g[None]).mean()
    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }

####################


if __name__ == '__main__':
    app.run(main)
