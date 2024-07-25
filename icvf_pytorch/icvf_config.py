import ml_collections
from ml_collections.config_dict import FieldReference

def default_wandb_config():
    config = ml_collections.ConfigDict()
    config.offline = True  # Syncs online or not?
    config.project = "jaxrl_m"  # WandB Project Name
    config.entity = FieldReference(None, field_type=str)  # Which entity to log as (default: your own user)

    group_name = FieldReference(None, field_type=str)  # Group name
    config.exp_prefix = group_name  # Group name (deprecated, but kept for backwards compatibility)
    config.group = group_name  # Group name

    experiment_name = FieldReference(None, field_type=str) # Experiment name
    config.name = experiment_name  # Run name (will be formatted with flags / variant)
    config.exp_descriptor = experiment_name  # Run name (deprecated, but kept for backwards compatibility)

    config.unique_identifier = ""  # Unique identifier for run (will be automatically generated unless provided)
    config.random_delay = 0  # Random delay for wandb.init (in seconds)
    return config

default_wandb_config().update(
    {
        'project': 'icvf',
        'group': 'icvf',
        # 'name': '{icvf_type}_{env_name}',
        'name': 'antmaze-large-diverse-v2',
    })

wandb_config = default_wandb_config()

config = ml_collections.ConfigDict({
        'optim_kwargs': {
            'learning_rate': 3e-4,
            'eps': 1e-8
        }, # LR for vision here. For FC, use standard 1e-3
        'discount': 0.99,
        'expectile': 0.9,  # The actual tau for expectiles.
        'target_update_rate': 0.005,  # For soft target updates.
        'no_intent': False,
        'min_q': True,
        'periodic_target_update': False,
    })
    
gcdataset_config = ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
            'p_samegoal': 0.5,
            'intent_sametraj': False,
            'max_distance': ml_collections.config_dict.placeholder(int),
            'curr_goal_shift': 0,
        })