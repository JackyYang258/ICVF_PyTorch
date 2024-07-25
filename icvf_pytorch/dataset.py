import numpy as np
import ml_collections
import dataclasses

class FrozenDict():
    def __init__(self, data):
        self._data = data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)

def get_size(data) -> int:
    """Helper function to determine the size of the dataset."""
    for key, value in data.items():
        return value.shape[0]

class Dataset(FrozenDict):
    """
    A class for storing (and retrieving batches of) data in nested dictionary format.

    Example:
        dataset = Dataset({
            'observations': {
                'image': np.random.randn(100, 28, 28, 1),
                'state': np.random.randn(100, 4),
            },
            'actions': np.random.randn(100, 2),
        })

        batch = dataset.sample(32)
        # Batch should have nested shape: {
        # 'observations': {'image': (32, 28, 28, 1), 'state': (32, 4)},
        # 'actions': (32, 2)
        # }
    """

    @classmethod
    def create(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        next_observations: np.ndarray,
        **extra_fields
    ):
        data = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "next_observations": next_observations,
            **extra_fields,
        }
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._data)

    def sample(self, batch_size: int, indx=None):
        """
        Sample a batch of data from the dataset. Use `indx` to specify a specific
        set of indices to retrieve. Otherwise, a random sample will be drawn.

        Returns a dictionary with the same structure as the original dataset.
        """
        if indx is None:
            indx = np.random.randint(self.size, size=batch_size)
        return self.get_subset(indx)

    def get_subset(self, indx):
        subset = {key: value[indx] for key, value in self._data.items()}
        return subset

@dataclasses.dataclass
class GCSDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    p_samegoal: float = 0.5
    intent_sametraj: bool = False
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    terminal: bool = True
    max_distance: int = None
    curr_goal_shift: int = 0

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size - self.curr_goal_shift, size=batch_size)

        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        if self.max_distance is not None:
            final_state_indx = np.clip(final_state_indx, 0, indx + self.max_distance)

        distance = np.random.rand(batch_size)
        middle_goal_indx = np.round(((indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)

        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size - 1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)

        if self.intent_sametraj:
            desired_goal_indx = self.sample_goals(indx, p_randomgoal=0.0, p_trajgoal=1.0 - self.p_currgoal, p_currgoal=self.p_currgoal)
        else:
            desired_goal_indx = self.sample_goals(indx)

        goal_indx = self.sample_goals(indx)
        goal_indx = np.where(np.random.rand(batch_size) < self.p_samegoal, desired_goal_indx, goal_indx)

        success = (indx == goal_indx)
        desired_success = (indx == desired_goal_indx)

        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        batch['desired_rewards'] = desired_success.astype(float) * self.reward_scale + self.reward_shift

        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
            batch['desired_masks'] = (1.0 - desired_success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)
            batch['desired_masks'] = np.ones(batch_size)

        goal_indx = np.clip(goal_indx + self.curr_goal_shift, 0, self.dataset.size - 1)
        desired_goal_indx = np.clip(desired_goal_indx + self.curr_goal_shift, 0, self.dataset.size - 1)
        batch['goals'] = {k: v[goal_indx] for k, v in self.dataset['observations'].items()}
        batch['desired_goals'] = {k: v[desired_goal_indx] for k, v in self.dataset['observations'].items()}

        return batch