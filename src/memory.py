import numpy as np
import torch


class Trajectory:
    """
    A trajectory is a list of (s, a, r, s') tuples, that represents an
    agent's transition from state s to state s', by taking action a and
    observing reward r
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.current_timestep = 0

    def add_timestep(self, state, action, action_probs, reward, next_state):
        """
        Add the given (s, a, r, s') tuple to the trajectory
        """
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_probs)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.current_timestep += 1

    def get_states(self, as_torch=True):
        """
        Return the list of states in the current trajectory
        either as a Numpy array or as a PyTorch tensor
        """
        return np.array(self.states) if not as_torch else torch.tensor(self.states)

    def get_actions(self, as_torch=True):
        """
        Return the list of actions in the current trajectory
        either as a Numpy array or as a PyTorch tensor
        """
        return np.array(self.actions) if not as_torch else torch.tensor(self.actions)

    def get_action_probs(self, as_torch=True):
        """
        Return the list of action probabilities in the current trajectory
        either as a Numpy array or as a PyTorch tensor
        """
        return (
            np.array(self.action_probs)
            if not as_torch
            else torch.tensor(self.action_probs)
        )

    def get_rewards(self, as_torch=True):
        """
        Return the list of rewards in the current trajectory
        either as a Numpy array or as a PyTorch tensor
        """
        return np.array(self.rewards) if not as_torch else torch.tensor(self.rewards)

    def get_next_states(self, as_torch=True):
        """
        Return the list of next states in the current trajectory
        either as a Numpy array or as a PyTorch tensor
        """
        return (
            np.array(self.next_states)
            if not as_torch
            else torch.tensor(self.next_states)
        )

    def get_returns(self, max_timestep=None, discount=1, to_go=False, as_torch=True):
        """
        Compute returns of the current trajectory. You can compute the following return types:
        - Finite-horizon undiscounted return: set `max_timestep=t` and `discount=1`
        - Infinite-horizon discounted return: set `max_timestep=None` and `discount=d`, with d in (0, 1)
        """
        assert discount > 0 and discount <= 1, "Discount should be in (0, 1]"
        if max_timestep is None:
            max_timestep = self.current_timestep
        max_timestep = np.clip(max_timestep, 0, self.current_timestep)
        discount_per_timestep = discount ** np.arange(max_timestep)
        returns_per_timestep = np.cumsum(
            np.array(self.rewards)[::-1] * discount_per_timestep[::-1]
        )[::-1]
        returns = returns_per_timestep[0] if not to_go else returns_per_timestep
        return returns if not as_torch else torch.tensor(returns.copy())

    def __getitem__(self, t):
        """
        Return the (s, a, r, s') tuple at time-step t
        """
        return (
            self.states[t],
            self.actions[t],
            self.action_probs[t],
            self.rewards[t],
            self.next_states[t],
        )

    def __len__(self):
        """
        Return the lenght of the trajectory
        """
        return self.current_timestep


class TrajectoryPool:
    """
    A trajectory pool is a set of trajectories
    """

    def __init__(self, n=0):
        self.trajectories = [Trajectory() for _ in range(n)]

    def add(self, trajectory):
        """
        Add the given trajectory to the pool
        """
        assert isinstance(
            trajectory, Trajectory
        ), "The given trajectory should be an instance of the Trajectory class"
        self.trajectories.append(trajectory)

    def extend(self, trajectory_pool):
        """
        Extend the current trajectory pool with the given one
        """
        assert isinstance(
            trajectory_pool, TrajectoryPool
        ), "The given trajectory pool should be an instance of the TrajectoryPool class"
        for trajectory in trajectory_pool:
            self.add(trajectory)

    def add_to_trajectory(self, i, state, action, action_probs, reward, next_state):
        """
        Add a (s, a, r, s') tuple to the i-th trajectory in the pool
        """
        self.trajectories[i].add_timestep(
            state, action, action_probs, reward, next_state
        )

    def tensorify(
        self, max_steps, state_shape, num_actions, discount=1, ignore_index=-100
    ):
        """
        Convert the current pool of trajectories to
        a set of PyTorch tensors
        """
        assert isinstance(
            state_shape, tuple
        ), "The given state shape should be a tuple of dimensions"

        # Initialize tensors
        states = torch.full(
            (len(self.trajectories), max_steps, *state_shape),
            ignore_index,
            dtype=torch.float32,
        )
        actions = torch.full(
            (len(self.trajectories), max_steps), ignore_index, dtype=torch.int64
        )
        action_probs = torch.full(
            (len(self.trajectories), max_steps, num_actions),
            ignore_index,
            dtype=torch.float32,
        )
        returns = torch.full_like(actions, ignore_index, dtype=torch.float32)
        next_states = torch.full_like(states, ignore_index, dtype=torch.float32)

        # Update tensors for each trajectory
        for i, trajectory in enumerate(self.trajectories):
            t_states = trajectory.get_states(as_torch=True)
            t_actions = trajectory.get_actions(as_torch=True)
            t_action_probs = trajectory.get_action_probs(as_torch=True)
            t_returns = trajectory.get_returns(
                discount=discount, to_go=True, as_torch=True
            )
            t_next_states = trajectory.get_next_states(as_torch=True)
            states[i, : t_states.shape[0]] = t_states
            actions[i, : t_actions.shape[0]] = t_actions
            action_probs[i, : t_action_probs.shape[0]] = t_action_probs
            returns[i, : t_returns.shape[0]] = t_returns
            next_states[i, : t_next_states.shape[0]] = t_next_states

        return states, actions, action_probs, returns, next_states

    def __getitem__(self, i):
        """
        Return the i-th trajectory in the pool
        """
        return self.trajectories[i]

    def __len__(self):
        """
        Return how many trajectories we have in the pool
        """
        return len(self.trajectories)
