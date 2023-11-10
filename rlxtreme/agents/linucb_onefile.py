import numpy as np
import pickle
import random


def argmax_rand(arr):
    """Pick max value, break ties randomly."""
    assert isinstance(arr, dict)

    # 将字典的值转换为 NumPy 数组
    values = np.array(list(arr.values()))

    # 检查数组维度
    assert len(values.shape) == 1

    # 找到最大值并随机选择索引
    max_value = np.max(values)
    max_indices = np.random.choice(np.flatnonzero(values == max_value))

    # 返回随机选择的索引
    return max_indices


class LinUCBAgent(object):
    def __init__(self, actions=[0, 1], alpha=0.5, context_size=4, model_db=None):
        self._model_db = model_db
        self.actions = actions
        self.context_size = context_size
        self.n_actions = len(self.actions)
        self.alpha = alpha
        self._init_model()

    def _init_model(self):
        model = {}
        A = [np.identity(self.context_size) for a in range(self.n_actions)]
        b = [np.zeros((self.context_size, 1)) for a in range(self.n_actions)]
        model["A"] = A
        model["b"] = b
        self.model = model

    def get_ucbs(self, obs_dict, not_allowed, model_id):
        allowed_actions = self._get_valid_actions(forbidden_actions=not_allowed)
        n_actions = len(allowed_actions)
        ucbs = np.zeros(n_actions)
        model = self.get_model(model_id=model_id)
        A = model["A"]
        b = model["b"]

        for action_idx_new, action in enumerate(allowed_actions):
            if isinstance(obs_dict, list):
                x_a = obs_dict
            else:
                x_a = obs_dict[action]

            action_idx = self.actions.index(action)

            A_inv = np.linalg.inv(A[action_idx])
            theta_a = np.dot(A_inv, b[action_idx])
            ucb = np.dot(theta_a.T, x_a) + self.alpha * np.sqrt(
                np.linalg.multi_dot([x_a.T, A_inv, x_a])
            )
            ucbs[action_idx_new] = ucb[0]
        return ucbs, allowed_actions

    def act(self, obs_dict, model_id, not_allowed=None):  # user vector is used
        ucbs, allowed_actions = self.get_ucbs(obs_dict, not_allowed, model_id)
        max_value = max(ucbs)
        max_indices = [i for i, value in enumerate(ucbs) if value == max_value]

        if len(max_indices) == 1:
            max_index = max_indices[0]
        else:
            max_index = random.choice(max_indices)

        return allowed_actions[max_index]

    def learn(self, obs, reward, action, model_id):
        model = self.get_model(model_id=model_id)
        A = model["A"]
        b = model["b"]
        action_idx = self.actions.index(action)
        x = np.atleast_2d(obs)
        A[action_idx] += np.dot(x.T, x)
        b[action_idx] += reward * np.atleast_2d(x).T
        model["A"] = A
        model["b"] = b

        self.save_model(model=model, model_id=model_id)

    def get_model(self, model_id):
        _model = self._model_db.get(model_id)
        if _model is None:
            model = self.model
        else:
            model = pickle.loads(_model)

        return model

    def save_model(self, model, model_id):
        self._model_db.set(model_id, pickle.dumps(model))

    def _get_valid_actions(self, forbidden_actions, all_actions=None):
        """
        Given a set of forbidden action IDs, return a set of valid action IDs.

        Parameters
        ----------
        forbidden_actions: Optional[Set[ActionId]]
            The set of forbidden action IDs.

        Returns
        -------
        valid_actions: Set[ActionId]
            The list of valid (i.e. not forbidden) action IDs.
        """
        if all_actions is None:
            all_actions = self.actions
        if forbidden_actions is None:
            forbidden_actions = set()
        else:
            forbidden_actions = set(forbidden_actions)

        if not all(a in all_actions for a in forbidden_actions):
            raise ValueError("forbidden_actions contains invalid action IDs.")
        valid_actions = set(all_actions) - forbidden_actions
        if len(valid_actions) == 0:
            raise ValueError(
                "All actions are forbidden. You must allow at least 1 action."
            )

        valid_actions = list(valid_actions)
        return valid_actions
