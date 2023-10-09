""" Thompson Sampling with Linear Payoff
In This module contains a class that implements Thompson Sampling with Linear
Payoff. Thompson Sampling with linear payoff is a contexutal multi-armed bandit
algorithm which assume the underlying relationship between rewards and contexts
is linear. The sampling method is used to balance the exploration and
exploitation. Please check the reference for more details.
"""
import logging
import six
from six.moves import zip

import numpy as np

from .base import Base
from ..utils import get_random_state

LOGGER = logging.getLogger(__name__)


class LinTS(Base):
    r"""Thompson sampling with linear payoff.

    Parameters
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.

    model_storage : ModelStorage object
        The ModelStorage object to store model parameters.

    action_storage : ActionStorage object
        The ActionStorage object to store actions.

    recommendation_cls : class (default: None)
        The class used to initiate the recommendations. If None, then use
        default Recommendation class.

    delta: float, 0 < delta < 1
        With probability 1 - delta, LinThompSamp satisfies the theoretical
        regret bound.

    R: float, R >= 0
        Assume that the residual  :math:`ri(t) - bi(t)^T \hat{\mu}`
        is R-sub-gaussian. In this case, R^2 represents the variance for
        residuals of the linear model :math:`bi(t)^T`.

    epsilon: float, 0 < epsilon < 1
        A  parameter  used  by  the  Thompson Sampling algorithm.
        If the total trials T is known, we can choose epsilon = 1/ln(T).

    random_state: {int, np.random.RandomState} (default: None)
        If int, np.random.RandomState will used it as seed. If None, a random
        seed will be used.

    References
    ----------
    .. [1]  Shipra Agrawal, and Navin Goyal. "Thompson Sampling for Contextual
            Bandits with Linear Payoffs." Advances in Neural Information
            Processing Systems 24. 2011.
    """

    def __init__(
        self,
        his_db=None,
        model_db=None,
        model=None,
        action_db=None,
        recommendation_cls=None,
        db_type="rlite",
        action_db_type="memory",
        context_dimension=128,
        delta=0.5,
        R=0.01,
        epsilon=0.2,
        random_state=None,
    ):
        super(LinTS, self).__init__(
            his_db,
            model_db,
            model,
            action_db,
            recommendation_cls,
            db_type,
            action_db_type,
        )
        self.random_state = get_random_state(random_state)
        self.context_dimension = context_dimension

        # 0 < delta < 1
        if not isinstance(delta, float):
            raise ValueError("delta should be float")
        elif (delta < 0) or (delta >= 1):
            raise ValueError("delta should be in (0, 1]")
        else:
            self.delta = delta

        # R > 0
        if not isinstance(R, float):
            raise ValueError("R should be float")
        elif R <= 0:
            raise ValueError("R should be positive")
        else:
            self.R = R  # pylint: disable=invalid-name

        # 0 < epsilon < 1
        if not isinstance(epsilon, float):
            raise ValueError("epsilon should be float")
        elif (epsilon < 0) or (epsilon > 1):
            raise ValueError("epsilon should be in (0, 1)")
        else:
            self.epsilon = epsilon

        # model initialization
        B = np.identity(self.context_dimension)  # pylint: disable=invalid-name
        mu_hat = np.zeros(shape=(self.context_dimension, 1))
        f = np.zeros(shape=(self.context_dimension, 1))
        if self.db_type == "rlite":
            self.model = {"B": B, "mu_hat": mu_hat, "f": f}
        else:
            self._model_storage.save_model({"B": B, "mu_hat": mu_hat, "f": f})

    def _linthompsamp_score(self, context, model_id):
        """Thompson Sampling"""
        action_ids = list(six.viewkeys(context))
        context_array = np.asarray([context[action_id] for action_id in action_ids])
        if self.db_type == "rlite":
            model = self._model_storage.get_model(model_id=model_id)
            if model is None:
                model = self.model
        else:
            model = self._model_storage.get_model()

        B = model["B"]  # pylint: disable=invalid-name
        mu_hat = model["mu_hat"]
        v = self.R * np.sqrt(
            24 / self.epsilon * self.context_dimension * np.log(1 / self.delta)
        )
        mu_tilde = self.random_state.multivariate_normal(
            mu_hat.flat, v**2 * np.linalg.inv(B)
        )[..., np.newaxis]
        estimated_reward_array = context_array.dot(mu_hat)
        score_array = context_array.dot(mu_tilde)

        estimated_reward_dict = {}
        uncertainty_dict = {}
        score_dict = {}
        for action_id, estimated_reward, score in zip(
            action_ids, estimated_reward_array, score_array
        ):
            estimated_reward_dict[action_id] = float(estimated_reward)
            score_dict[action_id] = float(score)
            uncertainty_dict[action_id] = float(score - estimated_reward)
        return estimated_reward_dict, uncertainty_dict, score_dict

    def get_action(self, context, n_actions=None, his_id=None, model_id=None):
        """Return the action to perform

        Parameters
        ----------
        context : dictionary
            Contexts {action_id: context} of different actions.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        if self.db_type == "rlite":
            action_ids_cnt = self._action_storage.count(model_id=model_id)
        else:
            action_ids_cnt = self._action_storage.count()
        if action_ids_cnt == 0:

            return self._get_action_with_empty_action_storage(
                context, n_actions, history_id=his_id, model_id=model_id
            )

        if not isinstance(context, dict):
            raise ValueError("LinThompSamp requires context dict for all actions!")
        if n_actions == -1:
            n_actions = action_ids_cnt

        estimated_reward, uncertainty, score = self._linthompsamp_score(
            context, model_id=model_id
        )

        if n_actions is None:
            recommendation_id = max(score, key=score.get)
            if self.act_db_type == "memory":
                action = self._action_storage.get(recommendation_id)
            else:
                action = self._action_storage.get(recommendation_id, model_id=model_id)
            recommendations = self._recommendation_cls(
                action=action,
                estimated_reward=estimated_reward[recommendation_id],
                uncertainty=uncertainty[recommendation_id],
                score=score[recommendation_id],
            )
        else:
            recommendation_ids = sorted(score, key=score.get, reverse=True)[:n_actions]
            recommendations = []  # pylint: disable=redefined-variable-type
            for action_id in recommendation_ids:
                if self.act_db_type == "memory":
                    action = self._action_storage.get(action_id)
                else:
                    action = self._action_storage.get(action_id, model_id=model_id)

                recommendations.append(
                    self._recommendation_cls(
                        action=action,
                        estimated_reward=estimated_reward[action_id],
                        uncertainty=uncertainty[action_id],
                        score=score[action_id],
                    )
                )

        if self.db_type == "rlite":
            history_id = self._history_storage.add_history(
                history_id=his_id,
                model_id=model_id,
                context=context,
                recoms=recommendations,
            )
        else:
            history_id = self._history_storage.add_history(context, recommendations)
        return history_id, recommendations

    def reward(self, history_id, rewards, model_id=None):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        if self.db_type == "rlite":
            context = self._history_storage.get_unrewarded_history(
                history_id, model_id=model_id
            ).context
            # Update the model
            model = self._model_storage.get_model(model_id=model_id)
        else:
            context = self._history_storage.get_unrewarded_history(history_id).context

            # Update the model
            model = self._model_storage.get_model()

        B = model["B"]  # pylint: disable=invalid-name
        f = model["f"]

        for action_id, reward in six.viewitems(rewards):
            context_t = np.reshape(context[action_id], (-1, 1))
            B += context_t.dot(context_t.T)  # pylint: disable=invalid-name
            f += reward * context_t
            mu_hat = np.linalg.inv(B).dot(f)

        if self.db_type == "rlite":
            self._model_storage.save_model(
                model_id=model_id, model={"B": B, "mu_hat": mu_hat, "f": f}
            )
            # Update the history
            self._history_storage.add_reward(history_id, model_id, rewards)

        else:
            self._model_storage.save_model({"B": B, "mu_hat": mu_hat, "f": f})
            # Update the history
            self._history_storage.add_reward(history_id, rewards)

    def add_action(self, actions, model_id=None):
        """Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action oBjects for recommendation
        """
        if self.act_db_type == "memory":
            new_action_ids = self._action_storage.add(actions)
        else:
            new_action_ids = self._action_storage.add(actions, model_id=model_id)

        return new_action_ids

    def remove_action(self, action_id, model_id=None):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
        """
        if self.act_db_type == "rlite":
            self._action_storage.remove(action_id, model_id=model_id)
        else:
            self._action_storage.remove(action_id)
