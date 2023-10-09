"""
rlxtreme interfaces
"""
from abc import abstractmethod

from ..storage import Recom
from ..storage import (
    MemoryActionStorage,
    MemoryHistoryStorage,
    MemoryModelStorage,
    RliteActionStorage,
    RliteHistoryStorage,
    RliteModelStorage,
)


class Base(object):
    r"""rl algorithm

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

    Attributes
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.
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
    ):
        if db_type == "rlite":
            self._history_storage = RliteHistoryStorage(his_db=his_db)
            self._model_storage = RliteModelStorage(model_db=model_db, model=model)
            if action_db_type == "memory":
                self._action_storage = MemoryActionStorage()
            else:
                self._action_storage = RliteActionStorage(action_db=action_db)

        else:
            self._history_storage = MemoryHistoryStorage()
            self._model_storage = MemoryModelStorage()
            if action_db_type == "memory":
                self._action_storage = MemoryActionStorage()
            else:
                self._action_storage = RliteActionStorage(action_db=action_db)

        if recommendation_cls is None:
            self._recommendation_cls = Recom
        else:
            self._recommendation_cls = recommendation_cls

    @property
    def history_storage(self):
        return self._history_storage

    @abstractmethod
    def get_action(self, context, n_actions=None):
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
        pass

    def _get_action_with_empty_action_storage(self, context, n_actions):
        if n_actions is None:
            recommendations = None
        else:
            recommendations = []
        history_id = self._history_storage.add_history(context, recommendations)
        return history_id, recommendations

    @abstractmethod
    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        pass

    @abstractmethod
    def add_action(self, actions):
        """Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation.
        """
        pass

    def update_action(self, action):
        """Update action.

        Parameters
        ----------
        action : Action
            The Action object to update.
        """
        self._action_storage.update(action)

    @abstractmethod
    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
        """
        pass
