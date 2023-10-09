"""
Action storage
"""
from abc import abstractmethod
from copy import deepcopy

import six
import pickle


class Action(object):
    r"""The action object

    Parameters
    ----------
    action_id: int
        The index of this action.
    """

    def __init__(self, action_id=None, action_type=None, action_text=None):
        self.id = action_id
        self.type = action_type
        self.text = action_text


class ActionStorage(object):
    """The object to store the actions."""

    @abstractmethod
    def get(self, action_id):
        r"""Get action by action id

        Parameters
        ----------
        action_id: int
            The id of the action.

        Returns
        -------
        action: Action object
            The Action object that has id action_id.
        """
        pass

    @abstractmethod
    def add(self, action):
        r"""Add action

        Parameters
        ----------
        action: Action object
            The Action object to add.

        Raises
        ------
        KeyError

        Returns
        -------
        new_action_ids: list of int
            The Action ids of the added Actions.
        """
        pass

    @abstractmethod
    def update(self, action):
        r"""Add action

        Parameters
        ----------
        action: Action object
            The Action object to update.

        Raises
        ------
        KeyError
        """
        pass

    @abstractmethod
    def remove(self, action_id):
        r"""Add action

        Parameters
        ----------
        action_id: int
            The Action id to remove.

        Raises
        ------
        KeyError
        """
        pass

    @abstractmethod
    def count(self):
        r"""Count actions"""
        pass

    @abstractmethod
    def iterids(self):
        r"""Return iterable of the Action ids.

        Returns
        -------
        action_ids: iterable
            Action ids.
        """


class MemoryActionStorage(ActionStorage):
    """The object to store the actions using memory."""

    def __init__(self):
        self._actions = {}
        self._next_action_id = 0

    def get(self, action_id):
        r"""Get action by action id

        Parameters
        ----------
        action_id: int
            The id of the action.

        Returns
        -------
        action: Action object
            The Action object that has id action_id.
        """
        return deepcopy(self._actions[action_id])

    def add(self, actions):
        r"""Add actions

        Parameters
        ----------
        action: list of Action objects
            The list of Action objects to add.

        Raises
        ------
        KeyError

        Returns
        -------
        new_action_ids: list of int
            The Action ids of the added Actions.
        """
        new_action_ids = []
        for action in actions:
            if action.id is None:
                action.id = self._next_action_id
                self._next_action_id += 1
            elif action.id in self._actions:
                raise KeyError("Action id {} exists".format(action.id))
            else:
                self._next_action_id = max(self._next_action_id, action.id + 1)
            self._actions[action.id] = action
            new_action_ids.append(action.id)
        return new_action_ids

    def update(self, action):
        r"""Update action

        Parameters
        ----------
        action: Action object
            The Action object to update.

        Raises
        ------
        KeyError
        """
        self._actions[action.id] = action

    def remove(self, action_id):
        r"""Remove action

        Parameters
        ----------
        action_id: int
            The Action id to remove.

        Raises
        ------
        KeyError
        """
        del self._actions[action_id]

    def count(self):
        r"""Count actions

        Returns
        -------
        count: int
            Number of Action in the storage.
        """
        return len(self._actions)

    def iterids(self):
        r"""Return iterable of the Action ids.

        Returns
        -------
        action_ids: iterable
            Action ids.
        """
        return six.viewkeys(self._actions)

    def __iter__(self):
        return iter(six.viewvalues(self._actions))


class RliteActionStorage(ActionStorage):
    """The object to store the actions using Rlite db."""

    def __init__(self, action_db=None):
        self._actions = {}
        self._next_action_id = 0
        self._action_db = action_db

    def get(self, action_id, model_id=None):
        r"""Get action by action id

        Parameters
        ----------
        action_id: int
            The id of the action.

        Returns
        -------
        action: Action object
            The Action object that has id action_id.
        """
        _actions = self._action_db.get(model_id)
        if _actions is None:
            return deepcopy(self._actions[action_id])
        else:
            actions = pickle.loads(_actions)
            return actions[action_id]

    def add(self, actions, model_id):
        r"""Add actions

        Parameters
        ----------
        action: list of Action objects
            The list of Action objects to add.

        Raises
        ------
        KeyError

        Returns
        -------
        new_action_ids: list of int
            The Action ids of the added Actions.
        """
        new_action_ids = []
        _actions = self._action_db.get(model_id)
        if _actions is None:
            exsist_actions = {}
        else:
            exsist_actions = pickle.loads(_actions)
        for action in actions:
            if action.id is None:
                action.id = self._next_action_id
                self._next_action_id += 1
            elif action.id in exsist_actions:
                raise KeyError("Action id {} exists".format(action.id))
            else:
                self._next_action_id = max(self._next_action_id, action.id + 1)
            exsist_actions[action.id] = action
            new_action_ids.append(action.id)

        self._action_db.set(model_id, pickle.dumps(exsist_actions))
        return new_action_ids

    def update(self, action, model_id):
        r"""Update action

        Parameters
        ----------
        action: Action object
            The Action object to update.

        Raises
        ------
        KeyError
        """
        _actions = self._action_db.get(model_id)
        if _actions is None:
            exsist_actions = {}
        else:
            exsist_actions = pickle.loads(_actions)
        exsist_actions[action.id] = action
        self._action_db.set(model_id, pickle.dumps(exsist_actions))

    def remove(self, action_id, model_id):
        r"""Remove action

        Parameters
        ----------
        action_id: int
            The Action id to remove.

        Raises
        ------
        KeyError
        """
        _actions = self._action_db.get(model_id)
        if _actions is None:
            exsist_actions = {}
        else:
            exsist_actions = pickle.loads(_actions)

        del exsist_actions[action_id]

        self._action_db.set(model_id, pickle.dumps(exsist_actions))

    def count(self, model_id):
        r"""Count actions

        Returns
        -------
        count: int
            Number of Action in the storage.
        """
        _actions = self._action_db.get(model_id)
        if _actions is None:
            exsist_actions = {}
        else:
            exsist_actions = pickle.loads(_actions)
        return len(exsist_actions)

    def iterids(self, model_id):
        r"""Return iterable of the Action ids.

        Returns
        -------
        action_ids: iterable
            Action ids.
        """
        _actions = self._action_db.get(model_id)
        if _actions is None:
            exsist_actions = {}
        else:
            exsist_actions = pickle.loads(_actions)
        return six.viewkeys(exsist_actions)

    def __iter__(self, model_id):
        _actions = self._action_db.get(model_id)
        if _actions is None:
            exsist_actions = {}
        else:
            exsist_actions = pickle.loads(_actions)
        return iter(six.viewvalues(exsist_actions))
