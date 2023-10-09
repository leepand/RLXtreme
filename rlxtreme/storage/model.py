"""
Model storage
"""
from abc import abstractmethod
import pickle


class ModelStorage(object):
    """The object to store the model."""

    @abstractmethod
    def get_model(self):
        """Get model"""
        pass

    @abstractmethod
    def save_model(self):
        """Save model"""
        pass


class MemoryModelStorage(ModelStorage):
    """Store the model in memory."""

    def __init__(self):
        self._model = None

    def get_model(self):
        return self._model

    def save_model(self, model):
        self._model = model


class RliteModelStorage(ModelStorage):
    """Store the model in rlite db."""

    def __init__(self, model_db=None, model=None):
        self._model_db = model_db
        self.model = model

    def get_model(self, model_id):
        model = self._model_db.get(model_id)
        if model is None:
            return self.model
        else:
            return pickle.loads(model)

    def save_model(self, model_id, model):
        self._model_db.set(model_id, pickle.dumps(model))
