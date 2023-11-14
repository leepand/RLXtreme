import numpy as np
import pickle
from .optim import adam


class PolicyNetwork(object):
    """
    Neural network policy. Takes in observations and returns probabilities of
    taking actions.

    ARCHITECTURE:
    {affine - relu } x (L - 1) - affine - softmax

    """

    def __init__(
        self,
        ob_n,
        ac_n,
        hidden_dim=200,
        lr=1e-3,
        gamma=0.99,
        model_db=None,
        his_db=None,
        dtype=np.float32,
    ):
        """
        Initialize a neural network to choose actions

        Inputs:
        - ob_n: Length of observation vector
        - ac_n: Number of possible actions
        - hidden_dims: List of size of hidden layer sizes
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self._model_db = model_db
        self._his_db = his_db
        self.ob_n = ob_n
        self.ac_n = ac_n
        self.hidden_dim = H = hidden_dim
        self.dtype = dtype
        self.lr = lr
        self.gamma = gamma
        self.H = H

        self._init_model()

    def _init_model(self):
        # Initialize all weights (model params) with "Xavier Initialization"
        # Randomly initialize the weights in the policy network.
        # weight matrix init = uniform(-1, 1) / sqrt(layer_input)
        # bias init = zeros()
        self.params = {}
        self.params["W1"] = (-1 + 2 * np.random.rand(self.ob_n, self.H)) / np.sqrt(
            self.ob_n
        )
        self.params["b1"] = np.zeros(self.H)
        self.params["W2"] = (-1 + 2 * np.random.rand(self.H, self.ac_n)) / np.sqrt(
            self.H
        )
        self.params["b2"] = np.zeros(self.ac_n)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

        self.his_data = {}
        # Neural net bookkeeping
        self.cache = {}
        self.grads = {}
        # RL specific bookkeeping
        self.saved_action_gradients = []
        # Configuration for Adam optimization
        self.optimization_config = {"learning_rate": self.lr}
        self.adam_configs = {}
        for p in self.params:
            d = {k: v for k, v in self.optimization_config.items()}
            self.adam_configs[p] = d

        self.model = {
            "params": self.params,
            "adam_configs": self.adam_configs,
            "grads": self.grads,
        }

    ### HELPER FUNCTIONS
    def _zero_grads_db(self, grads={}):
        """Reset gradients to 0. This should be called during optimization steps"""
        for g in grads:
            grads[g] = np.zeros_like(grads[g])
        return grads

    def _add_to_cache_db(self, name, val, cache={}):
        """Helper function to add a parameter to the cache without having to do checks"""
        if name in cache:
            cache[name].append(val)
        else:
            cache[name] = [val]
        return cache

    def _update_grad_db(self, name, val, grads={}):
        """Helper fucntion to set gradient without having to do checks"""
        if name in grads:
            grads[name] += val
        else:
            grads[name] = val
        return grads

    ### HELPER FUNCTIONS
    def _zero_grads(self):
        """Reset gradients to 0. This should be called during optimization steps"""
        for g in self.grads:
            self.grads[g] = np.zeros_like(self.grads[g])

    def _add_to_cache(self, name, val):
        """Helper function to add a parameter to the cache without having to do checks"""
        if name in self.cache:
            self.cache[name].append(val)
        else:
            self.cache[name] = [val]

    def _update_grad(self, name, val):
        """Helper fucntion to set gradient without having to do checks"""
        if name in self.grads:
            self.grads[name] += val
        else:
            self.grads[name] = val

    def _softmax(self, x):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        return probs

    ### MAIN NEURAL NETWORK STUFF
    def forward(self, x, model_id):
        """
        Forward pass observations (x) through network to get probabilities
        of taking each action

        [input] --> affine --> relu --> affine --> softmax/output

        """
        model = self.get_model(model_id=model_id)
        params = model["params"]
        p = params
        W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]

        # forward computations
        affine1 = x.dot(W1) + b1
        relu1 = np.maximum(0, affine1)
        affine2 = relu1.dot(W2) + b2

        logits = affine2  # layer right before softmax (i also call this h)
        # pass through a softmax to get probabilities
        probs = self._softmax(logits)
        return probs, affine1, relu1, affine2

    def cache_forward_data(self, x, affine1, relu1, dh, model_id):
        # cache values for backward (based on what is needed for analytic gradient calc)
        _his_data = self._his_db.get(model_id)
        if _his_data is None:
            his_data = self.his_data
            cache = self.cache
            saved_action_gradients = self.saved_action_gradients
        else:
            his_data = pickle.loads(_his_data)
            cache = his_data["cache"]
            saved_action_gradients = his_data["sa_grads"]

        cache = self._add_to_cache_db("fwd_x", x, cache=cache)
        cache = self._add_to_cache_db("fwd_affine1", affine1, cache=cache)
        cache = self._add_to_cache_db("fwd_relu1", relu1, cache=cache)
        saved_action_gradients.append(dh)
        his_data["cache"] = cache
        his_data["sa_grads"] = saved_action_gradients
        self._his_db.set(model_id, pickle.dumps(his_data))

    def reset_forward_data(self, model_id):
        his_data = self.his_data
        his_data["cache"] = {}
        his_data["sa_grads"] = []
        self._his_db.set(model_id, pickle.dumps(his_data))

    def get_forward_data(self, model_id):
        _his_data = self._his_db.get(model_id)
        if _his_data is None:
            his_data = self.his_data
            cache = self.cache
            saved_action_gradients = self.saved_action_gradients
        else:
            his_data = pickle.loads(_his_data)
            cache = his_data["cache"]
            saved_action_gradients = his_data["sa_grads"]
        return cache, saved_action_gradients

    def get_model(self, model_id):
        _model = self._model_db.get(model_id)
        if _model is None:
            model = self.model
        else:
            model = pickle.loads(_model)
        return model

    def save_model(self, model, model_id):
        self._model_db.set(model_id, pickle.dumps(model))

    def backward(self, dout, rewards, model_id):
        """
        Backwards pass of the network.

        affine <-- relu <-- affine <-- [gradient signal of softmax/output]

        Params:
            dout: gradient signal for backpropagation


        Chain rule the derivatives backward through all network computations
        to compute gradients of output probabilities w.r.t. each network weight.
        (to be used in stochastic gradient descent optimization (adam))
        """
        cache, saved_action_gradients = self.get_forward_data(model_id=model_id)
        model = self.get_model(model_id=model_id)
        params = model["params"]
        grads = model["grads"]
        adam_configs = model["adam_configs"]
        action_gradient = np.array(saved_action_gradients)

        returns = self.calculate_discounted_returns(rewards)
        # Multiply the signal that makes actions taken more probable by the discounted
        # return of that action.  This will pull the weights in the direction that
        # makes *better* actions more probable.
        policy_gradient = np.zeros(action_gradient.shape)
        for t in range(0, len(returns)):
            policy_gradient[t] = action_gradient[t] * returns[t]

        p = params
        W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]

        # get values from network forward passes (for analytic gradient computations)
        fwd_relu1 = np.concatenate(cache["fwd_relu1"])
        fwd_affine1 = np.concatenate(cache["fwd_affine1"])
        fwd_x = np.concatenate(cache["fwd_x"])

        dout = -policy_gradient
        # Analytic gradient of last layer for backprop
        # affine2 = W2*relu1 + b2
        # drelu1 = W2 * dout
        # dW2 = relu1 * dout
        # db2 = dout
        drelu1 = dout.dot(W2.T)
        dW2 = fwd_relu1.T.dot(dout)
        db2 = np.sum(dout, axis=0)

        # gradient of relu (non-negative for values that were above 0 in forward)
        daffine1 = np.where(fwd_affine1 > 0, drelu1, 0)

        # affine1 = W1*x + b1
        # dx
        dW1 = fwd_x.T.dot(daffine1)
        db1 = np.sum(daffine1)

        # update gradients
        grads = self._update_grad_db("W1", dW1, grads=grads)
        grads = self._update_grad_db("b1", db1, grads=grads)
        grads = self._update_grad_db("W2", dW2, grads=grads)
        grads = self._update_grad_db("b2", db2, grads=grads)

        # run an optimization step on all of the model parameters
        for p in params:
            next_w, adam_configs[p] = adam(
                params[p],
                grads[p],
                config=adam_configs[p],
            )
            params[p] = next_w
        grads = self._zero_grads_db(grads=grads)  # required every call to adam

        # reset cache for next backward pass
        self.cache = {}
        model["grads"] = grads
        model["params"] = params
        model["adam_configs"] = adam_configs
        self.save_model(model=model, model_id=model_id)
        self.reset_forward_data(model_id=model_id)

    def calculate_discounted_returns(self, rewards):
        """
        Calculate discounted reward and then normalize it
        (see Sutton book for definition)
        Params:
            rewards: list of rewards for every episode
        """
        returns = np.zeros(len(rewards))

        next_return = 0  # 0 because we start at the last timestep
        for t in reversed(range(0, len(rewards))):
            next_return = rewards[t] + self.gamma * next_return
            returns[t] = next_return
        # normalize for better statistical properties
        returns = (returns - returns.mean()) / (
            returns.std() + np.finfo(np.float32).eps
        )
        return returns
