import numpy as np
from ..algorithms.net import PolicyNetwork


"""

This file implements the standard vanilla REINFORCE algorithm, also
known as Monte Carlo Policy Gradient.

The main neural network logic is contained in the PolicyNetwork class,
with more algorithm specific code, including action taking and loss
computing contained in the REINFORCE class.  (NOTE: this only supports discrete actions)


    Resources:
        Sutton and Barto: http://incompleteideas.net/book/the-book-2nd.html
        Karpathy blog: http://karpathy.github.io/2016/05/31/rl/


    Glossary:
        (w.r.t.) = with respect to (as in taking gradient with respect to a variable)
        (h or logits) = numerical policy preferences, or unnormalized probailities of actions
"""


class REINFORCE:
    """
    Object to handle running the algorithm. Uses a PolicyNetwork
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

        self.gamma = gamma
        self.input_dim = ac_n
        self.hidden_dim = hidden_dim

        # RL specific bookkeeping
        self.saved_action_gradients = []
        self.rewards = []
        self.policy = PolicyNetwork(
            ob_n=ob_n,
            ac_n=ac_n,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            model_db=model_db,
            his_db=his_db,
            dtype=dtype,
        )

    def act(self, obs, model_id):
        """
        Pass observations through network and sample an action to take. Keep track
        of dh to use to update weights
        """
        obs = np.reshape(obs, [1, -1])
        netout, affine1, relu1, affine2 = self.policy.forward(
            obs, model_id=model_id
        )  # [0]

        probs = netout[0]
        # randomly sample action based on probabilities
        action = np.random.choice(self.policy.ac_n, p=probs)
        # derivative that pulls in direction to make actions taken more probable
        # this will be fed backwards later
        # (see README.md for derivation)
        dh = -1 * probs
        dh[action] += 1
        # self.saved_action_gradients.append(dh)

        return action, affine1, relu1, affine2, dh

    def learn(self, rewards, model_id):
        """
        At the end of the episode, calculate the discounted return for
        each time step and update the model parameters
        """
        self.policy.backward(dout="test", rewards=rewards, model_id=model_id)
