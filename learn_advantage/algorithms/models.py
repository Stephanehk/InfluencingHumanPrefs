import numpy as np
import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardFunctionPR(torch.nn.Module):
    """
    The partial return reward learning model
    """

    def __init__(self, gamma, n_features=6):
        super().__init__()
        self.n_features = n_features
        self.gamma = gamma
        self.linear1 = torch.nn.Linear(self.n_features, 1, bias=False)
        self.act = torch.nn.LogSigmoid()

    def forward(self, phi):
        pr = torch.squeeze(self.linear1(phi))

        left_pred = torch.sigmoid(torch.subtract(pr[:, 0:1], pr[:, 1:2]))
        right_pred = torch.sigmoid(torch.subtract(pr[:, 1:2], pr[:, 0:1]))
        phi_logit = torch.stack([left_pred, right_pred], axis=1)
        return phi_logit


class RewardFunctionRegret(torch.nn.Module):
    """
    The regret reward learning model
    """

    def __init__(
        self,
        args,
        preference_weights,
        n_features=6,
    ):
        super().__init__()
        self.n_features = n_features
        self.gamma = args.gamma
        succ_feats = np.array(args.succ_feats)
        self.succ_feats = torch.from_numpy(succ_feats).to(device).double()
        
        if args.succ_q_feats is not None:
            self.succ_q_feats = torch.tensor(args.succ_q_feats, dtype=torch.double).to(
                device
            )

        self.include_actions = args.include_actions

        self.linear1 = torch.nn.Linear(self.n_features, 1, bias=False).double()

        self.softmax = torch.nn.Softmax(dim=1)

        # optionally set weights for the deterministic regret model
        if preference_weights is not None:
            self.rw = preference_weights[0]
            self.v_stw = preference_weights[1]
            self.v_s0w = preference_weights[2]
        else:
            self.rw = 1
            self.v_stw = 1
            self.v_s0w = 1

        self.T = 0.001
       

    def get_vals(self, coords):
        """
        Approximate V* using successor feautres
        """

        coords = coords.long()

        selected_succ_feats = torch.stack([self.succ_feats[:, x, y] for x, y in coords])
        del coords

        vs = self.linear1(selected_succ_feats.double())
        del selected_succ_feats

        v_pi_approx = torch.sum(torch.mul(self.softmax(vs / self.T), vs), dim=1)
        v_pi_approx = torch.squeeze(v_pi_approx)
        del vs
        return v_pi_approx

    def forward(self, phi):
        if self.include_actions:
            raise ValueError("Current code release does not support computing per time-step regret")
        else:
            pr = torch.squeeze(self.linear1(phi[:, :, 0 : self.n_features].double()))
            ss_x = torch.squeeze(phi[:, :, self.n_features : self.n_features + 1])
            ss_y = torch.squeeze(phi[:, :, self.n_features + 1 : self.n_features + 2])
            ss_cord_pairs = torch.stack([ss_x, ss_y], dim=1)

            es_x = torch.squeeze(phi[:, :, self.n_features + 2 : self.n_features + 3])
            es_y = torch.squeeze(phi[:, :, self.n_features + 3 : self.n_features + 4])
            es_cord_pairs = torch.stack([es_x, es_y], dim=1)

            # build list of succ fears for start/end states
            v_ss = self.get_vals(ss_cord_pairs)
            v_es = self.get_vals(es_cord_pairs)

            left_pr = pr[:, 0:1]
            right_pr = pr[:, 1:2]

            left_vf_ss = v_ss[:, 0:1]
            right_vf_ss = v_ss[:, 1:2]

            left_vf_es = v_es[:, 0:1]
            right_vf_es = v_es[:, 1:2]

            # apply weights learned from logistic regression (if it exists)
            left_pr = torch.multiply(left_pr, self.rw)
            right_pr = torch.multiply(right_pr, self.rw)

            left_vf_ss = torch.multiply(left_vf_ss, self.v_s0w)
            right_vf_ss = torch.multiply(right_vf_ss, self.v_s0w)

            left_vf_es = torch.multiply(left_vf_es, self.v_stw)
            right_vf_es = torch.multiply(right_vf_es, self.v_stw)

            # calculate change in expected return
            left_delta_v = torch.subtract(left_vf_es, left_vf_ss)
            right_delta_v = torch.subtract(right_vf_es, right_vf_ss)

            left_delta_er = torch.add(left_pr, left_delta_v)
            right_delta_er = torch.add(right_pr, right_delta_v)

        left_pred = torch.sigmoid(torch.subtract(left_delta_er, right_delta_er))
        right_pred = torch.sigmoid(torch.subtract(right_delta_er, left_delta_er))
        phi_logit = torch.stack([left_pred, right_pred], axis=1)
        return phi_logit
    
    def forward_regret(self, phi):
        return self.forward(phi)
    
    def forward_pr(self, phi):
        pr = torch.squeeze(self.linear1(phi))

        left_pred = torch.sigmoid(torch.subtract(pr[:, 0:1], pr[:, 1:2]))
        right_pred = torch.sigmoid(torch.subtract(pr[:, 1:2], pr[:, 0:1]))
        phi_logit = torch.stack([left_pred, right_pred], axis=1)
        return phi_logit 

class FeedForwardNN(torch.nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.layer1 = torch.nn.Linear(in_dim, 64)
		self.layer2 = torch.nn.Linear(64, 64)
		self.layer3 = torch.nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = F.softmax(self.layer3(activation2))

		return output