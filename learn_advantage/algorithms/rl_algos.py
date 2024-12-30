import random

import numpy as np
import torch

from learn_advantage.env.grid_world import GridWorldEnv
from learn_advantage.algorithms.models import FeedForwardNN

# import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_random_reward_vector():
    """
    Generates a random reward vector by sampling without replacement from the set [-1,50,-50,1,-1,-2]

    Output:
    - vector: The randomly generated reward vector
    """
    space = [-1, 50, -50, 1, -1, -2]
    vector = []
    for _ in range(6):
        s = random.choice(space)
        space.remove(s)
        vector.append(s)
    return vector


def learn_successor_feature_iter(pi, Fgamma=0.999, rew_vec=None, env=None):
    """
    Uses value iteration to find the state, action, and (s,a) pair successor features (SFs).

    Output:
    - psi: The state SFs
    - psi_actions: The (s,a) pair SFs
    - psi_Q: The action SFs
    """
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    if isinstance(rew_vec, np.ndarray):
        env.set_custom_reward_function(rew_vec[0:6])

    THETA = 0.001
    # initialize Q
    num_actions = len(GridWorldEnv.actions)
    psi = [
        [np.zeros(env.feature_size) for i in range(env.width)]
        for j in range(env.height)
    ]
    psi_actions = [
        [np.zeros(num_actions * env.width * env.height) for i in range(env.width)]
        for j in range(env.height)
    ]
    psi_Q = [
        [
            [np.zeros(env.feature_size) for a in range(num_actions)]
            for i in range(env.width)
        ]
        for j in range(env.height)
    ]
    # iterativley learn state value
    while True:
        delta = 0
        new_psi = np.copy(psi)

        for i, j in env.positions():
            if env.is_blocked(i, j):
                continue
            # total = 0

            state_psi = []
            action_psi = []
            for trans in pi[(i, j)]:
                prob, a_index = trans
                next_state, _, done, phi = env.get_next_state((i, j), a_index)

                action_phi = np.zeros((env.height, env.width, num_actions))
                action_phi[i][j][a_index] = 1
                action_phi = np.ravel(action_phi)

                ni, nj = next_state
                if not done:
                    psi_sas = prob * (phi + Fgamma * psi[ni][nj])
                    psi_q = phi + Fgamma * psi[ni][nj]
                    action_feat = prob * (action_phi + Fgamma * psi_actions[ni][nj])
                else:
                    psi_sas = np.zeros(env.feature_size)
                    psi_q = np.zeros(env.feature_size)
                    action_feat = np.zeros(num_actions * env.width * env.height)

                psi_Q[i][j][a_index] = psi_q
                state_psi.append(psi_sas)
                action_psi.append(action_feat)

            psi_actions[i][j] = sum(action_psi)
            new_psi[i][j] = sum(state_psi)
            delta = max(delta, np.sum(np.abs(psi[i][j] - new_psi[i][j])))

        psi = new_psi

        if delta < THETA:
            break
    return psi, np.array(psi_actions), psi_Q


def build_reward_from_nn_feats(model, env):
    height = env.height
    width = env.width

    pred_OAF = np.zeros((height, width, len(env.actions)))

    for i, j in env.positions():
        with torch.no_grad():
            for a_i in range(len(env.actions)):
                state_embedding = torch.zeros((height, width))
                state_embedding[i][j] = 1

                action_embedding = torch.zeros(len(env.actions))
                action_embedding[a_i] = 1

                sa_embedding = torch.cat((state_embedding.flatten(), action_embedding))

                pred_OAF[i][j][a_i] = model.get_trans_val(
                    sa_embedding.to(device).float()
                ).cpu()
    return pred_OAF


def build_pi_from_nn_feats(model, env):
    """
    Given a neural network model (currently only supports instances of models.RewardFunctionPRGen), create a policy
    by acting greedily over the networks predicted rewards.

    Input:
    - model: The neural network model, must be an instance of models.RewardFunctionPRGen
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - pi: the policy, represented as a dictionary where pi(s) = [p(a1|s), p(a2|s), ...]
    """
    pi = {}

    height = env.height
    width = env.width
    for i, j in env.positions():
        with torch.no_grad():
            weights = []
            for a_i in range(len(env.actions)):
                state_embedding = torch.zeros((height, width))
                state_embedding[i][j] = 1

                action_embedding = torch.zeros(len(env.actions))
                action_embedding[a_i] = 1

                sa_embedding = torch.cat((state_embedding.flatten(), action_embedding))
                weights.append(
                    model.get_trans_val(sa_embedding.to(device).float()).cpu()
                )

            max_weight = np.max(weights)
            count = weights.count(max_weight)
            pi[(i, j)] = [
                (1 / count if weights[a_index] == max_weight else 0, a_index)
                for a_index in range(len(env.actions))
            ]

    return pi


def build_pi_from_feats(s_a_weights, env):
    """
    Given learned (s,a) pair weights, create a policy by acting greedily over the predicted weights.

    Input:
    - s_a_weights: The (s,a) pair weghts represented as a 1d vector
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - pi: the policy, represented as a dictionary where pi(s) = [p(a1|s), p(a2|s), ...]
    """
    pi = {}

    height = env.height
    width = env.width

    s_a_weights = s_a_weights.reshape((height, width, len(env.actions)))

    for i, j in env.positions():
        max_weight = max(s_a_weights[i][j])
        count = s_a_weights[i][j].tolist().count(max_weight)
        pi[(i, j)] = [
            (1 / count if s_a_weights[i][j][a_index] == max_weight else 0, a_index)
            for a_index in range(len(env.actions))
        ]

    return pi


def build_pi(Q, env):
    """
    Given a learned Q function, create a policy by acting greedily over the Q values.

    Input:
    - s_a_weights: The Q function
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP


    Output:
    - pi: the policy, represented as a dictionary where pi(s) = [p(a1|s), p(a2|s), ...]
    """
    pi = {}
    num_actions = len(GridWorldEnv.actions)

    for i, j in env.positions():
        V = max(Q[i][j])
        V_count = Q[i][j].tolist().count(V)
        pi[(i, j)] = [
            (1 / V_count if Q[i][j][a_index] == V else 0, a_index)
            for a_index in range(num_actions)
        ]
    return pi


def build_random_policy(env):
    """
    Generated a policy that uniform-randomly selects actions at each state.

    Input:
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - pi: the policy, represented as a dictionary where pi(s) = [p(a1|s), p(a2|s), ...]
    """
    pi = {}
    num_actions = len(GridWorldEnv.actions)

    for i, j in env.positions():
        pi[(i, j)] = [
            (1.0 / float(num_actions), a_index) for a_index in range(num_actions)
        ]
    return pi


def iterative_policy_evaluation(
    pi, *, rew_vec=None, set_rand_rew=False, gamma=0.999, env=None
):
    """
    Performs iterative policy evaluation.

    Input:
    - pi: The policy, represented as a dictionary whenre pi(s) = [p(a1|s), p(a2|s), ...]
    - rew_vec: The reward vector to evaluate the policy with. If none, the default reward vector is used.
    - set_rand_rew: If true, evaluate pi with a random reward vector
    - gamma: The discount factor
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - V: the value function
    """
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")

    if isinstance(rew_vec, np.ndarray):
        env.set_custom_reward_function(rew_vec[0:6])
    elif set_rand_rew:
        env.set_custom_reward_function(rew_vec)

    THETA = 0.001
    V = np.zeros((env.height, env.width))

    # iterativley learn state value
    while True:
        delta = 0
        new_V = V.copy()
        for i, j in env.positions():
            if env.is_blocked(i, j):
                continue
            # total = 0
            state_qs = []
            for trans in pi[(i, j)]:
                prob, a_index = trans
                next_state, reward, done, _ = env.get_next_state((i, j), a_index)
                ni, nj = next_state
                if not done:
                    state_qs.append(prob * (reward + gamma * V[ni][nj]))
                else:
                    state_qs.append(prob * reward)
            new_V[i][j] = sum(state_qs)
            delta = max(delta, np.abs(V[i][j] - new_V[i][j]))

        V = new_V
        if delta < THETA:
            break

    return V


def value_iteration(
    rew_vec=None,
    set_rand_rew=False,
    gamma=0.999,
    env=None,
    is_set=False,
    extended_SF=False,
):
    """
    Performs value iteration.

    Input:
    - pi: The policy, represented as a dictionary whenre pi(s) = [p(a1|s), p(a2|s), ...]
    - rew_vec: The reward vector to evaluate the policy with. If none, the default reward vector is used.
    - set_rand_rew: If true, evaluate pi with a random reward vector
    - gamma: The discount factor
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP
    - extended_SF: If false, treats rew_vec as a reward function. Otherwise treats rew_vec as an optimal advantage function.

    Output:
    - V: the value function
    - Qs: the Q function
    """
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")

    if isinstance(rew_vec, np.ndarray) and not is_set and not extended_SF:
        env.set_custom_reward_function(rew_vec[0:6])
    elif set_rand_rew and not extended_SF:
        rand_rew_vec = get_random_reward_vector()
        env.set_custom_reward_function(rand_rew_vec)

    THETA = 0.001
    # initialize Q
    V = np.zeros((env.height, env.width))
    Qs = [
        [np.zeros(len(env.actions)) for i in range(env.width)]
        for j in range(env.height)
    ]

    num_actions = len(GridWorldEnv.actions)

    # iterativley learn state value
    while True:
        delta = 0
        new_V = V.copy()
        for i, j in env.positions():
            if env.is_blocked(i, j):
                continue
            v = V[i][j]
            for a_index in range(num_actions):
                next_state, reward, done, _ = env.get_next_state((i, j), a_index)

                if extended_SF:
                    reward = rew_vec[i][j][a_index]

                ni, nj = next_state
                if not done:
                    Q = reward + gamma * V[ni][nj]
                else:
                    Q = reward
                Qs[i][j][a_index] = Q

            new_V[i][j] = max(Qs[i][j])
            delta = max(delta, np.abs(v - new_V[i][j]))

        V = new_V
        if delta < THETA:
            break

    return V, Qs


def get_gt_avg_return(gamma=0.999, gt_rew_vec=None, env=None):
    """
    Gets the average return of the maximum entropy policy

    Input:
    - gamma: The discount factor
    - gt_rew_vec: The reward vector to evaluate the policy with. If none, the default reward vector is used.
    - env: The GridWorld object for the current MDP. If None, use the delivery domain instantiations MDP

    Output:
    - gt_avg_return: The average return of the maximum entropy policy with respect to gt_rew_vec
    """
    _, Q = value_iteration(rew_vec=gt_rew_vec, env=env, gamma=gamma)
    pi = build_pi(Q, env=env)
    V_under_gt = iterative_policy_evaluation(pi, rew_vec=gt_rew_vec, env=env)
    if env is None:
        n_starts = 92
    else:
        n_starts = env.n_starts
    gt_avg_return = np.sum(V_under_gt / n_starts)

    return gt_avg_return


def get_start_state(env):
    i = np.random.randint(0, env.height)
    j = np.random.randint(0, env.width)
    while env.is_terminal(i, j):
        i = np.random.randint(0, env.height)
        j = np.random.randint(0, env.width)
    return (i, j)


def eps_greedy(epsilon, decay_rate, current_action):
    # Choose a random number between 0 and 1
    if np.random.random(1)[0] < epsilon:
        # Exploration: Choose a random action
        chosen_action = np.random.randint(4)
    else:
        # Exploitation: Choose the current best action
        chosen_action = current_action

    # Update epsilon
    epsilon *= decay_rate

    return chosen_action, epsilon



def q_learning_timestep_checkpointed(
    *,
    rew_vec=None,
    gamma=0.999,
    alpha=0.1,
    env=None,
    extended_SF=False,
    n_timesteps=1000,
    return_training_curve=False,
    gt_rew_vec=None,
    checkpoint_every=20,
    epsilon=0.4,
    decay_rate=0.99
):
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")

    if isinstance(rew_vec, np.ndarray) and not extended_SF:
        env.set_custom_reward_function(rew_vec[0:6])

    Qs = [
        [np.zeros(len(env.actions)) for i in range(env.width)]
        for j in range(env.height)
    ]
    training_avg_returns = []
    # np.random.seed(0)
    
    total_collected_steps = 0
    while total_collected_steps < n_timesteps:
        done = False
        state = get_start_state(env)
        n_steps = 0
        while not done and n_steps < 1000:
            if total_collected_steps % checkpoint_every == 0 and return_training_curve:
                assert type(gt_rew_vec) is np.ndarray
                pi = build_pi(Qs,env)
                V_under_gt = iterative_policy_evaluation(pi,rew_vec=gt_rew_vec, env=env)
                avg_return = np.sum(V_under_gt)/env.n_starts
                training_avg_returns.append(avg_return)
            n_steps += 1
            total_collected_steps += 1

            a_index = np.random.choice(np.where(Qs[state[0]][state[1]] == Qs[state[0]][state[1]].max())[0])

            a_index, epsilon = eps_greedy(epsilon, decay_rate, a_index)
            next_state, reward, done, _ = env.get_next_state(state, a_index)
            if extended_SF:
                reward = rew_vec[state[0]][state[1]][a_index]

            # if state == (0,4):
            #     print (a_index, next_state, reward)

            Qs[state[0]][state[1]][a_index] += alpha * (
                reward
                + gamma * np.max(Qs[next_state[0]][next_state[1]])
                - Qs[state[0]][state[1]][a_index]
            )
            state = next_state
       
    if not return_training_curve:
        pi = build_pi(Qs, env)
        V_under_gt = iterative_policy_evaluation(pi, rew_vec=gt_rew_vec, env=env)
        avg_return = np.sum(V_under_gt) / env.n_starts
        return Qs, avg_return
    return Qs, training_avg_returns


def q_learning(
    *,
    rew_vec=None,
    gamma=0.999,
    alpha=0.1,
    env=None,
    extended_SF=False,
    n_episodes=1000,
    return_training_curve=False,
    gt_rew_vec=None,
    checkpoint_every=20,
    epsilon=0.4,
):
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")

    if isinstance(rew_vec, np.ndarray) and not extended_SF:
        env.set_custom_reward_function(rew_vec[0:6])

    Qs = [
        [np.zeros(len(env.actions)) for i in range(env.width)]
        for j in range(env.height)
    ]
    training_avg_returns = []
    # np.random.seed(0)

    for episode in range(n_episodes + 1):  # +1 so that we plot the last epoch
        done = False
        state = get_start_state(env)
        n_steps = 0
        while not done and n_steps < 1000:
            n_steps += 1

            a_index = np.random.choice(np.where(Qs[state[0]][state[1]] == Qs[state[0]][state[1]].max())[0])

            a_index, epsilon = eps_greedy(epsilon, 0.99, a_index)
            next_state, reward, done, _ = env.get_next_state(state, a_index)
            if extended_SF:
                reward = rew_vec[state[0]][state[1]][a_index]

            if state == (0,4):
                print (a_index, next_state, reward)

            Qs[state[0]][state[1]][a_index] += alpha * (
                reward
                + gamma * np.max(Qs[next_state[0]][next_state[1]])
                - Qs[state[0]][state[1]][a_index]
            )
            state = next_state
        if episode % checkpoint_every == 0 and return_training_curve:
            assert isinstance(rew_vec, np.ndarray)
            pi = build_pi(Qs, env)
            V_under_gt = iterative_policy_evaluation(pi, rew_vec=gt_rew_vec, env=env)
            avg_return = np.sum(V_under_gt) / env.n_starts
            training_avg_returns.append(avg_return)
    if not return_training_curve:
        pi = build_pi(Qs, env)
        V_under_gt = iterative_policy_evaluation(pi, rew_vec=gt_rew_vec, env=env)
        avg_return = np.sum(V_under_gt) / env.n_starts
        return Qs, avg_return
    return Qs, training_avg_returns



def compute_rtgs(batch_rews,gamma=0.999):
    # The rewards-to-go (rtg) per episode per batch to return.
    # The shape will be (num timesteps per episode)
    batch_rtgs = []

    # Iterate through each episode
    for ep_rews in reversed(batch_rews):

        discounted_reward = 0 # The discounted reward so far

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * gamma
            batch_rtgs.insert(0, discounted_reward)

    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs


def ppo(
    *,
    rew_vec=None,
    gamma=0.999,
    alpha=0.1,
    env=None,
    extended_SF=False,
    n_episodes=1000,
    return_training_curve=False,
    gt_rew_vec=None,
    checkpoint_every=20,
    epsilon=0.4,
):
    if env is None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")

    if isinstance(rew_vec, np.ndarray) and not extended_SF:
        env.set_custom_reward_function(rew_vec[0:6])

    actor = FeedForwardNN(env.height*env.width, 4)
    critic = FeedForwardNN(env.height*env.width, 1)

    actor_lr = 0.005
    critic_lr = 0.005
    # total_timesteps = 500000
    clip = 0.2
    episodes_per_batch = 5
    n_updates_per_iteration = 10

    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    cov_var = torch.full(size=(4,), fill_value=0.5)
    training_avg_returns = []

    def rollout():
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        n_batch_steps = 0
        # while n_batch_steps < timesteps_per_batch:  # +1 so that we plot the last epoch
        for episode in range(episodes_per_batch):
            done = False
            state = get_start_state(env)
            n_steps = 0

            ep_rews = []

            while not done and n_steps < 1000:
                n_steps += 1
                # n_batch_steps += 1

                state_one_hot = torch.zeros((env.height, env.width))
                state_one_hot[state[0]][state[1]] = 1
                state_one_hot = state_one_hot.flatten()

                batch_obs.append(state_one_hot)
                a_index, log_prob = get_action(state_one_hot)
                
            
                next_state, reward, done, _ = env.get_next_state(state, a_index)
                if extended_SF:
                    reward = rew_vec[state[0]][state[1]][a_index]

                ep_rews.append(reward)
                batch_acts.append(a_index)
                batch_log_probs.append(log_prob)

                state = next_state
            batch_lens.append(n_steps)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.stack(batch_obs).float()
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = compute_rtgs(batch_rews) 
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens  

    def get_action(obs):
		# Query the actor network for a mean action
	    probs = actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
	    dist = torch.distributions.Categorical(probs)

		# Sample an action from the distribution
	    action = dist.sample()

		# Calculate the log probability for that action
	    log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
	    return action.detach().numpy().item(), log_prob.detach() 
    
    def evaluate(batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        probs = actor(batch_obs)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    t_so_far = 0 # Timesteps simulated so far
    i_so_far = 0 # Iterations ran so far
    last_checkpointed_timestep = 0
    n_collected_episodes = 0
    while n_collected_episodes < n_episodes + 1:                                                                       # ALG STEP 2
        # Autobots, roll out (just kidding, we're collecting our batch simulations here)
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = rollout()                     # ALG STEP 3

        # Calculate how many timesteps we collected this batch
        t_so_far += np.sum(batch_lens)
        n_collected_episodes += episodes_per_batch

        # print ( np.sum(batch_lens))

        # Increment the number of iterations
        i_so_far += 1

        
        # Calculate advantage at k-th iteration
        V, _ = evaluate(batch_obs, batch_acts)
        A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

        # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
        # isn't theoretically necessary, but in practice it decreases the variance of 
        # our advantages and makes convergence much more stable and faster. I added this because
        # solving some environments was too unstable without it.
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        # This is the loop where we update our network for some n epochs
        for _ in range(n_updates_per_iteration):                                                       # ALG STEP 6 & 7
            # Calculate V_phi and pi_theta(a_t | s_t)
            V, curr_log_probs = evaluate(batch_obs, batch_acts)

            # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            # NOTE: we just subtract the logs, which is the same as
            # dividing the values and then canceling the log with e^log.
            # For why we use log probabilities instead of actual probabilities,
            # here's a great explanation: 
            # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
            # TL;DR makes gradient ascent easier behind the scenes.
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses.
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - clip, 1 + clip) * A_k

            # Calculate actor and critic losses.
            # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
            # the performance function, but Adam minimizes the loss. So minimizing the negative
            # performance function maximizes it.
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

            # Calculate gradients and perform backward propagation for actor network
            actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optim.step()

            # Calculate gradients and perform backward propagation for critic network
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
    

        if (n_collected_episodes % checkpoint_every == 0) and return_training_curve:
            assert isinstance(rew_vec, np.ndarray)
            last_checkpointed_timestep = t_so_far

            pi = {}
            for i, j in env.positions():
                state_one_hot = torch.zeros((env.height, env.width))
                state_one_hot[i][j] = 1
                state_one_hot = state_one_hot.flatten()
                probs = actor(state_one_hot)
                probs = probs.detach().tolist()
                pi[(i,j)] = [(probs[a], a) for a in range(4)]

            V_under_gt = iterative_policy_evaluation(pi, rew_vec=gt_rew_vec, env=env)
            avg_return = np.sum(V_under_gt) / env.n_starts
            # print (avg_return)
            training_avg_returns.append(avg_return)
    return training_avg_returns