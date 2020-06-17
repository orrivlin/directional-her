import torch
import numpy
from models.conv_nets import DiscreteActor, DiscreteCritic
from torch.optim import Adam
import numpy as np
from memory.base import LazyMultiStepMemory
from memory.per import LazyPrioritizedMultiStepMemory




def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


class DiscreteSAC:

    def __init__(self, env, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=5,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, max_episode_steps=27000, log_interval=10,
                 device='cuda', seed=0):

        self.env = env
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.lr = lr
        self.memory_size = memory_size
        self.multi_step = multi_step
        self.gamma_n = gamma ** multi_step
        self.target_entropy_ratio = target_entropy_ratio
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.device = device

        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step)

        self.policy = DiscreteActor(3, self.env.action_space.n).to(device)
        self.critic = DiscreteCritic(3, self.env.action_space.n).to(device)
        self.target_critic = DiscreteCritic(3, self.env.action_space.n).to(device)
        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0


    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())


    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2


    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                    torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                    )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q


    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2


    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()


    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss


    def is_update(self):
        return self.steps % self.update_interval == 0 \
               and self.steps >= self.start_steps


    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:

            if self.start_steps > self.steps:
                action = self.env.action_space.sample()
            else:
                #action = self.explore(state)
                action = self.exploit(self.env.get_tensor(state, self.device))

            next_state, reward, done, _ = self.env.step(action)

            # Clip reward to [-1.0, 1.0].
            clipped_reward = max(min(reward, 1.0), -1.0)

            # To calculate efficiently, set priority=max_priority here.
            self.memory.append(state, action, clipped_reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if self.is_update():
                self.learn()

            if self.steps % self.target_update_interval == 0:
                self.update_target()


    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and \
               hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)


    def explore(self, state):
        # Act with randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()


    def exploit(self, state):
        # Act without randomness.
        # state = torch.ByteTensor(
        #     state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.policy.act(state)
        return action





