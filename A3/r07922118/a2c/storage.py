import torch
#why do we need RolloutStorage????
class RolloutStorage:
    def __init__(self, n_steps, n_processes, obs_shape, action_space,
            hidden_size):
        self.obs = torch.zeros(n_steps + 1, n_processes, *obs_shape)
        self.hiddens = torch.zeros(
            n_steps + 1, n_processes, hidden_size)
        self.rewards = torch.zeros(n_steps, n_processes, 1)
        self.value_preds = torch.zeros(n_steps + 1, n_processes, 1)
        self.returns = torch.zeros(n_steps + 1, n_processes, 1)
        self.actions = torch.zeros(n_steps, n_processes, 1).long()
        self.masks = torch.ones(n_steps + 1, n_processes, 1)
        self.action_log_probs = torch.zeros(n_steps, n_processes, 1) #I add this line based on 
        #https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py
        self.n_steps = n_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.hiddens = self.hiddens.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.action_log_probs = self.action_log_probs.to(device) #I add this line based on 
        #https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

    def insert(self, obs, hiddens, actions,action_log_probs , value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.hiddens[self.step + 1].copy_(hiddens)
        self.actions[self.step].copy_(actions)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.step = (self.step + 1) % self.n_steps

    def reset(self):
        self.obs[0].copy_(self.obs[-1])
        self.hiddens[0].copy_(self.hiddens[-1])
        self.masks[0].copy_(self.masks[-1])

