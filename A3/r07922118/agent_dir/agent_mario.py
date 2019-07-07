import torch
from torch.distributions import Categorical
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

from a2c.environment_a2c import make_vec_envs
from a2c.storage import RolloutStorage
from a2c.actor_critic import ActorCritic

from collections import deque
import os

use_cuda = torch.cuda.is_available()

class AgentMario: #actor agent
    def __init__(self, env, args):

        # Hyperparameters
        self.lr = 7e-4
        self.gamma = 0.99
        self.hidden_size = 512
        self.update_freq = 5
        self.n_processes = 16
        self.seed = 7122
        self.max_steps = 1e7
        self.grad_norm = 0.5
        self.entropy_weight = 0.05

        #######################    NOTE: You need to implement
        self.recurrent = False # <- ActorCritic._forward_rnn()
        #######################    Please check a2c/actor_critic.py
        
        if args.test_mario:
            self.load_model('./checkpoints/model.pt')
        self.display_freq = 4000
        self.save_freq = 10000
        self.save_dir = './checkpoints/'

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.envs = env
        if self.envs == None:
            self.envs = make_vec_envs('SuperMarioBros-v0', self.seed,
                    self.n_processes)
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.obs_shape = self.envs.observation_space.shape
        self.act_shape = self.envs.action_space.n
        #print(self.obs_shape) #(4, 84, 84)
        #print(self.act_shape) #12
        self.rollouts = RolloutStorage(self.update_freq, self.n_processes,
                self.obs_shape, self.act_shape, self.hidden_size) 
        self.model = ActorCritic(self.obs_shape, self.act_shape,
                self.hidden_size, self.recurrent).to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, 
                eps=1e-5)

        self.hidden = None
        self.init_game_setting()
   
    def _update(self):
        # TODO: Compute returns
        #print(self.rollouts.obs.size()) #torch.Size([6, 16, 4, 84, 84])
        obs_shape = self.rollouts.obs.size()[2:]
        #print(obs_shape) #torch.Size([4, 84, 84])
        #print(self.rollouts.actions.size()) #torch.Size([5, 16, 1])
        action_shape = self.rollouts.actions.size()[-1]
        #print(action_shape) #1
        num_steps, num_processes, _ = self.rollouts.rewards.size()
        #see https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py line 38-43
        #input()
        # R_t = reward_t + gamma * R_{t+1}
        discounted_return =  torch.zeros(self.update_freq, self.n_processes, 1).to(self.device)
        #print(self.rollouts.rewards)
        for t in range(self.update_freq-1, -1, -1):
            discounted_return[t]= self.rollouts.rewards[t]+ self.gamma*self.rollouts.value_preds[t+1]
            #print(t)
            #print(self.rollouts.masks[t])
        #print(self.rollouts.obs[:-1]) # [:-1] means don't take the last element
        #print(self.rollouts.obs[:-1].shape)#torch.Size([5, 16, 4, 84, 84])
       # print(self.rollouts.obs[:-1].view(-1, *obs_shape).shape)# torch.Size([80, 4, 84, 84]) n_steps*n_processes, 4, 84, 84
        #print(self.rollouts.hiddens[0].shape)#torch.Size([16, 512])
        #print(self.rollouts.hiddens[0].view(-1, self.model.hidden_size).shape) #torch.Size([16, 512])
        #print(self.rollouts.masks[:-1].view(-1, 1).shape) #torch.Size([80, 1])
        values, action_probs, hiddens= self.model(self.rollouts.obs[:-1].view(-1, *obs_shape), self.rollouts.hiddens[0].view(-1, self.model.hidden_size), self.rollouts.masks[:-1].view(-1, 1) )
        #print(values.shape) #torch.Size([5, 16, 1])
        #print(action_probs.shape) #torch.Size([5, 16, 12])
        #print(hiddens.shape) #torch.Size([16, 512])
        values= values.view(num_steps, num_processes, 1)
        action_probs =action_probs.view(num_steps, num_processes, -1)
        #print(action_probs)
        #print(action_probs.gather(2 ,self.rollouts.actions))
        #print(action_probs.gather(2 ,self.rollouts.actions).shape) #torch.Size([5, 16, 1])
        #m=Categorical(action_probs)
        action_probs= action_probs.gather(2 ,self.rollouts.actions)
        #print(m)
        #print(self.rollouts.actions)
        #print(action_probs)
        action_log_probs = action_probs.log()
        #action_log_probs = m.log_prob(self.rollouts.actions.view(-1, action_shape))
        #print(action_log_probs)
        #print(action_log_probs.shape) #torch.Size([5, 16, 1])
        #input()
        #deal with self.rollouts.actions later!
        #=self.model(self.rollouts.obs)
        #print(self.rollouts.rewards.shape) #torch.Size([5, 16, 1])
        #print(self.rollouts.value_preds.shape)#torch.Size([6, 16, 1])
        advantages= discounted_return - values

        #not so sure, advantage=  r_t+gamma* V(s_t+1) - V(s_t) ?????
        #print(advantages)
        #print(advantages.shape) #torch.Size([5, 16, 1])
        #print(self.rollouts.action_log_probs.shape) #torch.Size([5, 16, 1])
        #input()
        #self.gamma*
        # TODO:
        #value loss is the critic loss; action loss is the actor loss
        # Compute actor critic loss (value_loss, action_loss)
        # OPTIONAL: You can also maxmize entropy to encourage exploration
        #use output entropy as regularization for pi(s)
        # loss = value_loss + action_loss (- entropy_weight * entropy)
        #see https://github.com/jcwleo/mario_rl/blob/master/mario_a2c.py line 260-267
        critic_loss= advantages.pow(2).mean()
        #print(critic_loss.grad)
        #print(critic_loss) #tensor(1.2946, device='cuda:0', grad_fn=<MeanBackward1>)
        #print(critic_loss.shape) #torch.Size([])
        #https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/tree/master/a2c_ppo_acktr  -->USEFUL
        actor_loss=  -(advantages*action_log_probs).mean()
        #print(actor_loss.grad)
        #print(actor_loss) #tensor(1.1621, device='cuda:0', grad_fn=<NegBackward>)

        #print(actor_loss.shape) #torch.Size([])
        #input()
        loss = actor_loss +critic_loss
        #print(loss) #tensor(2.4567, device='cuda:0', grad_fn=<AddBackward0>)

        #print(loss.shape)
        #input()
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        
        # TODO:
        # Clear rollouts after update (RolloutStorage.reset())
        self.rollouts.reset()
        return loss.item()

    def _step(self, obs, hiddens, masks):#_step is just 1 step
        with torch.no_grad():
            #16 is n_processes, meaning 16 workers, means batch_size is 16(?)
            #print("obs.shape", obs.shape) #torch.Size([16, 4, 84, 84])
            #print(hiddens.shape) #torch.Size([16, 512])
            #print(masks.shape) #torch.Size([16, 1])
            #self.model has 3 inputs
            #I think we should for loop 16 times to get the state of each worker
            #which is WRONG!
            #for i in range(self.n_processes):
            values, action_probs, hiddens= self.model(obs, hiddens, masks)
            #values : V(st)  obs: st
            #print(values.shape) #
            #print(hiddens.shape)
            #print(action_probs) #torch.Size([1, 16, 12])
            #print(action_probs.shape) #torch.Size([16, 12])
            #action_probs means F.softmax(policy)
            m=Categorical(action_probs)  
            #print(m) #Categorical(probs: torch.Size([16, 12]))
            actions= m.sample()
            #print(m.log_prob(actions).shape)
            #input()
            action_log_probs =m.log_prob(actions).unsqueeze(1)
            #print(m.log_prob(actions))
            #print(m.log_prob(actions).shape) #torch.Size([1, 16])
            #input()
            #print(actions)#tensor([[9, 4, 8, 6, 4, 3, 9, 3, 0, 3, 5, 5, 1, 0, 2, 5]], device='cuda:0')
            #print(actions.shape) #torch.Size([16])
            actions=actions.squeeze(0)
            #print(actions.cpu().numpy()) #[ 0  0  1  4  4  2  8  8  0  4  7  7  6 11  9  3]
            #input()
            #if you don't use recurrent, you don't need hidden and masks
            #values, action_provs, hiddens =self.model(obs, hiddens, masks)
            #actions=self.make_actions(obs)
            # TODO:
            # Sample actions from the output distributions
            # HINT: you can use torch.distributions.Categorical
            #see https://github.com/jcwleo/mario_rl/blob/master/mario_a2c.py line 256-257
#see https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/main.py line 113~132        
        obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy())
        #obs here is s_t+1
        #the step you're calling here is in shmem_vec_env.py step_async
        #you are inputing 16 actions to 16 environments
        #print(dones) #[False False False False False False False False False False False False
# False False False False]
        #print(1-dones) #[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
        #print(infos)
        #input()
        #rewards : rt, truly obtain when taking actions at
        values= values.squeeze(0)
        actions= actions.unsqueeze(1)
        obs= torch.from_numpy(obs)
        rewards=torch.from_numpy(rewards).unsqueeze(1)
        #print(rewards.shape)
        masks= torch.from_numpy(1-dones).unsqueeze(1)
        # TODO:
        self.rollouts.insert(obs, hiddens, actions,action_log_probs , values, rewards, masks)
        # Store transitions (obs: s_t+1, hiddens, actions:a_t , values: V(s_t), rewards: r_t,  masks)
        # You need to convert arrays to tensors first
        # HINT: masks = (1 - dones)
         
    def train(self):

        print('Start training')
        running_reward = deque(maxlen=10)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0
        
        # Store first observation
        obs = torch.from_numpy(self.envs.reset()).to(self.device)
        #print(obs.shape) #torch.Size([16, 4, 84, 84])
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        #print(obs.shape) #torch.Size([16, 4, 84, 84])
        #print(self.rollouts.obs.shape) #torch.Size([6, 16, 4, 84, 84]) 
        # 6 is n_steps+1  --> see ../a2c/storage.py 
        while True:
            # Update once every n-steps
            for step in range(self.update_freq):
                self._step(
                    self.rollouts.obs[step],
                    self.rollouts.hiddens[step],
                    self.rollouts.masks[step])

                # Calculate episode rewards
                episode_rewards += self.rollouts.rewards[step]
                for r, m in zip(episode_rewards, self.rollouts.masks[step + 1]):
                    #print(r)
                    #print(m)
                    if m == 0:
                        running_reward.append(r.item())
                episode_rewards *= self.rollouts.masks[step + 1]

            loss = self._update()#update here
            total_steps += self.update_freq * self.n_processes

            # Log & save model
            if len(running_reward) == 0:
                avg_reward = 0
            else:
                avg_reward = sum(running_reward) / len(running_reward)

            if total_steps % self.display_freq == 0:
                print('Steps: %d/%d | Avg reward: %f'%
                        (total_steps, self.max_steps, avg_reward))
            
            if total_steps % self.save_freq == 0:
                self.save_model('model.pt')
            
            if total_steps >= self.max_steps:
                break

    def save_model(self, filename):
        print("Save the model to ", self.save_dir)
        torch.save(self.model, os.path.join(self.save_dir, filename))

    def load_model(self, path):
        print("Load the model from ", path)
        self.model = torch.load(path)

    def init_game_setting(self):
        if self.recurrent:
            self.hidden = torch.zeros(1, self.hidden_size).to(self.device)

    def make_action(self, observation, test=False):
        # TODO: Use you model to choose an action
        #self.load_model("./checkpoints/model.pt") #load the model somewhere else!
        #print(observation.shape) #(4, 84, 84)
        #print(observation)
        observation= torch.from_numpy(observation).to(self.device).unsqueeze(0)
        #when do we call this function??? -->../test/py line 41 will call this function
        #you also need to differentiate test=True and test=False
        #see https://github.com/jcwleo/mario_rl/blob/master/mario_a2c.py line 170
        #see https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/evaluation.py line 20-31
        eval_recurrent_hidden_states = torch.zeros(
        self.n_processes, self.model.hidden_size, device=self.device)
        eval_masks = torch.zeros(self.n_processes, 1, device=self.device)
        _, action_probs, _= self.model(observation, eval_recurrent_hidden_states, eval_masks) 
        #print(action_probs)
        #print(action_probs.shape) #torch.Size([1, 12])
        #print(action_probs.max(1)[1])
        #print(action_probs.max(1)[1].item())
        action= action_probs.max(1)[1].item()
        #print(action)
        return action
