import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from agent_dir.agent import Agent
from environment import Environment
import matplotlib.pyplot as plt
#No critic in the agent
#policy-based RL; search directly for optimal policy pi
#learning an actor
class PolicyNet(nn.Module): #see this as a classifier
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1) #here, we are implementing the stochastic policy
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        self.model_2 = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')
        # discounted reward
        self.gamma = 0.99 
        
        # training hyperparameters
        self.num_episodes = 10000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        
        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        self.saved_log_probs =[] 

        #save states
        self.states= []
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []
        self.saved_log_probs=[]

    def cal_log_prob(self, state, action):
        #print(action) #int 
        action=torch.tensor(action).unsqueeze(0).unsqueeze(0)
        #print(action) #tensor([[3]])
        #print(action.shape) #torch.Size([1, 1])
        state= torch.from_numpy(state).float().unsqueeze(0) #(1, num_of_states)
        #print(state.shape) #torch.Size([1, 8])
        action_probs=self.model_2(state)
        #print(action_probs)
        action=action_probs.gather(1, action)
        #print(action)
        log_prob= action.log()
        #print(log_prob)
        #input()
        #print(action_prob.shape) #torch.Size([1, 4])
        #print(action_prob) #tensor([[0.1969, 0.2707, 0.3099, 0.2225]], grad_fn=<SoftmaxBackward>)
        #m= Categorical(action_probs) #https://pytorch.org/docs/stable/distributions.html
        #print(m) #Categorical(probs: torch.Size([1, 4]))
        #action = m.sample()
        #print(action) #tensor([1])
        return log_prob 
    
    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        #print(state) #a numpy array
        #print(state.shape) #(8,) 
        state= torch.from_numpy(state).float().unsqueeze(0) #(1, num_of_states)
        #print(state.shape) #torch.Size([1, 8])
        action_probs=self.model(state)
        #print(action_prob.shape) #torch.Size([1, 4])
        #print(action_prob) #tensor([[0.1969, 0.2707, 0.3099, 0.2225]], grad_fn=<SoftmaxBackward>)
        m= Categorical(action_probs) #https://pytorch.org/docs/stable/distributions.html
        #print(m) #Categorical(probs: torch.Size([1, 4]))
        action = m.sample()
        #print(action) #tensor([1])
        #print(m.log_prob(action)) #tensor([-1.2286], grad_fn=<SqueezeBackward1>)
        self.saved_log_probs.append(m.log_prob(action) ) #save logP(at|st, theta)), use this to compute the gradient later

        action= action.item()
        #print(action) #1
        return action
#policy gradient is on-policy....
#on-policy: the agent learned and the agent interacting with the env is the same
#off-policy: the agent learned and the agent interacting with the env is different
#so if you want to do off-policy, you need 2 different agents! -->HOW???
    def update(self):
        # TODO:
        # discount your saved reward 
        #maximize total discounted reward by SGD
        #vanilla policy gradient 
        R=0 #cumulative reward R(tau) of the whole trajectory instead of immediate reward rt
        loss=[]
        returns= []
        #print(self.rewards)
        for r in self.rewards[::-1]: #self.rewards store reward produced by the environment after the agent made each action
            #print(r)
            R= r + self.gamma*R      
            #print(R)
            returns.insert(0, R)     #retur
            #print(r)                #r1, r2, ..., rTi, but you need to read the list in reverse order
        #print(returns)
        #returns =[ GAMMA^(T-1)*rT+ GAMMA^(T-2)*rT-1+...+GAMMA*r2+r1, ..., GAMMA*rT+rT-1, rT]
        Z_s=[]
        time=1
        for (states, actions, Rs, log_probs) in zip(self.states, self.saved_actions, returns, self.saved_log_probs):
            p_log_prob=0
            q_log_prob=0
            for t in range(time):
                #print("Time: ", t)
                q_log_prob +=self.cal_log_prob(self.states[t], self.saved_actions[t]).data.numpy().squeeze(1)
                p_log_prob+=self.saved_log_probs[t].data.numpy()
                #print(p_log_prob.shape)
                #print(q_log_prob.shape)
            #input()
            Z_= math.exp(p_log_prob) / math.exp(q_log_prob)
            #print(Z_)
            #input()
            time+=1
            Z_s.append(Z_)
        #variance reduction with discounts
        returns= torch.tensor(returns)
        #print(returns)
        returns= (returns - returns.mean())/ (returns.std()) #normalize the returns #advantage estimate
        #I think returns.mean() means the baseline -->YES! 
        #average reward is NOT the best baseline, but it's pretty good
        #print(returns)
        #print(len(returns)) #len(returns) == len(self.saved_log_probs)
        #print(len(self.saved_log_probs)) #84
        #print(len(Z_s))
        #input()
        for log_prob, z,R in zip(self.saved_log_probs, Z_s, returns):
            #print(-log_prob*R) #tensor([1.0843], grad_fn=<MulBackward0>)
            loss.append(-log_prob*z*R) #need more study!! Ask TA if needed-->different from https://www.youtube.com/watch?v=puKAEWkZDSE 26:12
        # TODO:
        #maximizing log_prob*R means minimizing -log_prob*R
        #future actions do not depend on past rewards....
        #https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Reinforcement-Learning-Through-Policy-Optimization
        #see the above video at 26:26 and 27:20 
        # compute loss
        #Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        #print(loss)
        #print(torch.cat(loss))
        #input()
        self.optimizer.zero_grad()
        loss=torch.cat(loss).sum() #sum the loss up, sum over the whole trajectory
        loss.backward() #compute gradient of the loss with respect to model parameters
        self.optimizer.step() #update the policy here
        #Calling the step function on an Optimizer makes an update to its parameters
        del self.rewards[:] #
        del self.saved_log_probs[:] #
        del self.saved_actions[:] #reset every episode
        del self.states[:]

    def train(self):
        avg_reward = None # moving average of reward
        num_of_steps=0
        x=[]
        y=[]
        for epoch in range(self.num_episodes): #1 episode is 1 trajectory; play the game for 1 time; you get 1 tau
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done): #the while loop here is the true trajectory
                action = self.make_action(state) #s_t
                self.states.append(state) #store s_t
                #print(action) #1 action
                #well, we did not store the states here....maybe we don't need them
                state, reward, done, _ = self.env.step(action)
                #s_t+1
                #print(state.shape)   #[-0.00976191  1.4096465  -0.49909645 -0.04140926  0.01292099  0.14601704
                                  #  0.          0.        ] (8,)

                #print(reward) #-1.6515174521975087
                #input()
                self.saved_actions.append(action)  #action the agent made, a_t
                self.rewards.append(reward)  #reward from the env, r_t
                num_of_steps+=1

            #print("Current timestep: ", num_of_steps)
            x.append(num_of_steps)
            # for logging 
            last_reward = np.sum(self.rewards) #sum rewards in 1 episode
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            y.append(avg_reward) #append avg reward at the end of 1 episode
            # update model
            self.update()

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
            
            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                plt.plot(x, y)
                plt.xlabel('Timesteps')
                plt.ylabel('Avg reward in last 1 episode')
                plt.show()
                plt.savefig('pg_baseline.png')
                self.save('pg.cpt')
                break
