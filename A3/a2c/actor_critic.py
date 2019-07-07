import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_size, recurrent):
        super(ActorCritic, self).__init__()
        
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        #obs_shape= (4, 84, 84) 
        #act_shape =12
        #hidden_size =512
        #obs_shape[0] should be 4
        self.head = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, hidden_size),
            nn.ReLU()
        )

        if self.recurrent:
            self.rnn =  nn.GRU(hidden_size, hidden_size)
        
        self.actor = nn.Linear(hidden_size, act_shape)  #policy network is just nn.Linear...
        self.critic = nn.Linear(hidden_size, 1)  #value function is just nn.Linear.....
        
        self.reset_parameters()

    def reset_parameters(self):
        
        def _weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.head.apply(_weights_init)

        if self.recurrent:
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.constant_(self.critic.bias, 0)

    def _forward_rnn(self, x, hiddens, masks):
        '''
        Args:
            x: observations -> (n_steps * n_processes, hidden_size)
            hiddens: hidden states of 1st step -> (n_processes, hidden_size)
            masks: whether to reset hidden state -> (n_steps * n_processes, 1)
        Returns:
            x: outputs of RNN -> (n_steps * n_processes, hidden_size)
            hiddens: hidden states of last step -> (n_processes, hidden_size)
        '''
        # TODO
        # step 1: Unflatten the tensors to (n_steps, n_processes, -1) 
        # step 2: Run a for loop through time to forward rnn
        # step 3: Flatten the outputs
        # HINT: You must set hidden states to zeros when masks == 0 in the loop 
        n_processes, hidden_size = hiddens.shape
        #print(x.shape) #torch.Size([16, 512])
        n_steps = int(x.size(0)/n_processes)
        x= x.view(n_steps, n_processes, hidden_size)
        #print(x.shape) #torch.Size([1, 16, 512]) # (seq_len, batch, input_size)
        #print(masks.shape)
        #https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py  line 129-164
        masks= masks.view(n_steps, n_processes, 1)
        print(masks)
        has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())
        
        #print(masks.shape) #torch.Size([1, 16, 1])
        #print(has_zeros) #tensor([], size=(0, 2), dtype=torch.int64)
        #print(has_zeros.shape) #torch.Size([0, 2])
        #input()
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
        #print(has_zeros) # []; 
        has_zeros = [0] + has_zeros + [n_steps]
        print(has_zeros) #[0, 1]
        print(len(has_zeros)) #has_zeros is a list
        #input()
        hiddens=hiddens.unsqueeze(0)
        outputs=[]
        for i in range(len(has_zeros) - 1):
            # We can now process steps that don't have any zeros in masks together!
            # This is much faster
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, hiddens = self.rnn(
                    x[start_idx:end_idx],
                    hiddens * masks[start_idx].view(1, -1, 1))

            outputs.append(rnn_scores)
        #print(outputs)
        print(len(outputs))
        print(n_steps)
        #x, h_n=self.rnn(x )# -->you need to put hiddens in self.rnn
        #see https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py line 112-114
        #print(n_processes) #16
        #print(hidden_size) #512
        #print(x.shape) #torch.Size([1 ,16, 512]) # (seq_len, batch, input_size)
        #print(hiddens.shape) #torch.Size([16, 512])
        #print(masks)
        #print(masks.shape)#torch.Size([16, 1])
        #masks= masks.to(torch.uint8)
        #print(masks)
        #print(h_n.shape) #torch.Size([1, 16, 512])
        #print(h_n)
        #h_n= h_n.squeeze(0)
        #print(h_n.masked_select(masks).shape)
        #hiddens= h_n.masked_select(masks).view( -1,n_processes, hidden_size)[-1]  #-->hidden state of the last step (?)
        #NOT SO SURE
        #print(hiddens)
        #print(hiddens.shape)
        assert len(outputs) == n_steps
        # x is a (T, N, -1) tensor
        x = torch.cat(outputs, dim=0)
        #print(x)
        #print(x.shape) #torch.Size([1, 16, 512])
        # flatten
        x = x.view(n_steps *n_processes, -1)
        hiddens = hiddens.squeeze(0)
        #input()
        return x, hiddens

    def forward(self, inputs, hiddens, masks):
        x = self.head(inputs / 255.0) #head is feature
        #print(masks.shape)
        #see https://github.com/jcwleo/mario_rl/blob/master/model.py line 212
        if self.recurrent:
            x, hiddens = self._forward_rnn(x, hiddens, masks)
        
        values = self.critic(x)
        action_probs = self.actor(x)
        action_probs = F.softmax(action_probs, -1)
        
        return values, action_probs, hiddens

        
