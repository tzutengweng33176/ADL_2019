'''
1. How to train my model? 
I failed to finish A2C implementation. But I have implemented policy gradient with baseline, DQN, and double DQN.

python3.6 main.py --train_pg

        self.gamma= 0.99
        self.num_episodes= 10000


python3.6 main.py --train_dqn

        # discounted reward
        self.GAMMA = 0.99 
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network
        #epsilon greedy policy hyperparameters
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay= 200

I trained DQN for 9760 episodes.

Episode: 9760 | Steps: 2620245/3000000 | Avg reward: 9.100000 | loss: 0.037604

'''


'''
2. How to plot the figures in my report? 
I created two lists x and y to store number of timesteps and avg reward respectively.
And then I dump x and y to pickle files. (There are different x and y pickle files for different hyperparameters and different algorithms)

Below is 1 example of dumping out the pickle files.


          pickle_out_x= open('x_pg_with_baseline.pkl', 'wb')
          pickle_out_y=open('y_pg_with_baseline.pkl', 'wb')
          pickle.dump(x, pickle_out_x)
          pickle.dump(y, pickle_out_y)
          pickle_out_x.close()
          pickle_out_y.close()


Then I load pickle files and plot them by using matplotlib.

plt.figure(figsize=(10, 10))
pickle_in_x = open("x_0.99.pkl","rb")
pickle_in_y = open("y_0.99.pkl","rb")
x_099 = pickle.load(pickle_in_x)
y_099 = pickle.load(pickle_in_y)

pickle_in_x_ddqn = open("ddqn_x_0.99.pkl","rb")
pickle_in_y_ddqn = open("ddqn_y_0.99.pkl","rb")
x_ddqn = pickle.load(pickle_in_x_ddqn)
y_ddqn = pickle.load(pickle_in_y_ddqn)


plt.plot(x_099, y_099, label='DQN')
plt.plot(x_ddqn, y_ddqn, label='Double DQN')


plt.legend(loc='lower right')
plt.xlabel('Timesteps')
plt.ylabel('Avg Reward in the last 10 episodes')
plt.title('Avg reward of DQN and Double DQN',fontweight='bold', fontsize=16 )
plt.savefig('DDQN_DQN.png')

'''
