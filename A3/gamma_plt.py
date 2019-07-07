import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 10))
pickle_in_x = open("x_0.99.pkl","rb")
pickle_in_y = open("y_0.99.pkl","rb")
x_099 = pickle.load(pickle_in_x)
y_099 = pickle.load(pickle_in_y)

pickle_in_x_ddqn = open("ddqn_x_0.99.pkl","rb")
pickle_in_y_ddqn = open("ddqn_y_0.99.pkl","rb")
x_ddqn = pickle.load(pickle_in_x_ddqn)
y_ddqn = pickle.load(pickle_in_y_ddqn)

#pickle_in_x80 = open("x_0.80.pkl","rb")
#pickle_in_y80 = open("y_0.80.pkl","rb")
#x_080 = pickle.load(pickle_in_x80)
#y_080 = pickle.load(pickle_in_y80)

#pickle_in_x50 = open("x_0.5.pkl","rb")
#pickle_in_y50 = open("y_0.5.pkl","rb")
#x_050 = pickle.load(pickle_in_x50)
#y_050 = pickle.load(pickle_in_y50)

#pickle_in_x01 = open("x_0.01.pkl","rb")
#pickle_in_y01 = open("y_0.01.pkl","rb")
#x_001 = pickle.load(pickle_in_x01)
#y_001 = pickle.load(pickle_in_y01)

plt.plot(x_099, y_099, label='DQN')
plt.plot(x_ddqn, y_ddqn, label='Double DQN')

#plt.plot(x_080, y_080, label='gamma=0.80')
#plt.plot(x_050, y_050, label='gamma=0.50')
#plt.plot(x_001, y_001, label='gamma=0.01')

plt.legend(loc='lower right')
plt.xlabel('Timesteps')
plt.ylabel('Avg Reward in the last 10 episodes')
plt.title('Avg reward of DQN and Double DQN',fontweight='bold', fontsize=16 )
#new_ticks = np.linspace(0, 10, 100)
#print(new_ticks)
#plt.yticks(new_ticks)

plt.savefig('DDQN_DQN.png')


