import pickle
import matplotlib.pyplot as plt

rew_file_name = 'ag0_rewards.pkl'
ddpg_rew_file_name = 'ag1_rewards.pkl'
fp=open(rew_file_name,'rb')
ddpg_fp=open(ddpg_rew_file_name,'rb')
rew = pickle.load(fp)
ddpg_rew = pickle.load(ddpg_fp)
rew = rew[:int(len(rew))]
ddpg_rew = ddpg_rew[:int(len(ddpg_rew))]
fp.close()
ddpg_fp.close()
x1=[i for i in range(len(rew)) if i%2==0]
x2 = [i for i in range(len(ddpg_rew)) if i%2==0]
rew = [rew[x1[i]] for i in range(len(x1))]
ddpg_rew = [ddpg_rew[x2[i]] for i in range(len(x2))]
plt.xlabel('Episodes')
plt.ylabel('Points')
plt.plot(x1,rew,'b',label='ddpg')
plt.plot(x2,ddpg_rew,'r',label='spac')
plt.legend(loc='best')
plt.show()
