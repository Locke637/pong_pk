rew_n = [-1, 0.1 , 1]
for i in range(len(rew_n)):
    if rew_n[i] != -1:
        rew_n[i] = rew_n[i] * 10
print(rew_n)