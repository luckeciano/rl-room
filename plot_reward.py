import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

v = pd.read_csv('run-stats.csv', engine = 'python')


plt.plot(start_line, v['mean_reward'])
#plt.plot(start_line, min_vel, c='g', ls='dashed')
plt.fill_between(start_line, v['mean_reward'] - v['std_reward'], v['mean_reward'] + v['std_reward'],  alpha = 0.4)
plt.grid()
plt.title('Behavior Cloning - Evaluation')
plt.xlabel('Number of Rollouts')
plt.ylabel('Reward')
plt.legend(['Average', 'CI'])
# plt.yticks(np.arange(1.350, 3.86, .250))
plt.savefig('bc-rw-rollouts.pdf', format='pdf')
plt.show()
