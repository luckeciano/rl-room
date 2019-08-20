import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

rw_mean = [452.8695197063927, 436.1876193166848, 812.8307270172436, 1133.6374329917526, 2079.78239935714, 3016.146423122095,
			4233.35359986044, 6156.204532177981, 7836.316595881401, 9507.640722141297]

rw_std = [138.59100357967102, 166.2629992589841, 343.78876369331675, 855.6059219578616, 1352.6492220479627, 2376.074999695875,
			3322.521088414047, 3456.0774811147885, 3357.8317035487908, 3059.205978257879]

plt.axhline(y = 396.12, ls=':', c='r')
plt.axhline(y = 10676.401721281738, ls=':', c='g')
plt.errorbar(range(len(rw_mean)), rw_mean, rw_std)
plt.grid()
plt.title('DAgger - Evaluation - Humanoid-v2')
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.legend(['Behavior Cloning', 'Expert Policy', 'Dagger'])
# plt.yticks(np.arange(1.350, 3.86, .250))
plt.savefig('dagger-rw-rollouts.pdf', format='pdf')
plt.show()
