import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = np.loadtxt("channel_current.txt")

tau = np.array([0.2, 0.45, 0.7, 0.95, 1.2, 1.45, 1.7, 1.95, 2.2])

current_threshold = -0.5

def P_exp(data, tau, current_threshold):
    Z = data.shape[0]
    closed = np.where(data < current_threshold, data, 0)
    P0 = np.zeros(len(tau)) + np.count_nonzero(closed, axis=0)
    return P0/Z
    

P0 = P_exp(data, tau, current_threshold)

plt.title(f"Current threshold {current_threshold}pN")
plt.plot(tau, P0)
plt.ylabel("Probability")
plt.xlabel(r"$\tau$ [pN/nm]")
plt.show()

delta_epsilon = -5 #k_B T
delta_epsilon_std = 1.1

def kT_to_pNnm(x):
    return x*4.114

delta_epsilon = kT_to_pNnm(delta_epsilon) #pN nm
delta_epsilon_std = kT_to_pNnm(delta_epsilon_std) #pN nm

delta_A = -10 #nm^2

def P_model(delta_epsilon, tau, A):
    return 1/(1 + np.exp(-1/kT_to_pNnm(1)*(delta_epsilon - tau*A)))

R2_i = 2

P0_model = P_model(delta_epsilon, tau, R2_i*delta_A)
P0_model_std_min = P_model(delta_epsilon-delta_epsilon_std, tau, R2_i*delta_A)
P0_model_std_max = P_model(delta_epsilon+delta_epsilon_std, tau, R2_i*delta_A)

plt.title(f"Current threshold {current_threshold}pN")
plt.plot(tau, P0_model, label=rf"Model $\Delta A={R2_i*delta_A}nm^2$")
plt.fill_between(tau, P0_model_std_min, P0_model_std_max, alpha=.3)
plt.plot(tau, P0, label="experimental")
plt.ylabel("Probability")
plt.xlabel(r"$\tau$ [pN/nm]")
plt.legend()
plt.show()


# low_R2 = -9999 
# R2_i = 1
# for i in range(1, 10):
#     P_i = P_model(delta_epsilon, tau, i*delta_A)
#     # print(P_i)
#     R2 = r2_score(P0, P_i)
#     if R2 > low_R2:
#         low_R2 = R2
#         R2_i = i