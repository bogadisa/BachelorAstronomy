import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

R = 8.3144598

mass = [28.97e-3 , 28.97e-3, 28.97e-3]


fairrom = np.array([140, 285, 418, 561, 699, 838, \
976, 1115, 1254, 1394, 1533, 1672, 1812, 1951])
Rairrom = np.array([109.68, 109.65, 109.62, 109.59,\
 109.57, 109.56, 109.49, 109.45, 109.42, 109.37,\
  109.31, 109.26, 109.22, 109.18])

fair50 = np.array([152, 302, 457, 605, 754,\
 903, 1053, 1203, 1354, 1505, 1655, 1805, 1955])
Rair50 = np.array([20.24, 20.24, 20.21, 20.16,\
 20.12, 20.03, 20, 19.96, 19.93, 19.90, 19.85, 19.81, 19.74])

fair70 = np.array([142, 308, 444, 587, 733, 877,\
 1024, 1168, 1314, 1461, 1607, 1751, 1899])
Rair70 = np.array([41.36, 41.27, 41.24, 41.21\
, 41.18, 41.12, 41.09, 41.06, 41.01, 41, 41.02, 41.06, 41.02])


Rlist = [Rairrom, Rair50, Rair70]
flist = [fairrom, fair50, fair70]

alist = np.zeros(3)
astd = np.zeros(3)
clist = np.zeros(3)
cstd = np.zeros(3)

Llist = [[1253e-3, 1.5e-3], [1244e-3,1.5e-3], [1244e-3,1.5e-3]]

def T(Rt):
    r = np.array(Rt) * 1e-2
    k = (25 - 24*np.log(r)) +273
    m = np.mean(k)
    std = np.std(k)
    return m, std

for k in range(3):
    alist[k] = stats.linregress(np.linspace(1,len(flist[k]), len(flist[k])), flist[k])[0]
    astd[k]  = stats.linregress(np.linspace(1,len(flist[k]), len(flist[k])), flist[k])[4]
    clist[k] = alist[k]*2*Llist[k][0]
    cstd[k] = np.sqrt((alist[k]*2*Llist[k][1])**2 + (astd[k]*2*Llist[k][0])**2) #C standrd dev

def degfree():
    freelist = np.zeros(3)
    stdlist = np.zeros(3)
    for j in range(3):
        freelist[j] = 2/(clist[j]**2 * mass[j] / (R * T(Rlist[j])[0]) - 1)
        stdlist[j] = np.sqrt( \
        (-4*mass[j]*R*T(Rlist[j])[0]*clist[j] / \
        (R*T(Rlist[j])[0] -mass[j]*clist[j]**2)**2 \
        *cstd[j])**2 \
        + (2*mass[j]*R*clist[j]**2 / \
        (mass[j]*clist[j]**2 - R*T(Rlist[j])[0])**2 \
        *T(Rlist[j])[1])**2)
    return freelist, stdlist

words = ["20^o C", "50^o C", "70^o C"]

for i in range(3):
    print(f"Frekvens for luft i {T(Rlist[i])[0]-273:.0f}C")

    print(flist[i])

print()
print()
print()
print()

print(cstd)
print(astd)
free, stds = degfree()
print(free)
print(clist)
print(stds)
for i in range(3):
    print(T(Rlist[i])[1])


"""
PLOTTING
"""
words = ["20^o C", "50^o C", "70^o C"]

for n in range(3):
    plt.title(fr"frekvens for luft i ${words[n]}$")
    plt.xlabel("n-te ressonanstop")
    plt.ylabel("frekvens, $v$")
    plt.plot(np.linspace(1, len(flist[n]), len(flist[n])), flist[n])
    plt.plot(np.linspace(1, len(flist[n]), len(flist[n])), flist[n], "r*")
    plt.show()
