import torch

import numpy as np

def print_graph(g, level=0):
    if g == None: return
    print('*'*level*1, g)
    for subg in g.next_functions:
        print_graph(subg[0], level+1)




a=torch.ones((2,1),requires_grad=True)

b=torch.tensor( 2*np.ones((2,1),dtype=np.float32),requires_grad=True)

c=torch.tensor( 3*np.ones((2,1),dtype=np.float32),requires_grad=True)


d=a+b
e=d*c
print(e)


print_graph(e.grad_fn, 0)
