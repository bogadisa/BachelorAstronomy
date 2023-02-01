import torch

import numpy as np

a=torch.tensor(0.25*np.ones((2),dtype=np.float32) ,requires_grad=True)

b=torch.tensor( 2*np.ones((2),dtype=np.float32),requires_grad=True)

c=torch.tensor( 3*np.ones((2),dtype=np.float32),requires_grad=True)


d=a*b                                                                       #    
e=torch.dot ( d,c)  #      ( [0.25,0.25]*[2,2] ) dot [3,3] =   [0.5,0.5]  dot [3,3] = [ 0.25    *2 *3 +  0.25*2 *3] = [3]




print('e.requires_grad? ',e.requires_grad)
print(e)
print(e.grad_fn)

da,db=torch.autograd.grad(e,[a,b]) # computes gradients for a,b ONLY in return arguments
#requires_grad must be set anyway for it to work


print( 'de/dc', c.grad  ) #None
print( 'de/db', b.grad  )
print( 'de/da', a.grad  )

#   [a[0,0]   * b[0,0] * c[0,0] +  a[0,1] * b[0,1] * c[0,1] ]
# e=[ 0.25    *  2     *  3 +       0.25  *   2    *    3   ]

# e= a*K --> de/da[0,0] = K
#            de/da[0,0] = 2 * 3


print( ' ')
print( ' ')
print( ' ')

print( 'input value of tensor a as numpy: a.data.numpy() ',  a.data.numpy()  )
print( 'de/da as numpy: a.grad.numpy() ', da.data.numpy()  )



