from __future__ import annotations
import torch

'''
quaternion representation and multiplication in torch torch.tensors
'''
'''
quaternion rules
i^2 = j^2 = k^2 = ijk = -1

i = jk = -kj
j = ki = -ik
k = ij = - ji
'''

'''
# multiplication matrix m_p representing quaternion p
# the product of m_p and quaternion q in vector form,
# is the equivalent of multiplying p and q

for p = w + x i + y j + z k
m_p = 
w   -x  -y  -z
x   w   -z  y
y   z   w   -x
z   -y  x   w

this process is split into 3 steps
1. creating each row by copying and moving q's dimensions
2. appending the created rows
3. negating the right values
used for quaternion multiplication
'''

multip_positional_matrix = torch.tensor([
    [0,1,2,3],
    [1,0,3,2],
    [2,3,0,1],
    [3,2,1,0]
], dtype=torch.int64)

multip_negation_matrix = torch.tensor([
    [1,-1,-1,-1],
    [1,1,-1,1],
    [1,1,1,-1],
    [1,-1,1,1]
])

# conjugate
# use element wise multiplication any quaternion tensor for its conjugate
conjugate_factor = torch.tensor([1,-1,-1,-1])


class Quaternion:
    '''
    quaternion
    if initialized with torch.Tensor, tensor should have shape of (4,1)
    '''
    def __init__(self, w: int=0, i: int=0, j: int=0, k: int=0, 
                 tensor: torch.Tensor = None):
        if tensor == None:
            self.tensor = torch.tensor([w,i,j,k], dtype=torch.float32)
            return
        self.tensor = tensor.clone().detach()
        
    def multiply(self, q: Quaternion) -> Quaternion:
        '''
        quaternion multiplication p x q, where p is this quaternion
        and q is another
        we transform p into a matrix m_p, then multiply m_p x q
        '''
        
        # source data: create 4x4 matrix, where each row is a copy of p
        p_4x_stack = torch.stack([self.tensor.data] * 4)
        
        # create a new 4x4 matrix, with the same data type
        placeholder = torch.zeros(4,4,dtype=p_4x_stack.dtype
                                # shuffle around internally the positions 
                                # (see positional matrix docstring for details)
                                   ).scatter_(1, multip_positional_matrix, p_4x_stack)

        # negate where required
        placeholder = placeholder.mul(multip_negation_matrix)
        
        # multiple m_p * q
        r = placeholder.matmul(q.tensor)
        return Quaternion(tensor = r)
        
    
    def inverse(self) -> Quaternion:
        '''
        the inverse of a quaternion can be found with the following
        q* / ||q||^2
        
        (the conjugate divided by the norm squared)
        '''
        
        # find the quaternion conjugate
        q_conj = self.tensor.mul(conjugate_factor)
        # find the sqaure of the norm
        normsq = torch.linalg.norm(self.tensor).square()
        return Quaternion(tensor=q_conj.div_(normsq))
    
        
        
