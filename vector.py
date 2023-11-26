from __future__ import annotations
import torch
from .quaternion import Quaternion

# x,y,z -> i,j,k in a quaternion
vector_quaternion_indices = torch.tensor([1,2,3])
# pi
torch.half_pi = torch.acos(torch.zeros(1)).item()

class Vector:
    def __init__(self, x=0, y=0, z=0, tensor=None):
        if tensor == None:
            self.tensor = torch.tensor([x, y, z], dtype=torch.float32)
            return
        self.tensor = tensor.clone().detach()
        
    def rotate(self, rotor: Vector):
        '''
        The resulting position vector when this vector is rotated around
        the rotor, by the magnitude * pi radians
        ex: a rotor [1,0,0] rotates this vector 180 degrees around the 
        x axis
        
        resulting vector is returned as a new Vector 
        '''
        
        # p is the vector being rotated, represented as a quaternion
        # (x,y,z) -> 0+ xi + yj + zk
        p = Quaternion(tensor = torch.zeros(4, dtype=self.tensor.dtype).scatter_(
                0, vector_quaternion_indices,self.tensor))
        
        # q is the rotation quaternion
        mag = rotor.tensor.norm() # magnitude of rotation
        direction = rotor.tensor.div(mag) # direction unit vector of rotation
        mag.mul_(torch.half_pi) # we want to rotate by mag pi radians
        # quaternion rotation requires sin(theta/2) and cos(theta/2),
        # so we use pi/2
        direction.mul_(mag.sin())
        
        w = mag.cos().unsqueeze(0)
        q = Quaternion(tensor=torch.cat((w, direction)))
        
        return Vector(tensor = q.multiply(p).multiply(q.inverse()).tensor[1:])
    
def rotateVectors(*vectors):
    '''test function to make sure its working properly
    rotateVectors(vectors as [x,y,z] to rotate)
    ex
    rotateVectors([0,1,0], [0.5, 0,0]) returns [0,0,1]
    '''
    rotatedVector = Vector(*vectors[0])
    vectors = vectors[1:]
    for vector in vectors:
        rotatedVector = rotatedVector.rotate(Vector(*vector))
        
    return rotatedVector.tensor