import numpy as np
from scipy.linalg import eigh

def create_lossObject(loss_name, A, b, args):
    if loss_name == "ls":
        return Least_Square(A, b, args)

        
###########################
# 2. Least_Square
###########################
class Least_Square:
  def __init__(self, A, b, args):
    self.ylabel = "bp loss: $f(x) = \\frac{1}{2} x^T A x - b^Tx$"
    self.L2 = args["L2"]
    self.n_params = A.shape[1]

    assert self.L2 == 0

    self.lipschitz = np.sum(A ** 2, axis=0) + self.L2

  def f_func(self, x, A, b):
    reg = 0.5 * self.L2 * np.sum(x ** 2) 

    b_pred = np.dot(A, x) 
    
    loss = 0.5 * np.sum((b_pred - b)**2) + reg

    return loss

  def g_func(self, x, A, b, block=None):
    b_pred = np.dot(A, x)
    residual = b_pred - b

    if block is None:
      grad_persample = np.multiply(A.T,residual)
      tmp = self.L2 *  x
      grad_persample += tmp[:,np.newaxis]
      grad = np.sum(grad_persample,1)
    
    else:
      grad_persample = np.multiply(A[:, block].T,residual)
      tmp = self.L2 *  x[block]
      grad_persample += tmp[:,np.newaxis]
      grad = np.sum(grad_persample,1) 

    return grad, grad_persample

  def h_func(self, x, A, b, block=None):
    #b_pred = np.dot(A, x)

    if block is None:
      hessian = np.dot(A.T, A)
      hessian += self.L2 * np.identity(self.n_params)
      
    elif block.size == 1:
      #import ipdb; ipdb.set_trace()
      hessian = np.sum(A[:, block[0]]**2)      
      hessian += self.L2
      
    else:
      # Block case
      hessian = np.dot(A[:, block].T, A[:, block])
      hessian += self.L2 * np.identity(block.size)

    return hessian 

  def Lb_func(self, x, A, b, block=None):
    if block is None:
      E = np.linalg.eig(A.T.dot(A))[0]
      L_block = np.max(E) + self.L2
    else:
      A_b = A[:, block]

      E = np.linalg.eig(A_b.T.dot(A_b))[0]
      L_block = np.max(E) + self.L2
    
    return L_block

  def Hb_func(self, x, A, b, block=None):
    if block is None:
      L_block = np.dot(A.T, A)
      L_block += self.L2 * np.identity(self.n_params)
    else:
      A_b = A[:, block]
      L_block = A_b.T.dot(A_b) + self.L2 * np.identity(block.size)
    
    return L_block