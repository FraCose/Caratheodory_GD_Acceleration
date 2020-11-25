# THIS REPOSITORY CONTATINS THE ALGORITHMS EXPLAINED IN THE WORK
# Cosentino, Oberhauser, Abate - "Caratheodory Sampling for Stochastic Gradient Descent" 

# ####################################################################################
# 
# This file contains functions necessary to use the selection rules from  
# https://github.com/IssamLaradji/BlockCoordinateDescent.
# 
# In particular, the function:
#       - update_Caratheodory is necessary to run the Caratheodory Sampling Procedure
#       - recomb_step & Caratheodory_Acceleration support the update_Caratheodory 
#         function
# 
# For more information please read the cited works and/or the README file.
# 
####################################################################################

import numpy as np
from . import line_search
import cvxopt

from cvxopt import solvers, matrix, spmatrix
from scipy import sparse
from cvxopt import matrix, solvers
from scipy.optimize import minimize

import recombination as rb

cvxopt.solvers.options['show_progress'] = False


def update_Caratheodory(rule, x, A, b, loss, args, block, iteration):
  f_func = loss.f_func
  g_func = loss.g_func
  h_func = loss.h_func
  lipschitz = loss.lipschitz

  block_size = block.size
  param_size = x.size

  if rule in ["quadraticEg", "Lb"]:    
    """This computes the eigen values of the lipschitz values corresponding to the block"""

    G, G_persample = g_func(x, A, b, block)
    L_block = loss.Lb_func(x, A, b, block)
    d = - G / L_block
    factor = 1 / L_block
    w_star, idx_star, _ = recomb_step(G / A.shape[0], G_persample)
    w_star *= A.shape[0]
    x_tm1 = np.copy(x[block])
    iter_ca, x[block] = Caratheodory_Acceleration(A[idx_star[:,None],block], 
                                  b[idx_star], w_star,
                                  h_func(x_tm1, A, b, block),factor,
                                  x_tm1, G, 
                                  args["L2"], args["L1"])

  elif rule in ["newtonUpperBound", "Hb"]:    

    G, G_persample = g_func(x, A, b, block)
    H = loss.Hb_func(x, A, b, block)
    d = - np.linalg.pinv(H).dot(G)
    
    factor = np.linalg.pinv(H)
    w_star, idx_star, _ = recomb_step(G / A.shape[0], G_persample)
    w_star = w_star * A.shape[0]

    x_tm1 = np.copy(x[block])

    iter_ca, x[block] = Caratheodory_Acceleration(A[idx_star[:,None],block],b[idx_star],w_star,
                                  H,factor,
                                  x_tm1, G, 
                                  args["L2"], args["L1"])

  elif rule == "LA":
    G, G_persample = g_func(x, A, b, block)
    L_block =loss.Lb_func(x, A, b, block)
    Lb = np.max(args["LA_lipschitz"][block])
    
    while True:
      x_new = x.copy()
      x_new[block] = x_new[block] - G / Lb

      RHS = f_func(x,A,b) - (1./(2. * Lb)) * (G**2).sum()
      LHS = f_func(x_new,A,b)
      
      if LHS <= RHS:
        break

      Lb *= 2.

    args["LA_lipschitz"][block] = Lb
    
    d = - G / Lb
    factor = 1 / Lb
    w_star, idx_star, _ = recomb_step(G / A.shape[0], G_persample)
    w_star = w_star * A.shape[0]
    x_tm1 = np.copy(x[block])

    iter_ca, x[block] = Caratheodory_Acceleration(A[idx_star[:,None],block],b[idx_star],w_star,
                                  h_func(x_tm1, A, b, block),factor,
                                  x_tm1, G, 
                                  args["L2"], args["L1"])

  ### Constrained update rules
  elif rule in ["Lb-NN"]:
    G, G_persample = g_func(x, A, b, block)
    L_block =loss.Lb_func(x, A, b, block)
    d = - G / L_block
    factor = 1 / L_block
    w_star, idx_star, _ = recomb_step(G / A.shape[0], G_persample)
    w_star = w_star * A.shape[0]

    x_tm1 = np.copy(x[block])

    iter_ca, x[block] = Caratheodory_Acceleration(A[idx_star[:,None],block],b[idx_star],w_star,
                                  h_func(x, A, b, block),factor,
                                  x_tm1, G, 
                                  args["L2"], args["L1"], True)

  elif rule == "TMP-NN":
    L = lipschitz[block]

    grad_list, G_persample = g_func(x, A, b, block)
    hess_list = h_func(x, A, b, block)

    H = np.zeros((block_size, block_size))
    G = np.zeros(block_size) 

    # The active set is on the bound close to x=0
    active = np.logical_and(x[block] < 1e-4, grad_list > 0)
    work = np.logical_not(active)

    # active
    ai = np.where(active == 1)[0]
    gA = grad_list[active]

    G[ai] = gA / (np.sum(L[active]))
    H[np.ix_(ai, ai)] = np.eye(ai.size)
    # work set
    wi = np.where(work == 1)[0]

    gW = grad_list[work]
    hW = hess_list[work][:, work]

    G[wi] = gW
    H[np.ix_(wi, wi)] = hW

    # Perform Line search
    alpha = 1.0
    
    u_func = lambda alpha: (- alpha * np.dot(np.linalg.inv(H), G))
    f_simple = lambda x: f_func(x, A, b, assert_nn=0)

    alpha = line_search.perform_line_search(x.copy(), G, 
                              block, f_simple, u_func, alpha0=1.0,
                                proj=lambda x: np.maximum(0, x))

    factor = alpha * np.linalg.inv(H)
    w_star, idx_star, _ = recomb_step(G / A.shape[0], G_persample)
    w_star = w_star * A.shape[0]
    x_tm1 = np.copy(x[block])
    iter_ca, x[block] = Caratheodory_Acceleration(A[idx_star[:,None],block],b[idx_star],w_star,
                                  hess_list,factor, #x_tm1, 
                                  x_tm1, grad_list, 
                                  args["L2"], args["L1"], True)
                                  
  elif rule == "qp-nn":
    cvxopt.setseed(1)
    non_block = np.delete(np.arange(param_size), block)
    k = block.size

    # 0.5*xb ^T (Ab^T Ab) xb + xb^T[Ab^T (Ac xc - b) + lambda*ones(nb)]
    Ab = matrix(A[:, block])
    bb = matrix(A[:, non_block].dot(x[non_block]) - b)

    P = Ab.T*Ab
    q = (Ab.T*bb + args["L1"]*matrix(np.ones(k)))

    G = matrix(-np.eye(k))
    h = matrix(np.zeros(k))
    x_block = np.array(solvers.qp(P=P, q=q, 
                                G=G, h=h, solver = "glpk")['x']).ravel()

    # cvxopt.solvers.options['maxiters'] = 1000
    cvxopt.solvers.options['abstol'] = 1e-16
    cvxopt.solvers.options['reltol'] = 1e-16
    cvxopt.solvers.options['feastol'] = 1e-16

    x_old = x.copy()
    d = x_block - x_old[block]

    factor = 1.
    G, G_persample = g_func(x, A, b, block)
    w_star, idx_star, ERR = recomb_step(d/ A.shape[0], G_persample)
    if ERR !=0:
      iter_ca = 0
      return x, args, iter_ca
    
    w_star = w_star * A.shape[0]
    x_tm1 = np.copy(x[block])

    iter_ca, x[block] = Caratheodory_Acceleration(A[idx_star[:,None],block],b[idx_star],w_star,
                                  h_func(x, A, b, block),factor, #x_tm1, 
                                  x_tm1, G, 
                                  args["L2"], args["L1"], True)

  else:
    print(("update rule %s doesn't exist" % rule))
    raise

  return x, args, iter_ca

def recomb_step(d,G_persample):

  check_rows = np.any(G_persample.T != 0,1)
  check_rows = np.arange(G_persample.T.shape[0])[check_rows]
  
  check_rows_0 = np.all(G_persample.T == 0,1)
  check_rows_0 = np.arange(G_persample.T.shape[0])[check_rows_0]
  check_rows_0 = check_rows_0[0]

  check_rows = np.append(check_rows,check_rows_0)
  
  recomb_data = G_persample.T[check_rows,:] - d

  # delete zeros columns
  check_col = np.all(recomb_data == 0,0)
  check_col = np.arange(d.size)[check_col]
  recomb_data = np.delete(recomb_data, check_col, 1)
  
  w_star, idx_star, _, _, ERR_recomb = rb.recomb_combined(np.copy(recomb_data))[0:5]
  idx_star = check_rows[idx_star]
  w_star = w_star/G_persample.T.shape[0]*check_rows.size
  return w_star, idx_star, ERR_recomb

def Caratheodory_Acceleration(a_star,b_star,w_star,
                              hess, factor,
                              x_0,gr_0, 
                              L2, L1,
                              constrained = False, max_iter=1e2):
  
  iteration_CA = 0
  x = np.copy(x_0)
  lr = 1e-2 #1e-3
  condition = True

  while condition:

    iteration_CA += 1
    x_tm1 = np.copy(x)

    if iteration_CA > 1:
      error_persample = np.dot(a_star,x)-b_star
      g_persample = np.multiply(a_star.T,error_persample[np.newaxis])

      if L1 != 0 or L1 == None:
        g_persample += L1

      d = np.sum(g_persample * w_star,1)
      d = np.dot(factor, d)
      x -= lr * d
    else: 
      x -= np.dot(factor, gr_0)

    if constrained:
      x = np.maximum(x, 0.)
    
    Delta_x = x-x_0
    # tmp_1 = Delta using the notation of the pdf
    tmp_1 = np.dot(gr_0,Delta_x)
    tmp_1 += 0.5 * np.dot(Delta_x,np.dot(hess,Delta_x))

    if iteration_CA == 1:
      condition = tmp_1<0
    else:
      condition = tmp_1 < tmp and iteration_CA<max_iter
    
    tmp = tmp_1

  if iteration_CA == max_iter:
    return iteration_CA, x 
  else:
    return iteration_CA-1, x_tm1