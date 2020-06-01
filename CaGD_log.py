# THIS LIBRARY CONTATINS THE ALGORITHMS EXPLAINED IN THE WORK 
# "Acceleration of Descent-based Optimization Algorithms via Caratheodory's Theorem"

####################################################################################
# 
# This library is focused only on the development of the Caratheodory accelerated 
# algorithms in the case of the logistic regression.
# 
# In general X represents the data/features, Y the labels and theta_0 the initial 
# parameters. It returns theta (the desired argmin) and other variables to reconstruct
# the history of the algorithm.

# We can split the functions into two groups GD and BCD functions, of which 
# then there is another subdivision: CA accelerated (*_CA) vs not accelerated.
# 
# The structure of the accelerated functions is this:
#   - the *_CA functions is the outer while of the algorithms described in 
#     the cited work
#   - the *_mod functions represent the inner while, i.e. where we use the 
#     the reduced measure
# 
####################################################################################

import numpy as np
import copy, timeit
import recombination as rb

# Dlog_L compute the gradient using the logistic model.

def Dlog_L(X,Y,theta,mu, idx="all"):
    h_theta = np.dot(theta,np.transpose(X))
    h_theta = 1+np.exp(-h_theta)
    h_theta = np.divide(1,h_theta)
    h_theta = np.expand_dims(h_theta-Y, -1)
    if idx=="all":
        D = np.multiply(h_theta, X)
    else:
        D = np.multiply(h_theta, X[:,idx])
    return(np.multiply(D,mu[np.newaxis].T)) 

# GD SECTION

def gd_log(X,Y,theta_0,step=1e-3,error=1e-4,max_iter=5e3,mu="Uniform"):
    
    # it runs the gradient descent method
    # for a logistic model (2 classes)
    
    tic = timeit.default_timer()
    iteration = 0
    gr_norm = np.Inf
    theta = np.array(theta_0)[np.newaxis]
    if mu=="Uniform":
        mu = np.array([1/np.shape(X)[0]])

    while (gr_norm>error and iteration<max_iter):
        gr = Dlog_L(X,Y,theta[-1,:],mu)
        gr_mean = np.sum(gr,0)
        theta = np.append(theta, [theta[-1,:]-step*gr_mean],0)
        gr_norm = np.power(gr_mean,2)
        gr_norm = np.sum(gr_norm)**0.5
        iteration += 1
        if iteration%100==0:
            print("iteration ", iteration, " | norm of the gradient = ", gr_norm)
    
    toc = timeit.default_timer()-tic
    return(gr_norm,iteration,theta,toc) 

def gd_log_mod(X,Y,theta_0,mu,gradient_mean,step=1e-3,error=1e-4,max_iter=5e3,H=0.5):

    # it runs the acceleration of the algorithm, i.e. the inner while of the descibed Algorithms

    # mu = mu hat
    # gradient_mean = the expecatation of the gradient computed using the wieght used to reduce the measure
    # H = approximation of the second derivative

    tic = timeit.default_timer()
    iteration = 0
    gr_mean_norm = np.Inf
    theta = np.array(theta_0)[np.newaxis]
    b = True
    tmp = np.Inf # this means that the first step is always good

    while b:
        gr = Dlog_L(X,Y,theta[-1,:],mu)
        gr_mean = np.sum(gr,0)
        theta = np.append(theta, [theta[-1,:]-step*gr_mean],0)
        iteration += 1

        # tmp_1 = Delta using the notation the presented work
        vec = theta[-1,:]-theta_0
        tmp_1 = np.dot(gradient_mean,vec)
        tmp_1 += 0.5*np.dot(np.dot(vec,H),vec)
        
        b = np.around(tmp_1-tmp,6)<0
        tmp = tmp_1
        b = b and iteration<max_iter
       
    toc = timeit.default_timer()-tic
    
    if iteration == max_iter:
        return(gr_mean_norm,iteration,theta,toc)
    else:
        return(gr_mean_norm,iteration,theta[:-1,:],toc)

def gd_log_CA(X,Y,theta_0,step=1e-3,error=1e-4,max_iter=5e3,
              max_iter_CA=0,DEBUG=False):
    
    # it runs the gd_log with the accelration via the Caratheodory Theorem,
    # using gd_log_mod

    tic = timeit.default_timer()
    N, n = X.shape
    iteration = 0
    gradient_mean_norm = error+1
    theta = np.array(theta_0)[np.newaxis]
    mu = np.array([1/np.shape(X)[0]])

    idx_star = []
    n_GD = 2    # number of times we want to repeat the plain GD
                # two to compute the approximation of the Hessian
    gradient_mean = np.zeros((n_GD,np.shape(X)[1]))

    while (gradient_mean_norm > error and iteration < max_iter):
        for i in range(n_GD):
            gradient = Dlog_L(X,Y,theta[-1,:],mu)
            gradient_mean[i,:] = np.sum(gradient,0)
            gradient_mean_norm = np.linalg.norm(gradient_mean)
            if i==0:

                if iteration>=1:
                    print("iteration ", iteration, 
                      " | norm of the gradient after the acceleration procedure = ", 
                      gradient_mean_norm)

                theta = np.append(theta, [theta[-1,:]-step*gradient_mean[i,:]],0)
            
            if i==1:
                print("iteration ", iteration, 
                      " | norm of the gradient after a full iteration = ", 
                      gradient_mean_norm)
            
            iteration += 1
    
        if (gradient_mean_norm > error and iteration < max_iter):
            if DEBUG:
                print("Start Caratheodory")
            
            # maximum number of iterations n*100
            w_star, idx_star, _, _, ERR_recomb = rb.recomb_log(gradient,n*100)[0:5]            
            
            if ERR_recomb == 2:
                print("############ Error: recombination procedure needs more iterations")
                continue
            elif ERR_recomb != 0:
                print("############ Error: Numerical insabilty and/or Strong depedndence between the data")
                continue
            
            x_star = X[idx_star,:]
            y_star = Y[idx_star]

            if DEBUG and (np.all(y_star == 1) or np.all(y_star == 0)):
                print("semi-problem CA measure, y_star all 0/1")
            
            vec = theta[-1,:]-theta[-2,:]
            vec = vec[np.newaxis]
            gr = gradient_mean[1,:]-gradient_mean[0,:]
            gr = gr[np.newaxis]
            H = np.dot(np.reciprocal(vec).T,gr)
            
            if max_iter_CA == 0:
                max_iter_CA = min(int(10/step),10000)

            iteration_CA, theta_CA = gd_log_mod(x_star,y_star,theta[-1,:],
                                                w_star,gradient_mean[1,:],
                                                step,error,max_iter_CA,H)[1:3]

            print("iterations done with the reduced measure = ", iteration_CA)
            if DEBUG:
                print("End Carateodory")

            iteration += iteration_CA*len(idx_star)/N         
            theta = np.append(theta,theta_CA,0)
    
    toc = timeit.default_timer()-tic
    return(gradient_mean_norm,iteration,theta,toc)

# BCD SECTION
# 
# BCD_log was used only as a test and has not been used for the work, 
# we have left it for experimental purposes.

def BCD_log(X,Y,theta_0,step=1e-3,error=1e-4,max_iter=5e3,size_block=5):

    # it runs the BCD method using the GS rule
    # for a logistic model (2 classes) 

    tic = timeit.default_timer()
    iteration = 0
    gr_norm = np.Inf
    theta = np.array(theta_0)[np.newaxis]
    mu = np.array([1./np.shape(X)[0]])
    if len(theta_0)<=size_block:
        return gd_log(X,Y,theta_0,step,error,max_iter,mu)

    maximum_percentage_gr = 0.75

    while (gr_norm>error and iteration<max_iter):
        gr = Dlog_L(X,Y,theta[-1,:],mu)
        gr_mean = np.sum(gr,0)
        theta = np.append(theta, [theta[-1,:]-step*gr_mean],0)
        
        gr_norm = np.power(gr_mean,2)
        gr_norm = np.sum(gr_norm)**0.5
        
        iteration += 1
        
        # split the coordinates in groups using GS rule
        idx_sorted = np.argsort(np.abs(gr_mean))
        idx_sorted = np.flip(idx_sorted)
        cumsum = np.cumsum(np.abs(gr_mean[idx_sorted]))
        cumsum /= np.sum(np.abs(gr_mean))
        idx_sorted = idx_sorted[cumsum<=maximum_percentage_gr]
        n_blocks = int(np.ceil(len(idx_sorted)/size_block))
        
        for i in range(n_blocks):
            j = i*size_block
            if i!=n_blocks-1:
                idx = idx_sorted[j:j+size_block]
            else:
                idx = idx_sorted[j:]

            gr_tmp = Dlog_L(X,Y,theta[-1,:],mu,idx)
            gr_mean_tmp = np.zeros(len(theta_0))
            gr_mean_tmp[idx] = np.sum(gr_tmp,0)

            theta = np.append(theta, [theta[-1,:]-step*gr_mean_tmp],0)
            iteration += 1

        if iteration%50==0:
            print("iteration ", iteration, " | norm of the gradient = ", gr_norm)
    
    toc = timeit.default_timer()-tic
    return(gr_norm,iteration,theta,toc) 

def BCD_log_CA(X,Y,theta_0,step=1e-3,error=1e-4,max_iter=5e3,size_block=5,
               max_iter_CA=0,DEBUG=False):

    # it runs the BCD_log with the accelration via the Caratheodory Theorem,
    # using BCD_log_mod

    tic = timeit.default_timer()
    iteration = 0
    gradient_mean_norm = error+1
    theta = np.array(theta_0)[np.newaxis]
    N = np.shape(X)[0]
    mu = np.array([1/N])

    n_GD = 2    # number of times we want to repeat the plain GD
                # two to compute the approximation of the Hessian

    if len(theta_0)<=size_block:
        return gd_log_CA(X,Y,theta_0,step,error,max_iter)
    
    maximum_percentage_gr = 0.75
    gradient_mean = np.zeros((n_GD,np.shape(X)[1]))

    while (gradient_mean_norm > error and iteration < max_iter):
        for i in range(n_GD):
            gradient = Dlog_L(X,Y,theta[-1,:],mu)
            gradient_mean[i,:] = np.sum(gradient,0)
            gradient_mean_norm = np.linalg.norm(gradient_mean)

            if i==0:

                if iteration>=1:
                    print("iteration ", iteration, 
                      " | norm of the gradient after the acceleration procedure = ", 
                      gradient_mean_norm)

                theta = np.append(theta, [theta[-1,:]-step*gradient_mean[i,:]],0)
            
            if i==1:
                print("iteration ", iteration, 
                      " | norm of the gradient after a full iteration = ", 
                      gradient_mean_norm)
            
            iteration += 1

        # split the coordinates in groups using GS rule
        idx_sorted = np.argsort(np.abs(gradient_mean[-1,:]))
        idx_sorted = np.flip(idx_sorted)
        cumsum = np.cumsum(np.abs(gradient_mean[-1,idx_sorted]))
        cumsum /= np.sum(np.abs(gradient_mean[-1,:]))
        idx_sorted = idx_sorted[cumsum<=maximum_percentage_gr]
        n_blocks = int(np.ceil(len(idx_sorted)/size_block))

        if gradient_mean_norm > error:
            
            start = 0
            for i in range(n_blocks):
                end = start+size_block
                if end>len(idx_sorted):
                    end = len(idx_sorted)
                idx = idx_sorted[start:end]
                start = end
                
                if DEBUG:
                    print("Start Caratheodory")
                
                tmp = gradient[:,idx] - gradient_mean[-1,idx]
                
                ##########################
                # CHECK independence
                ##########################
                check = np.all(tmp == 0,0)
                check = np.arange(len(idx))[check]
                tmp = np.delete(tmp, check,1)

                # max muber of iterations for the recomb procedure = len(idx)*100
                w_star, idx_star, _, _, ERR_recomb = rb.recomb_log(tmp,len(idx)*100)[0:5]

                if ERR_recomb == 2:
                    print("############ Error: recombination procedure needs more iterations")
                    continue
                elif ERR_recomb != 0:
                    print("############ Error: Numerical insabilty and/or Strong depedndence between the data")
                    continue

                x_star = X[idx_star,:]
                y_star = Y[idx_star]

                if DEBUG and (np.all(y_star == 1) or np.all(y_star == 0)):
                    print("semi-problem CA measure, y_star all 0/1")

                vec = theta[-1,idx]-theta[-2,idx]
                vec = vec[np.newaxis]
                gr = gradient_mean[-1,idx]-gradient_mean[-2,idx]
                gr = gr[np.newaxis]
                H = np.dot(gr.T,np.reciprocal(vec))
                
                if max_iter_CA == 0:
                    max_iter_CA = min(int(10/step),10000)
                iteration_CA, theta_CA = BCD_log_mod(x_star,y_star,theta[-1,:],
                                                    w_star,gradient_mean[-1,idx],idx,#np.linalg.norm(gr),
                                                    step,error,max_iter_CA,H)[1:3]

                print("iterations done with the reduced measure = ", iteration_CA)
                if DEBUG:
                    print("End Carateodory")

                iteration += iteration_CA*len(idx_star)/N
                theta = np.append(theta,theta_CA,0)
    
    toc = timeit.default_timer()-tic
    return(gradient_mean_norm,iteration,theta,toc)

def BCD_log_mod(X,Y,theta_0,mu,gradient_mean,idx,
            step=1e-3,error=1e-4,max_iter=5e3,H=0.5):

    # it runs the acceleration of the algorithm, i.e. the inner while of the descibed Algorithm

    # mu = mu hat
    # gradient_mean = the expecatation of the gradient computed using the wieght used to reduce the measure
    # H = approximation of the second derivative

    tic = timeit.default_timer()
    iteration = 0
    gr_mean_norm = np.Inf
    theta = np.array(theta_0)[np.newaxis]
    b = True
    tmp = np.Inf # this means that the first step is always good
    gr_mean = np.zeros(len(theta_0))
    while b:
        gr = Dlog_L(X,Y,theta[-1,:],mu,idx)
        gr_mean[idx] = np.sum(gr,0)
        theta = np.append(theta, [theta[-1,:]-step*gr_mean],0)
    
        iteration += 1

        vec = theta[-1,idx]-theta_0[idx]
        # tmp_1 = Delta using the notation the presented work
        tmp_1 = np.dot(gradient_mean,vec)
        tmp_1 += 0.5*np.dot(np.dot(vec,H),vec)

        b = np.around(tmp_1-tmp,6)<0
        tmp = tmp_1
        b = b and iteration<max_iter
    
    toc = timeit.default_timer()-tic
    if iteration == max_iter:
        return(gr_mean_norm,iteration,theta,toc)
    else:
        return(gr_mean_norm,iteration,theta[:-1,:],toc)

