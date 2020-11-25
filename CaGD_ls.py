# THIS LIBRARY CONTATINS THE ALGORITHMS EXPLAINED IN THE WORK
# "Acceleration of Descent-based Optimization Algorithms via Caratheodory's Theorem"

####################################################################################
# 
# This library is focused only on the development of the Caratheodory accelerated 
# algorithms in the case of least-square with and without Lasso regularization.
# 
# In general X represents the data/features, Y the labels and theta_0 the initial 
# parameters. It returns theta (the desired argmin) and other variables to reconstruct
# the history of the algorithm.
# 
# We can split the functions into three groups:
#   - ADAM, SAG
#   - BCD algorithm with the Caratheodory Sampling Procedure(CSP).
#     The structure of the accelerated functions is this:
#     a) the *_CA_* functions is the outer while of the algorithms described in 
#     the cited work
#     b) the *_mod_* functions represent the inner while, i.e. where we use the 
#     the reduced measure
#     c) directions_CA_steps_* functions are necessary for the parallelziation  
#     of the code
#   - BCD w/out the Caratheodory Sampling Procedure.
#     The structure of the accelerated functions is this:
#     a) mom_BCD_GS_ls, mom_BCD_random_ls, BCD_GS_ls are the outer while of 
#     the algorithms described in the cited work w/out the CSP
#     b) parallel_BCD_mom, parallel_BCD are necessary for the parallelziation  
#     of the code
#
####################################################################################

import os
import numpy as np
import copy, timeit, psutil
import recombination as rb
from numba import njit, prange
import multiprocessing as mp

###############################################
# ADAM
###############################################

def ADAM_ls(X,Y,theta_0,lambda_LASSO=0.,batch_size=256,
            lr=1e-3,loss_accepted=1e-8,max_iter=1e2):
         
    # it runs the ADAM algorithm specialized in the case of least-square 
    # with a LASSO regularization term.
    # Copied from the original paper

    tic = timeit.default_timer()
    N = np.shape(X)[0]
    iteration = 0.
    loss = loss_accepted+1.
    theta = np.array(theta_0) 

    # Adam Parameter
    beta_1 = 0.9
    beta_2 = 0.999
    eps = 1e-8
    t = 0

    m = np.zeros(np.size(theta_0))
    v = np.zeros(np.size(theta_0))
    m_hat = np.zeros(np.size(theta_0))
    v_hat = np.zeros(np.size(theta_0))

    loss_story = []
    time_story = []
    iteration_story = []

    n_cycles = int(N/batch_size)

    while iteration<=max_iter:
        
        error_persample = np.dot(X,theta)-Y
        error_persample = error_persample[np.newaxis].T
        loss = np.dot(error_persample.T,error_persample)[0,0]/N
        loss += lambda_LASSO * np.abs(theta).sum()
        loss_story.append(loss.item())
        toc = timeit.default_timer()-tic
        time_story.append(toc)
        iteration_story.append(iteration)

        print("iteration = ", int(iteration+0.5), " | loss = ", loss,
              " | time = ",timeit.default_timer()-tic)
        
        idx_shuffled = np.random.choice(N,N, replace=False)

        for i in np.arange(n_cycles):
            t += 1
            idx = idx_shuffled[i*batch_size:i*batch_size+batch_size]
            error_persample = np.dot(X[idx],theta)-Y[idx]
            error_persample = error_persample[np.newaxis].T
            gr = 2*np.matmul(X[idx].T,error_persample)/N
            gr += lambda_LASSO * np.sign(theta).reshape(-1,1)
            m = beta_1*m + (1-beta_1)*gr[:,0]
            v = beta_2*v + (1-beta_2)*np.power(gr[:,0],2)
            m_hat = m/(1-beta_1**t)
            v_hat = v/(1-beta_2**t)
            theta -= lr*m_hat/(np.sqrt(v_hat)+eps) 
        
        iteration += 1
    
    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    loss = np.dot(error_persample.T,error_persample)[0,0]/N
    loss += lambda_LASSO * np.abs(theta).sum()
    loss_story.append(loss.item())
    toc = timeit.default_timer()-tic
    time_story.append(toc)
    iteration_story.append(iteration)
    print("iteration = ", int(iteration+0.5), " | loss = ", loss,
              " | time = ",timeit.default_timer()-tic)

    return (loss_story,iteration_story,theta,time_story)

###############################################
# SAG
# Observation:  the leanring rate must be small
#               or ''more clever strategy''
###############################################

def SAG_ls(X,Y,theta_0,lambda_LASSO=0.,batch_size=256,
            lr=1e-3,loss_accepted=1e-8,max_iter=1e2):
    
    # it runs the SAG algorithm specialized in the case of least-square 
    # with a LASSO regularization term.
    # Copied from the original paper

    tic = timeit.default_timer()
    N, n = np.shape(X)
    iteration = 0.
    loss = loss_accepted+1.
    theta = np.array(theta_0) 

    loss_story = []
    time_story = []
    iteration_story = [] 

    n_cycles = int(N/batch_size)
    gr_persample = np.zeros((N,n))

    while iteration<=max_iter:
        
        error_persample = np.dot(X,theta)-Y
        error_persample = error_persample[np.newaxis].T
        loss = np.dot(error_persample.T,error_persample)[0,0]/N
        loss += lambda_LASSO * np.abs(theta).sum()
        loss_story.append(loss.item())
        toc = timeit.default_timer()-tic
        time_story.append(toc)
        iteration_story.append(iteration)
        print("iteration = ", int(iteration+0.5), " | loss = ", loss,
              " | time = ",timeit.default_timer()-tic)
        
        idx_shuffled = np.random.choice(N,N, replace=False)

        if iteration == 0:
            sum_total = 0.
            for i in range(n_cycles):
                idx = idx_shuffled[i*batch_size:(i+1)*batch_size]
                error_persample = np.dot(X[idx],theta)-Y[idx]
                error_persample = error_persample[np.newaxis].T
                gr_persample[idx,:] = 2*np.multiply(X[idx,:],error_persample)
                gr_persample[idx,:] += lambda_LASSO * np.sign(theta)
                sum_new_idx = np.sum(gr_persample[idx,:],0)
                sum_total += sum_new_idx
                theta -= lr * sum_total/((i+1)*batch_size)
        else:
            for i in range(n_cycles):
                idx = idx_shuffled[i*batch_size:i*batch_size+batch_size]
                sum_old_idx = np.sum(gr_persample[idx,:],0)
                error_persample = np.dot(X[idx],theta)-Y[idx]
                error_persample = error_persample[np.newaxis].T
                gr_persample[idx,:] = 2*np.multiply(X[idx,:],error_persample)
                gr_persample[idx,:] += lambda_LASSO * np.sign(theta)
                sum_new_idx = np.sum(gr_persample[idx,:],0)
                sum_total = sum_total - sum_old_idx + sum_new_idx
                theta -= lr * sum_total/N
        
        iteration += 1
    
    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    loss = np.dot(error_persample.T,error_persample)[0,0]/N
    loss += lambda_LASSO * np.abs(theta).sum()
    loss_story.append(loss.item())
    toc = timeit.default_timer()-tic
    time_story.append(toc)
    iteration_story.append(iteration)
    print("iteration = ", int(iteration+0.5), " | loss = ", loss,
            " | time = ",timeit.default_timer()-tic)

    return (loss_story,iteration_story,theta,time_story)

###############################################
# Momentum_BCD w/out CaratheodorySP
# GS and Random
###############################################

def mom_BCD_GS_ls(X,Y,theta_0,lambda_LASSO=0.,
            lr=1e-3,loss_accepted=1e-8,max_iter=1e2,
            size_block=2,percentage_gr = 0.75):

    # it runs the BCD with momentum ,
    # using mom_CA_BCD_mod and the GS rule for the selection of the blocks

    num_cpus = psutil.cpu_count(logical=False)
    tic = timeit.default_timer()
    N = np.shape(X)[0]
    iteration = 0.
    loss = loss_accepted+1.

    assert np.size(theta_0)>=size_block, "less parameters than size_block, decrease the size block"
    
    # MOMENTUM param
    beta = 0.9
    v = np.zeros(np.size(theta_0))

    theta = np.array(theta_0)

    loss_story = []
    time_story = []
    iteration_story = []

    gr1d = np.empty(np.size(theta_0))
    max_number_blocks = np.infty #8*num_cpus
    to_extract = min(max_number_blocks*size_block,len(theta_0)*percentage_gr)
    to_extract -= to_extract % size_block
    to_extract = int(to_extract)

    while (loss > loss_accepted and iteration < max_iter):
        
        # for i in range(1):

        error_persample = np.dot(X,theta)-Y
        error_persample = error_persample[np.newaxis].T

        if iteration == 0:
            loss = np.dot(error_persample.T,error_persample)[0,0]/N
            loss += lambda_LASSO * np.abs(theta).sum()
            loss_story.append(loss)
            toc = timeit.default_timer()-tic
            time_story.append(toc)
            iteration_story.append(iteration)
            print("iteration = ", int(iteration),
                    " | loss = ", loss,
                    " | time = ", toc)

        gr_persample = 2*np.multiply(X,error_persample)
        gr_persample += lambda_LASSO * np.sign(theta)
        gr1d = np.mean(gr_persample,0)
        
        v = beta*v - lr*gr1d
        # if i == 0:
        theta += v
        iteration += 1

        loss = np.dot(error_persample.T,error_persample)[0,0]/N
        loss += lambda_LASSO * np.abs(theta).sum()
        loss_story.append(loss)
        toc = timeit.default_timer()-tic
        time_story.append(toc)
        iteration_story.append(iteration)
        print("iteration = ", int(iteration),
                " | loss = ", loss,
                " | time = ", toc)

        blocks = building_blocks_cumsum(gr1d,size_block,percentage_gr,max_number_blocks,
                                "sorted") # sorted, random or balanced
        n_blocks = len(blocks)

        if loss > loss_accepted:

            # start parallel part
            manager = mp.Manager()
            results = manager.dict()
            processes = []

            for i in range(n_blocks):
                p = mp.Process(target = parallel_BCD_mom,
                               args = (results,i,X,Y,lambda_LASSO,
                                    blocks[i], # direction_persample[:,blocks[i]],
                                    theta, # theta_tm1,
                                    # gr1d[blocks[i]], # gr1d_tm1[blocks[i]],
                                    # max_iter,
                                    v,
                                    iteration,lr,loss_accepted))
                processes.append(p)
                p.start()

            i = 0
            for process in processes:
                process.join()
                i += 1
            
            # collecting results from the parallel execution
            for i in range(n_blocks):
                # if results[i][3] != 0:
                #     continue
                # if results[i][0] == 1:
                #     continue
                theta[blocks[i]] = results[i][0][blocks[i]]
                v[blocks[i]] = results[i][1][blocks[i]]
            
            iteration += n_blocks #*(len(blocks[i])+1)/N

    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    loss = np.dot(error_persample.T,error_persample)[0,0]/N
    loss += lambda_LASSO * np.abs(theta).sum()
    loss_story.append(loss)
    toc = timeit.default_timer()-tic
    time_story.append(toc)
    iteration_story.append(iteration)
    print("iteration = ", int(iteration),
            " | loss = ", loss,
            " | time = ", toc)
            
    toc = timeit.default_timer()-tic
    return loss_story,iteration_story,theta,time_story

def mom_BCD_random_ls(X,Y,theta_0,lambda_LASSO=0.,
            lr=1e-3,loss_accepted=1e-8,max_iter=1e2,
            size_block=2,percentage_gr = 0.75):

    # it runs the BCD with momentum ,
    # using parallel_BCD_mom and the random rule for the selection of the blocks

    num_cpus = psutil.cpu_count(logical=False)
    tic = timeit.default_timer()
    N = np.shape(X)[0]
    iteration = 0.
    loss = loss_accepted+1.

    assert np.size(theta_0)>=size_block, "less parameters than size_block, decrease the size block"
    
    # MOMENTUM param
    beta = 0.9
    v = np.zeros(np.size(theta_0))

    theta = np.array(theta_0)

    loss_story = []
    time_story = []
    iteration_story = []

    gr1d = np.empty(np.size(theta_0))
    max_number_blocks = np.infty #8*num_cpus
    to_extract = min(max_number_blocks*size_block,len(theta_0)*percentage_gr)
    to_extract -= to_extract % size_block
    to_extract = int(to_extract)

    while (loss > loss_accepted and iteration < max_iter):
        
        # for i in range(1):

        error_persample = np.dot(X,theta)-Y
        error_persample = error_persample[np.newaxis].T

        if iteration == 0:
            loss = np.dot(error_persample.T,error_persample)[0,0]/N
            loss += lambda_LASSO * np.abs(theta).sum()
            loss_story.append(loss)
            toc = timeit.default_timer()-tic
            time_story.append(toc)
            iteration_story.append(iteration)
            print("iteration = ", int(iteration),
                    " | loss = ", loss,
                    " | time = ", toc)

        # gr_persample = 2*np.multiply(X,error_persample)
        # gr_persample += lambda_LASSO * np.sign(theta)
        # gr1d = np.mean(gr_persample,0)
        
        # v = beta*v - lr*gr1d
        # # if i == 0:
        # theta += v
        # iteration += 1

        loss = np.dot(error_persample.T,error_persample)[0,0]/N
        loss += lambda_LASSO * np.abs(theta).sum()
        loss_story.append(loss)
        toc = timeit.default_timer()-tic
        time_story.append(toc)
        iteration_story.append(iteration)
        print("iteration = ", int(iteration),
                " | loss = ", loss,
                " | time = ", toc)

        blocks = np.random.choice(len(theta),to_extract,replace = False)
        n_blocks = len(blocks)

        if loss > loss_accepted:

            # start parallel part
            manager = mp.Manager()
            results = manager.dict()
            processes = []

            for i in range(n_blocks):
                p = mp.Process(target = parallel_BCD_mom,
                               args = (results,i,X,Y,lambda_LASSO,
                                    blocks[i], # direction_persample[:,blocks[i]],
                                    theta, # theta_tm1,
                                    # gr1d[blocks[i]], # gr1d_tm1[blocks[i]],
                                    # max_iter,
                                    v,
                                    iteration,lr,loss_accepted))
                processes.append(p)
                p.start()

            i = 0
            for process in processes:
                process.join()
                i += 1
            
            # collecting results from the parallel execution
            for i in range(n_blocks):
                # if results[i][3] != 0:
                #     continue
                # if results[i][0] == 1:
                #     continue
                theta[blocks[i]] = results[i][0][blocks[i]]
                v[blocks[i]] = results[i][1][blocks[i]]
            
            iteration += n_blocks #*(len(blocks[i])+1)/N

    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    loss = np.dot(error_persample.T,error_persample)[0,0]/N
    loss += lambda_LASSO * np.abs(theta).sum()
    loss_story.append(loss)
    toc = timeit.default_timer()-tic
    time_story.append(toc)
    iteration_story.append(iteration)
    print("iteration = ", int(iteration),
            " | loss = ", loss,
            " | time = ", toc)
            
    toc = timeit.default_timer()-tic
    return loss_story,iteration_story,theta,time_story

def parallel_BCD_mom(results,proc_numb,X,Y,lambda_LASSO,
                    block, # direction_persample[:,blocks[i]],
                    theta, # theta_tm1,
                    # gr1d[blocks[i]], # gr1d_tm1[blocks[i]],
                    # max_iter,
                    v,
                    iteration,lr,loss_accepted):
    beta = 0.9
    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    gr_persample = 2*np.multiply(X,error_persample)
    gr_persample += lambda_LASSO * np.sign(theta)
    gr1d = np.mean(gr_persample,0)
    v = beta*v - lr*gr1d
    # if i == 0:
    theta += v
    print("PID parallel: ", os.getpid(), " | process number ", proc_numb) #," | iterations CA = ", iteration_CA)
    results[proc_numb] = [theta,v]
    return 

###############################################
# BCD w/out momentum GS w/out CaratheodorySP
###############################################

def BCD_GS_ls(X,Y,theta_0,lambda_LASSO=0.,
            lr=1e-3,loss_accepted=1e-8,max_iter=1e2,
            size_block=2,percentage_gr = 0.75):

    # it runs the BCD with momentum ,
    # using mom_CA_BCD_mod and the GS rule for the selection of the blocks

    num_cpus = psutil.cpu_count(logical=False)
    tic = timeit.default_timer()
    N = np.shape(X)[0]
    iteration = 0.
    loss = loss_accepted+1.

    assert np.size(theta_0)>=size_block, "less parameters than size_block, decrease the size block"
    
    # MOMENTUM param
    # beta = 0.9
    # v = np.zeros(np.size(theta_0))

    theta = np.array(theta_0)

    loss_story = []
    time_story = []
    iteration_story = []

    gr1d = np.empty(np.size(theta_0))
    max_number_blocks = np.infty #8*num_cpus
    to_extract = min(max_number_blocks*size_block,len(theta_0)*percentage_gr)
    to_extract -= to_extract % size_block
    to_extract = int(to_extract)

    while (loss > loss_accepted and iteration < max_iter):
        
        # for i in range(1):

        error_persample = np.dot(X,theta)-Y
        error_persample = error_persample[np.newaxis].T

        if iteration == 0:
            loss = np.dot(error_persample.T,error_persample)[0,0]/N
            loss += lambda_LASSO * np.abs(theta).sum()
            loss_story.append(loss)
            toc = timeit.default_timer()-tic
            time_story.append(toc)
            iteration_story.append(iteration)
            print("iteration = ", int(iteration),
                    " | loss = ", loss,
                    " | time = ", toc)

        gr_persample = 2*np.multiply(X,error_persample)
        gr_persample += lambda_LASSO * np.sign(theta)
        gr1d = np.mean(gr_persample,0)
        
        # v = beta*v - lr*gr1d
        # if i == 0:
        theta -= lr*gr1d
        iteration += 1

        loss = np.dot(error_persample.T,error_persample)[0,0]/N
        loss += lambda_LASSO * np.abs(theta).sum()
        loss_story.append(loss)
        toc = timeit.default_timer()-tic
        time_story.append(toc)
        iteration_story.append(iteration)
        print("iteration = ", int(iteration),
                " | loss = ", loss,
                " | time = ", toc)

        blocks = building_blocks_cumsum(gr1d,size_block,percentage_gr,max_number_blocks,
                                "sorted") # sorted, random or balanced
        n_blocks = len(blocks)

        if loss > loss_accepted:

            # start parallel part
            manager = mp.Manager()
            results = manager.dict()
            processes = []

            for i in range(n_blocks):
                p = mp.Process(target = parallel_BCD,
                               args = (results,i,X,Y,lambda_LASSO,
                                    blocks[i], # direction_persample[:,blocks[i]],
                                    theta, # theta_tm1,
                                    # gr1d[blocks[i]], # gr1d_tm1[blocks[i]],
                                    # max_iter,
                                    # v,
                                    iteration,lr,loss_accepted))
                processes.append(p)
                p.start()

            i = 0
            for process in processes:
                process.join()
                i += 1
            
            # collecting results from the parallel execution
            for i in range(n_blocks):
                # if results[i][3] != 0:
                #     continue
                # if results[i][0] == 1:
                #     continue
                theta[blocks[i]] = results[i][0][blocks[i]]
                # v[blocks[i]] = results[i][1][blocks[i]]
            
            iteration += n_blocks #*(len(blocks[i])+1)/N

    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    loss = np.dot(error_persample.T,error_persample)[0,0]/N
    loss += lambda_LASSO * np.abs(theta).sum()
    loss_story.append(loss)
    toc = timeit.default_timer()-tic
    time_story.append(toc)
    iteration_story.append(iteration)
    print("iteration = ", int(iteration),
            " | loss = ", loss,
            " | time = ", toc)
            
    toc = timeit.default_timer()-tic
    return loss_story,iteration_story,theta,time_story

def parallel_BCD(results,proc_numb,X,Y,lambda_LASSO,
                    block, # direction_persample[:,blocks[i]],
                    theta, # theta_tm1,
                    # gr1d[blocks[i]], # gr1d_tm1[blocks[i]],
                    # max_iter,
                    # v,
                    iteration,lr,loss_accepted):
    # beta = 0.9
    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    gr_persample = 2*np.multiply(X,error_persample)
    gr_persample += lambda_LASSO * np.sign(theta)
    gr1d = np.mean(gr_persample,0)
    # v = beta*v - lr*gr1d
    # if i == 0:
    theta -= lr*gr1d
    print("PID parallel: ", os.getpid(), " | process number ", proc_numb) #," | iterations CA = ", iteration_CA)
    results[proc_numb] = [theta]
    return 

###############################################
# Momentum_CA_BCD parallel GS
###############################################

def mom_CA_BCD_GS_ls(X,Y,theta_0,lambda_LASSO=0.,
            lr=1e-3,loss_accepted=1e-8,max_iter=1e2,
            size_block=2,percentage_gr = 0.75):

    # it runs the CaBCD with momentum and with the acceleration via the Caratheodory Theorem,
    # using mom_CA_BCD_mod and the GS rule for the selection of the blocks

    num_cpus = psutil.cpu_count(logical=False)
    tic = timeit.default_timer()
    N = np.shape(X)[0]
    iteration = 0.
    loss = loss_accepted+1.

    assert np.size(theta_0)>=size_block, "less parameters than size_block, decrease the size block"
    
    # MOMENTUM param
    beta = 0.9
    v = np.zeros(np.size(theta_0))

    theta = np.array(theta_0)

    loss_story = []
    time_story = []
    iteration_story = []

    gr1d = np.empty(np.size(theta_0))
    max_number_blocks = np.infty #8*num_cpus
    to_extract = min(max_number_blocks*size_block,len(theta_0)*percentage_gr)
    to_extract -= to_extract % size_block
    to_extract = int(to_extract)

    while (loss > loss_accepted and iteration < max_iter):
        
        for i in range(2):

            error_persample = np.dot(X,theta)-Y
            error_persample = error_persample[np.newaxis].T

            if iteration == 0:
                loss = np.dot(error_persample.T,error_persample)[0,0]/N
                loss += lambda_LASSO * np.abs(theta).sum()
                loss_story.append(loss)
                toc = timeit.default_timer()-tic
                time_story.append(toc)
                iteration_story.append(iteration)
                print("iteration = ", int(iteration),
                        " | loss = ", loss,
                        " | time = ", toc)

            gr_persample = 2*np.multiply(X,error_persample)
            gr_persample += lambda_LASSO * np.sign(theta)
            gr1d = np.mean(gr_persample,0)

            if i == 0:
                theta_tm1 = np.copy(theta)
                gr1d_tm1 = np.copy(gr1d)
            
            if i==1:
                direction_persample = beta*v - lr*gr_persample
            
            v = beta*v - lr*gr1d
            if i == 0:
                theta += v
            iteration += 1

        loss = np.dot(error_persample.T,error_persample)[0,0]/N
        loss += lambda_LASSO * np.abs(theta).sum()
        loss_story.append(loss)
        toc = timeit.default_timer()-tic
        time_story.append(toc)
        iteration_story.append(iteration)
        print("iteration = ", int(iteration),
                " | loss = ", loss,
                " | time = ", toc)

        blocks = building_blocks_cumsum(gr1d,size_block,percentage_gr,max_number_blocks,
                                "sorted") # sorted, random or balanced
        n_blocks = len(blocks)

        if loss > loss_accepted:

            # start parallel part
            manager = mp.Manager()
            results = manager.dict()
            processes = []

            for i in range(n_blocks):
                p = mp.Process(target = directions_CA_steps_mom,
                               args = (results,i,X,Y,lambda_LASSO,
                                    blocks[i],direction_persample[:,blocks[i]],
                                    theta,theta_tm1,
                                    gr1d[blocks[i]],gr1d_tm1[blocks[i]],max_iter,v,
                                    iteration,lr,loss_accepted))
                processes.append(p)
                p.start()

            i = 0
            for process in processes:
                process.join()
                i += 1
            
            # collecting results from the parallel execution
            for i in range(n_blocks):
                if results[i][3] != 0:
                    continue
                if results[i][0] == 1:
                    continue
                theta[blocks[i]] = results[i][1][blocks[i]]
                v[blocks[i]] = results[i][2][blocks[i]]
                iteration += results[i][0]*(len(blocks[i])+1)/N

    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    loss = np.dot(error_persample.T,error_persample)[0,0]/N
    loss += lambda_LASSO * np.abs(theta).sum()
    loss_story.append(loss)
    toc = timeit.default_timer()-tic
    time_story.append(toc)
    iteration_story.append(iteration)
    print("iteration = ", int(iteration),
            " | loss = ", loss,
            " | time = ", toc)
            
    toc = timeit.default_timer()-tic
    return loss_story,iteration_story,theta,time_story

def mom_CA_BCD_mod(X,Y,
            theta_0,lambda_LASSO,v,
            mu,gradient_mean,iteration,
            idx,H,N,
            lr,loss_accepted,max_iter):
    
    # it runs the acceleration of the algorithm, i.e. the inner while of the descibed Algorithms
    # mu = mu hat
    # gradient_mean = the expecatation of the gradient computed using the wieght used to reduce the measure
    # H = approximation of the second derivative
    # v = momentum part 

    tic = timeit.default_timer()
    iteration_CA = 0
    loss = loss_accepted+1.
    theta = np.copy(theta_0)
    beta = 0.9
    b = True
    tmp = 0.
    
    while b:

        iteration_CA += 1
        theta_tm1 = np.copy(theta)
        v_tm1 = np.copy(v)
        
        error_persample = np.dot(X,theta)-Y
        error_persample = error_persample[np.newaxis]
        gr_persample = 2 * np.multiply(X[:,idx].T,error_persample)
        gr = np.sum(gr_persample*mu,1)
        gr += lambda_LASSO * np.sign(theta[idx])
        v[idx]= beta*v[idx]-lr*gr
        theta[idx] += v[idx]

        Delta_theta = theta[idx]-theta_0[idx]
        # tmp_1 = Delta using the notation of the pdf
        tmp_1 = np.dot(gradient_mean,Delta_theta)
        tmp_1 += 0.5*np.dot(Delta_theta,np.dot(H,Delta_theta))

        if iteration_CA == 1:
            b = True
            tmp = tmp_1
        else:
            b = np.around(tmp_1-tmp,5)<0
            tmp = tmp_1
            b = b and iteration_CA<max_iter and tmp<0
    
    if iteration_CA == max_iter:
        return(iteration_CA,theta,v)
    else:
        return(iteration_CA-1,theta_tm1,v_tm1)

def directions_CA_steps_mom(results,proc_numb,
                        X,Y,lambda_LASSO,
                        idx,recomb_data,theta,theta_tm1,
                        gr1d,gr1d_tm1,
                        max_iter,v,
                        iteration,lr,loss_accepted):

    # This function constitutes the parallel part of the Algorithm

    p = psutil.Process()
    recomb_data = recomb_data - v[idx]

    # CHECK INDEPENDENCY
    check = np.all(recomb_data == 0,0)
    check = np.arange(len(idx))[check]
    recomb_data = np.delete(recomb_data, check, 1)
    check = []
    for i in range(0,recomb_data.shape[1]):
        for j in range(recomb_data.shape[1]-1,i,-1):
            if np.allclose(np.abs(recomb_data[:,i]), np.abs(recomb_data[:,j])):
                check.append(i)
    # CHECK FINISHED
    
    recomb_data = np.delete(recomb_data, check, 1)
    w_star, idx_star, _, _, ERR_recomb = rb.recomb_combined(np.copy(recomb_data))[0:5]
    
    if ERR_recomb == 2:
        print("######################### ERR_recomb, max iter")
        iteration_CA, theta_CA, v = 0, 0, 0
        results[proc_numb] = [iteration_CA,theta_CA,v,ERR_recomb]
        return
    elif ERR_recomb == 6:
        print("######################### ERR_recomb, Numerical Instability and/or dependency in the data")
        iteration_CA, theta_CA, v = 0, 0, 0
        results[proc_numb] = [iteration_CA,theta_CA,v,ERR_recomb]
        return

    x_star = X[idx_star,:]
    y_star = Y[idx_star]

    # BUILDING HESSIAN APPROXIMATION
    vec = theta[idx]-theta_tm1[idx]
    vec = vec[np.newaxis]
    Delta_gr = gr1d-gr1d_tm1
    Delta_gr = Delta_gr[np.newaxis]
    H = np.dot(np.reciprocal(vec).T,Delta_gr)
    
    N = np.shape(X)[0]
    max_iter_BCDCA = 1/lr/10

    iteration_CA,theta_CA,v = mom_CA_BCD_mod(x_star,y_star,
                                        theta,lambda_LASSO,v, 
                                        w_star,gr1d,iteration,
                                        idx, H, N, 
                                        lr,loss_accepted,max_iter_BCDCA)
    
    print("PID parallel: ", os.getpid(), " | process number ",
          proc_numb," | iterations CA = ", iteration_CA)
    results[proc_numb] = [iteration_CA,theta_CA,v,ERR_recomb]
    return

###############################################
# Momentum_CA_BCD parallel random
###############################################

def mom_CA_BCD_random_ls(X,Y,theta_0,lambda_LASSO,
            lr=1e-3,loss_accepted=1e-8,max_iter=1e2,
            size_block=2,percentage_gr = 0.5):

    # it runs the CaBCD with momentum and with the acceleration via the Caratheodory Theorem,
    # using mom_CA_BCD_mod and the random rule for the seclection of the blocks

    num_cpus = psutil.cpu_count(logical=False)
    tic = timeit.default_timer()
    N = np.shape(X)[0]
    iteration = 0.
    loss = loss_accepted+1.

    assert np.size(theta_0)>=size_block, "less parameters than size_block, decrease the size block"
    
    # MOMENTUM param
    beta = 0.9
    v = np.zeros(np.size(theta_0))

    theta = np.array(theta_0)

    loss_story = []
    time_story = []
    iteration_story = []

    gr1d = np.zeros(np.size(theta_0))
    max_number_blocks = np.infty #8*num_cpus 
    to_extract = min(max_number_blocks*size_block,len(gr1d)*percentage_gr)
    to_extract -= to_extract % size_block
    to_extract = int(to_extract)

    while (loss > loss_accepted and iteration < max_iter):
        
        blocks = np.random.choice(len(theta),to_extract,replace = False)

        for i in range(2):
            
            error_persample = np.dot(X,theta)-Y
            error_persample = error_persample[np.newaxis].T
            
            if iteration == 0:
                loss = np.dot(error_persample.T,error_persample)[0,0]/N
                loss += lambda_LASSO * np.abs(theta).sum()
                loss_story.append(loss)
                toc = timeit.default_timer()-tic
                time_story.append(toc)
                iteration_story.append(iteration)
                print("iteration = ", int(iteration),
                        " | loss = ", loss,
                        " | time = ", toc)
            
            gr_persample = 2*np.multiply(X[:,blocks],error_persample)
            gr_persample += lambda_LASSO * np.sign(theta[blocks])
            gr1d = np.mean(gr_persample,0)

            if i==0:
                theta_tm1 = np.copy(theta)
                gr1d_tm1 = np.copy(gr1d)

            if i==1:
                direction_persample = beta*v[blocks] - lr*gr_persample
            
            v[blocks] = beta*v[blocks] - lr*gr1d
            if i==0:
                theta[blocks] += v[blocks]
            
            iteration += 1
        
        loss = np.dot(error_persample.T,error_persample)[0,0]/N
        loss += lambda_LASSO * np.abs(theta).sum()
        loss_story.append(loss)
        toc = timeit.default_timer()-tic
        time_story.append(toc)
        iteration_story.append(iteration)
        print("iteration = ", int(iteration),
                " | loss = ", loss,
                " | time = ", toc)
        
        blocks = blocks.reshape(-1,size_block)
        n_blocks = blocks.shape[0]

        if loss > loss_accepted:

            # start parallel part
            manager = mp.Manager()
            results = manager.dict()
            processes = []
            
            for i in range(n_blocks):
                p = mp.Process(target = directions_CA_steps_mom,
                               args = (results,i,X,Y,lambda_LASSO,
                                    blocks[i],direction_persample[:,i*size_block:(i+1)*size_block],
                                    theta,theta_tm1,
                                    gr1d[i*size_block:(i+1)*size_block],gr1d_tm1[i*size_block:(i+1)*size_block],max_iter,v, #m_hat,v_hat,
                                    iteration,lr,loss_accepted))
                processes.append(p)
                p.start()

            i = 0
            for process in processes:
                process.join()
                i += 1
            
            # collecting results from the parallel execution
            for i in range(n_blocks):
                if results[i][3] != 0:
                    continue
                if results[i][0] == 1:
                    continue
                theta[blocks[i]] = results[i][1][blocks[i]]
                v[blocks[i]] = results[i][2][blocks[i]]
                iteration += results[i][0]*len(blocks[i]+1)/N

    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    loss = np.dot(error_persample.T,error_persample)[0,0]/N
    loss += lambda_LASSO * np.abs(theta).sum()
    loss_story.append(loss)
    toc = timeit.default_timer()-tic
    time_story.append(toc)
    iteration_story.append(iteration)
    print("iteration = ", int(iteration),
            " | loss = ", loss,
            " | time = ", toc)

    toc = timeit.default_timer()-tic
    return (loss_story,iteration_story,theta,time_story)

###############################################
# CA_BCD parallel GS rule, No momentum
###############################################

def CA_BCD_GS_ls(X,Y,theta_0,lambda_LASSO=0.,
            lr=1e-3,loss_accepted=1e-8,max_iter=1e2,
            size_block=2,percentage_gr = 0.75):

    # it runs the CaBCD with the acceleration via the Caratheodory Theorem,
    # using CA_BCD_mod and the GS rule for the selection of the blocks

    num_cpus = psutil.cpu_count(logical=False)
    tic = timeit.default_timer()
    N = np.shape(X)[0]
    iteration = 0.
    loss = loss_accepted+1.

    assert np.size(theta_0)>=size_block, "less parameters than size_block, decrease the size block"
    
    theta = np.array(theta_0)

    loss_story = []
    time_story = []
    iteration_story = []

    gr1d = np.empty(np.size(theta_0))
    max_number_blocks = np.infty #8*num_cpus 
    to_extract = min(max_number_blocks*size_block,len(theta_0)*percentage_gr)
    to_extract -= to_extract % size_block
    to_extract = int(to_extract)

    while (loss > loss_accepted and iteration < max_iter):
        
        for i in range(2):
            
            error_persample = np.dot(X,theta)-Y
            error_persample = error_persample[np.newaxis].T

            if iteration == 0:
                loss = np.dot(error_persample.T,error_persample)[0,0]/N
                loss += lambda_LASSO * np.abs(theta).sum()
                loss_story.append(loss)
                toc = timeit.default_timer()-tic
                time_story.append(toc)
                iteration_story.append(iteration)
                print("iteration = ", int(iteration),
                        " | loss = ", loss,
                        " | time = ", toc)

            gr_persample = 2*np.multiply(X,error_persample)
            gr_persample += lambda_LASSO * np.sign(theta)
            gr1d = np.mean(gr_persample,0)
            
            if i == 0:
                theta_tm1 = np.copy(theta)
                gr1d_tm1 = np.copy(gr1d)

            if i==1:
                direction_persample = - lr*gr_persample
            
            if i == 0:
                theta -= lr*gr1d
            iteration += 1

        loss = np.dot(error_persample.T,error_persample)[0,0]/N
        loss += lambda_LASSO * np.abs(theta).sum()
        loss_story.append(loss)
        toc = timeit.default_timer()-tic
        time_story.append(toc)
        iteration_story.append(iteration)
        print("iteration = ", int(iteration),
                " | loss = ", loss,
                " | time = ", toc)

        blocks = building_blocks_cumsum(gr1d,size_block,percentage_gr,max_number_blocks,
                                "sorted") # sorted, random or balanced
        n_blocks = len(blocks)

        if loss > loss_accepted:

            # start parallel part
            manager = mp.Manager()
            results = manager.dict()
            processes = []
            
            for i in range(n_blocks):
                p = mp.Process(target = directions_CA_steps_standard,
                               args = (results,i,X,Y,lambda_LASSO,
                                    blocks[i],direction_persample[:,blocks[i]],
                                    theta,theta_tm1,
                                    gr1d[blocks[i]],gr1d_tm1[blocks[i]],max_iter,#v, #m_hat,v_hat,
                                    iteration,lr,loss_accepted))
                processes.append(p)
                p.start()

            i = 0
            for process in processes:
                process.join()
                i += 1
            
            # collecting results from the parallel execution
            for i in range(n_blocks):
                if results[i][3] != 0:
                    continue
                if results[i][0] == 1:
                    continue
                theta[blocks[i]] = results[i][1][blocks[i]]
                iteration += results[i][0]*(len(blocks[i])+1)/N

    error_persample = np.dot(X,theta)-Y
    error_persample = error_persample[np.newaxis].T
    loss = np.dot(error_persample.T,error_persample)[0,0]/N
    loss += lambda_LASSO * np.abs(theta).sum()
    loss_story.append(loss)
    toc = timeit.default_timer()-tic
    time_story.append(toc)
    iteration_story.append(iteration)
    print("iteration = ", int(iteration),
            " | loss = ", loss,
            " | time = ", toc)
            
    toc = timeit.default_timer()-tic
    return (loss_story,iteration_story,theta,time_story)

def CA_BCD_mod(X,Y,
            theta_0,lambda_LASSO,
            mu,gradient_mean,iteration,
            idx,H,N,
            lr,loss_accepted,max_iter):
    
    # it runs the acceleration of the algorithm, i.e. the inner while of the descibed Algorithms
    # mu = mu hat
    # gradient_mean = the expecatation of the gradient computed using the wieght used to reduce the measure
    # H = approximation of the second derivative

    tic = timeit.default_timer()
    iteration_CA = 0
    loss = loss_accepted+1.
    theta = np.copy(theta_0)
    b = True
    tmp = 0.
    gr = np.copy(gradient_mean)

    while b:

        iteration_CA += 1
        theta_tm1 = np.copy(theta)
        gr_tm1 = np.copy(gr)
        
        error_persample = np.dot(X,theta)-Y
        error_persample = error_persample[np.newaxis]
        gr_persample = 2 * np.multiply(X.T,error_persample)
        gr = np.sum(gr_persample*mu,1)
        gr += lambda_LASSO * np.sign(theta)
        theta[idx] -= lr * gr[idx]

        Delta_theta = theta[idx]-theta_0[idx]
        # tmp_1 = Delta using the notation of the pdf
        tmp_1 = np.dot(gradient_mean,Delta_theta)
        tmp_1 += 0.5*np.dot(Delta_theta,np.dot(H,Delta_theta))

        if iteration_CA == 1:
            b = True
            tmp = tmp_1
        else:
            b = np.around(tmp_1-tmp,5)<0
            tmp = tmp_1
            b = b and iteration_CA<max_iter and tmp<0

    if iteration_CA == max_iter:
        return(iteration_CA,theta,gr)
    else:
        return(iteration_CA-1,theta_tm1,gr_tm1)

def directions_CA_steps_standard(results,proc_numb,
                        X,Y,lambda_LASSO,
                        idx,recomb_data,theta,theta_tm1,
                        gr1d,gr1d_tm1,
                        max_iter,
                        iteration,lr,loss_accepted):

    # This function constitutes the parallel part of the Algorithm

    p = psutil.Process()
    recomb_data = recomb_data - gr1d
    
    # CHECK INDEPENDENCY
    check = np.all(recomb_data == 0,0)
    check = np.arange(len(idx))[check]
    recomb_data = np.delete(recomb_data, check, 1)
    check = []
    for i in range(0,recomb_data.shape[1]):
        for j in range(recomb_data.shape[1]-1,i,-1):
            if np.allclose(np.abs(recomb_data[:,i]), np.abs(recomb_data[:,j])):
                check.append(i)
    # CHECK FINISHED
    
    recomb_data = np.delete(recomb_data, check, 1)
    w_star, idx_star, _, _, ERR_recomb = rb.recomb_combined(np.copy(recomb_data))[0:5]
    
    if ERR_recomb == 2:
        print("######################### ERR_recomb, max iter")
        iteration_CA, theta_CA, v = 0, 0, 0
        results[proc_numb] = [iteration_CA,theta_CA,v,ERR_recomb]
        return
    elif ERR_recomb == 6:
        print("######################### ERR_recomb, Numerical Instability and/or dependency in the data")
        iteration_CA, theta_CA, v = 0, 0, 0
        results[proc_numb] = [iteration_CA,theta_CA,v,ERR_recomb]
        return

    x_star = X[idx_star,:]
    y_star = Y[idx_star]

    # BUILDING HESSIAN APPROXIMATION
    vec = theta[idx]-theta_tm1[idx]
    vec = vec[np.newaxis]
    Delta_gr = gr1d-gr1d_tm1
    Delta_gr = Delta_gr[np.newaxis]
    H = np.dot(np.reciprocal(vec).T,Delta_gr)
    
    N = np.shape(X)[0]
    max_iter_BCDCA = 1/lr/10 

    iteration_CA,theta_CA,v = CA_BCD_mod(x_star,y_star,
                                        theta,lambda_LASSO,
                                        w_star,gr1d,iteration,
                                        idx, H, N,
                                        lr,loss_accepted,max_iter_BCDCA)
    
    print("PID parallel: ", os.getpid(), " | process number ",
          proc_numb," | iterations CA = ", iteration_CA)
    results[proc_numb] = [iteration_CA,theta_CA,v,ERR_recomb]
    return

###############################################
# Auxiliary functions
###############################################

def tens_pow(X,pow):
    # compute the tenor power "pow" of X, i.e.
    # for any row add the columns with the mixed product

    tens_to = prod_tens(X,X)
    Y = np.append(X,tens_to,1)
    for _ in range(pow-2):
        tens_to = prod_tens(tens_to,X)
        Y = np.append(Y,tens_to,1)
    return Y

@njit(cache=True, parallel=True, fastmath=True)
def prod_tens(X,Y):
    # it computes the tensor produc of X and Y, i.e. 
    # if x_i^j indicates the i-th row, j-th column the function reuturns the matrix 
    # A = (x_i^j * y_i^h) for any h,j

    # X and Y must have the same first dimension

    N = X.shape[0]
    n1 = X.shape[1]
    n2 = Y.shape[1]
    A = np.empty((N,n1*n2))
    ii = 0
    for i in range(n1):
        for j in prange(n2):
            A[:,ii] = np.multiply(X[:,i],Y[:,j])
            ii += 1
    return A

def add_bias(X):
    # returns a matrix with one more column with all 
    # the elements equal to one
    N = X.shape[0]
    return np.append(X,np.ones((N,1)),axis = 1)

def building_blocks_cumsum(gr1d,size_block=2,percentage=0.75,max_number_blocks=np.Infinity,
                    strategy = "sorted"):
    # given the gradient this function returns the blocks in terms of indeces
    # strategy:
    #   - sorted = Gauss-Southwell
    #   - random 
    #   - balanced = similar to GS, but instead of order the directions by magnitude
    #                it groups the direction trying to build block with similar magnitude
    #                NOT USED IN THE WORK CITED
     
    idx_sorted = np.argsort(np.abs(gr1d))
    idx_sorted = np.flip(idx_sorted)
    cumsum = np.cumsum(np.abs(gr1d[idx_sorted]))
    cumsum /= np.sum(np.abs(gr1d))
    
    idx = cumsum<=percentage
    idx = np.arange(len(cumsum))[idx]
    if len(idx)<=1:
        idx_sorted = idx_sorted[0:2]
        blocks = [idx_sorted]
        return blocks
    else:
        idx = np.append(idx,idx[-1]+1)
        idx_sorted = idx_sorted[idx]

    n_blocks = int(np.floor(len(idx_sorted)/size_block))

    # DEBUG
    # print("total n_blocks = ", int(len(gr1d)/size_block),
    #       " | accepted = ", min(n_blocks, max_number_blocks))

    if strategy == "sorted":
        blocks = idx_sorted[:n_blocks*size_block].reshape(-1,size_block)
    elif strategy == "random":
        blocks = np.random.shuffle(idx_sorted[:n_blocks*size_block])
        blocks = blocks.reshape(-1,size_block)
    elif strategy == "balanced":
        for i in range(min(n_blocks, max_number_blocks)):
            idx = np.arange(len(idx_sorted))%n_blocks==i
            idx = idx_sorted[idx]
            blocks.append(idx)

    return np.array(blocks)

