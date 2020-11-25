# THIS REPOSITORY CONTATINS THE ALGORITHMS EXPLAINED IN THE WORK
# Cosentino, Oberhauser, Abate - "Caratheodory Sampling for Stochastic Gradient Descent" 

####################################################################################
# 
# This file contains functions necessary to use the selection rules from  
# https://github.com/IssamLaradji/BlockCoordinateDescent.
# 
# In particular, the function:
#       - train_carathodory is necessary to run the Caratheodory Sampling Procedure
#       - compute_for_plots & comparative_plots are used to run the experiments and
#         print the plots in the notebook plot.ipynb
# For more information please read the cited works and/or the README file.
# 
####################################################################################

import numpy as np
import copy, timeit, json, h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src import losses
from src import partition_rules
from src.selection_rules import VB_selection_rules
from src.selection_rules import FB_selection_rules
from src.update_rules import update_rules

# N = 1000000, n = 500
OPTIMAL_LOSS = {"A_ls": 500221.5830602676}      # computed with sklearn.linear_model.LinearRegression
plt.rcParams.update({'font.size': 14})

def load():

    data = h5py.File('exp4.mat', 'r')

    A, b = data['X'], data['y']
    A = np.array(A[:,:]).T
    b = np.array(b[:,:]).T
    b = b.ravel()

    A = A.astype(float)
    b = b.astype(float)

    return {"A":A, "b":b, "args":{}}  

def train_carathodory(A, b, datasetargs, loss_name, block_size, 
            partition_rule, selection_rule, update_rule, 
            n_iters, L1, L2, debug = False,
            optimal=None):

    np.random.seed(1)

    args = copy.copy(datasetargs)
    args.update({"L2": L2, "L1": L1, "block_size": block_size,
                 "update_rule": update_rule})

    # loss function & Get partitions
    lossObject = losses.create_lossObject(loss_name, A, b, args)
    partition = partition_rules.get_partition(
        A, b, lossObject, block_size, p_rule=partition_rule)

    # Initialize x
    x = np.zeros(lossObject.n_params)
    score_list = []

    ############ TRAINING STARTS HERE ############
    block = np.array([])
    iter_ca, total_CA_iter, i = 0, 0, 0
    tic = timeit.default_timer()
    while total_CA_iter <= n_iters+1:
        # Compute loss
        loss = lossObject.f_func(x, A, b)
        dis2opt = loss - OPTIMAL_LOSS["A" + "_" + loss_name]
        if i>0:
            total_CA_iter += 1 + (iter_ca-1) * block_size/A.shape[0] 
    
        score_list += [{"dis2opt": dis2opt, "iteration": i, 
                        "selected": block, "total_CA_iter": total_CA_iter, "time": timeit.default_timer()-tic}]

        if debug: print(score_list[-1])

        # Check increase
        if (i > 0) and (dis2opt > score_list[-2]["dis2opt"]):
            print("WARNING Caratheodory SP: loss value has increased...")
        
        # Select block
        np.random.seed(i)
        if partition is None:
            block, args = VB_selection_rules.select(
                selection_rule, x, A, b, lossObject, args, iteration=i)
        else:
            block, args = FB_selection_rules.select(
                selection_rule, x, A, b, lossObject, args, partition, iteration=i)

        # Update block
        x, args, iter_ca = update_rules.update_Caratheodory(
            update_rule, x, A, b, lossObject, args=args, block=block, iteration=i)

        i += 1

    return score_list

def compute_for_plots(loss, EXP_GROUPS, max_iter_array, write=False, debug=False):

    dataset = load()
    A, b, datasetargs = dataset["A"], dataset["b"], dataset["args"]
    
    p_index, b_index = 0, 0     # partition and block index
    score_list = []             # to be written on txt file          

    for fig in EXP_GROUPS:
        for partition in EXP_GROUPS[fig]["partition"]:
            for bs in EXP_GROUPS[fig]["block_size"]:
                for selection in EXP_GROUPS[fig]["selection"]:
                    for update in EXP_GROUPS[fig]["update"]:
                        print (fig, partition, selection, update)

                        max_iter = max_iter_array[p_index][b_index]

                        L1 = EXP_GROUPS[fig]["l1"]
                        L2 = EXP_GROUPS[fig]["l2"]                

                        if debug: print('Start Caratheodory')
                        try: 
                            score = train_carathodory(A, b, datasetargs, loss, bs, 
                                        partition, selection, update, 
                                        max_iter, L1, L2, debug,
                                        optimal=None)
                            score = np.array([list(x.values()) for x in score])
                            tmp = sum(score[:,3]<max_iter)

                            score_list.append(score[:tmp+1,3])
                            score_list.append(score[:tmp+1,0])
                            score_list.append(score[:tmp+1,4])

                        except: ValueError: print("selection rule doesn't exist!")

                b_index += 1
            b_index = 0
            p_index += 1
        p_index = 0

        score_list = [ x.tolist() for x in score_list ]
        if write:
            with open('figure/'+fig+'_list.txt', 'w') as f:
                f.write(json.dumps(score_list))

def comparative_plots(loss, EXP_GROUPS, max_iter_array, write=False):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    c = 0
    for fig in EXP_GROUPS: list_filename = 'figure/'+fig+'_list.txt'
    with open(list_filename, 'r') as f:
        score_list = json.loads(f.read())
    score_list = [ np.array(x) for x in score_list ]
    
    i = 0
    p_index, b_index = 0, 0    # partition and block index

    dataset = load()
    A, b, args = dataset["A"], dataset["b"], dataset["args"]
    x = np.zeros(A.shape[1])

    for fig in EXP_GROUPS:
        for partition in EXP_GROUPS[fig]["partition"]:
            for bs in EXP_GROUPS[fig]["block_size"]:
                
                min_dist2opt, min_dist2opt_CA = np.inf, np.inf
                max_dist2opt = np.max(score_list[i+1])
                plot, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10))

                for selection in EXP_GROUPS[fig]["selection"]:
                    for update in EXP_GROUPS[fig]["update"]:
                        
                        L1 = EXP_GROUPS[fig]["l1"]
                        L2 = EXP_GROUPS[fig]["l2"]

                        args.update({"L2": L2, "L1": L1, "block_size": 1,
                             "update_rule": update})

                        lossObject = losses.create_lossObject(loss, A, b, args)
                        partition_check = partition_rules.get_partition(
                                        A, b, lossObject, 1, p_rule=partition)

                        print (fig, partition, selection, update)
                        max_iter = max_iter_array[p_index][b_index]

                        try: 
                            
                            if partition_check is None:
                                block, args = VB_selection_rules.select(
                                    selection, x, A, b, lossObject, args, 0)
                            else:
                                block, args = FB_selection_rules.select(
                                    selection, x, A, b, lossObject, args, partition_check, 0)
                            
                            # score_x_iterations = score_list[i]
                            # i += 1
                            # score_y_dist2opt = score_list[i]
                            # i += 1
                            # score_x_time = score_list[i]
                            # i += 1
                            # min_dist2opt = min(min_dist2opt, np.min(score_y_dist2opt[:max_iter]))
                            
                            # if both:
                            #     ax1.plot(score_x_iterations[:max_iter],score_y_dist2opt[:max_iter],'--',
                            #             color = colors[c%10],
                            #             label=partition+'_'+selection+'_'+update)
                            #     ax2.plot(score_x_time[:max_iter],score_y_dist2opt[:max_iter],'--',
                            #             color = colors[c%10],
                            #             label=partition+'_'+selection+'_'+update)

                            score_x_iterations = score_list[i]
                            i += 1
                            score_y_dist2opt = score_list[i]
                            i += 1
                            score_x_time = score_list[i]
                            i += 1
                            tmp = sum(score_x_iterations < max_iter)
                            min_dist2opt_CA = min(min_dist2opt_CA, np.min(score_y_dist2opt[:tmp+1]))

                            ax1.plot(score_x_iterations[:tmp+1],score_y_dist2opt[:tmp+1],
                                    color = colors[c%10],
                                    label=partition+'_'+selection+'_'+update+'_CA')
                            
                            ax2.plot(score_x_time[:tmp+1],score_y_dist2opt[:tmp+1],
                                    color = colors[c%10],
                                    label=partition+'_'+selection+'_'+update+'_CA')
                        except: ValueError: print("selection rule doesn't exist!")

                        c += 1
                
                # if both:
                #     ax1.axhline(y = min_dist2opt, color = 'black', alpha=0.7, ls='--')
                #     ax1.text(y = min_dist2opt + (max_dist2opt-min_dist2opt)*0.05, x = 0, 
                #             s = 'min = '+"{:.2e}".format(min_dist2opt), 
                #             alpha=0.7, color='black')
                ax1.axhline(y = min_dist2opt_CA, color = 'black', alpha=0.7)
                ax1.text(y = min_dist2opt_CA + (max_dist2opt-min_dist2opt_CA)*0.05, x = max_iter*0.35, 
                        s = 'min_CA = '+"{:.2e}".format(min_dist2opt_CA), 
                        alpha=0.7, color='black')

                    
                ax1.set_title(fig+'_'+partition+'_block'+str(bs))
                ax1.set_xlabel('iteration')
                ax1.set_ylabel('loss - loss$^*$')
                ax1.ticklabel_format(axis="y", style="sci")
                # ax1.set_yscale('log')
                ax1.legend(loc = "right")
                
                ax2.set_xlabel('time')
                ax2.set_ylabel('loss - loss$^*$')
                ax2.ticklabel_format(axis="y", style="sci")
                # ax2.set_yscale('log')
                ax2.legend(loc = "right")
                
                ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                plt.tight_layout()
                
                if write:
                    plt.savefig('figure/'+fig+'_'+partition+'_block'+str(bs)+'_onlyCA.pdf')
                
                plt.show()
                
                b_index += 1
                c = 0
                
            b_index = 0
            p_index += 1
            
        p_index = 0
                
# TO TEST
if __name__ == "__main__":
    iterations = [[10, 10]]
    EXP_GROUPS = {}
    EXP_GROUPS['fig4a_and_8_and_10'] = {'partition':["VB"],
                    'selection':['Random', 'Cyclic'],
                    'update':['Lb'],
                    'block_size': [5],
                    'l1':0, 'l2':0}
    write_exp = True
    write_fig = True
    debug = True
    compute_for_plots("ls", EXP_GROUPS, iterations, write_exp, debug)
    comparative_plots("ls", EXP_GROUPS, iterations, write_fig)