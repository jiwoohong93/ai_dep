import gurobipy as gb
import os
import numpy as np
import pandas as pd
import pdb
import gnureadline
# import load_IPW
import load_sIPM        # 241211 초은 수정
import argparse
import math, time
os.environ["MKL_NUM_THREADS"] = "3" 
os.environ["NUMEXPR_NUM_THREADS"] = "3"  
os.environ["OMP_NUM_THREADS"] = "3" 
res = pd.DataFrame({'tau':[],'obv':[],'black':[],'hispanic':[],'white':[]})

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))


def all_of_it(sim, frac, TT):

    print('sim=' + str(sim))
    print('frac=' + str(frac))
    print('TT=' + str(TT))
    sim=5
    # _, _, S, X, A, A_oh, neigh = load_IPW.get_data(sim, 0)
    _, _, S, X, A, A_oh, neigh, Cov, x2 = load_sIPM.get_data(sim, 0)
    n = S.shape[0]
    da = A.shape[-1]
    if frac:
        DICT = np.load('school_weights_linear_mse_sim' + str(sim) + '_max1_frac_sIPM_root_res.npz')
    else:
        DICT = np.load('school_weights_linear_mse_sim' + str(sim) + '_max1_sIPM_root_res.npz')
    w = DICT['w'].T
    
    # get weight matrix, for x2
    w1 = w[0:da,:] #beta
    w2 = w[da:da*2,:] #alpha
    w3 = w[da*2:da*3,:] #gamma
    w4 = w[da*3:da*4,:] #theta
    neigh = neigh.astype(int)
    # we just care about int_on for now
    x1 = X[:,0]
    x3 = X[:,2]
    
    A_ix = np.argmax(A,axis=1)
    
    bit_mask=np.zeros([2**neigh.shape[1],neigh.shape[1]]) # E
     
    ints=np.arange(2**neigh.shape[1],dtype=np.int)
    for i in range(neigh.shape[1]):
        bit_mask[:,i]=ints%2
        ints//=2
    

    
    #한 인종만 있을 때 potential outcome
    def EY(index,mask,a):
        neighS = S[index,neigh[index,:]]
        first  = w1[a]*np.max(neighS*x1[neigh[index,:]])
        second = w2[a]*np.max(neighS*mask)
        third  = w3[a]*x3[index]
        fourth = w4[a]
        return first + second + third + fourth
    
    # 인종섞여있을 때 potential outcome
    def EY_inner(index,mask,a):
        neighS = S[index,neigh[index,:]]
        first  = np.dot(a,w1)*np.max(neighS*x1[neigh[index,:]])
        second = np.dot(a,w2)*np.max(neighS*mask)
        third  = np.dot(a,w3)*x3[index]
        fourth = np.dot(a,w4)
        return first + second + third + fourth
    
    
    def f(index,mask,constraint=0):
        if frac:
            return EY_inner(index,mask,A[np.newaxis,index,:])
        else:
            return EY(index,mask,A_ix[index])
        
    def count_f(index,mask,constraint):
        eya= EY(index,mask,constraint)
        return eya
    
    #모든 경우에 대해 potential outcome 게산
    def get_weights(i,newf=f,constraint=0):
        weights=np.empty(bit_mask.shape[0])
        for r in range(bit_mask.shape[0]):
            weights[r]=newf(i,bit_mask[r],constraint)
        return weights

    all_times = []


    # DICT = np.load('results/TAUS_frac.npz')
    # TAUS = DICT['TAUS']
    print('new')
    #print('TAUS=' + str(TAUS))
    ind = 0 
    TAUS = [0.04631579]#,0.18526316 ]
    print('TAUS=' + str(TAUS))
    for t in range(TT):
        print('t=' + str(t))
        for Tau in TAUS:#np.linspace(0.04, 0.16, 20):
            print('running tau=' + str(Tau))
            start = time.time()
            #Now build variables
            model = gb.Model()
            
            interventions=model.addVars(np.arange(neigh.shape[0]),
                                                  lb=0,#np.zeros(neigh.shape[0]),
                                                  ub=1,#np.ones(neigh.shape[0]),
                                                  vtype=gb.GRB.BINARY)
            K = 25 
            expr = gb.LinExpr()
            for i in range(len(interventions)):
                expr += interventions[i]
            model.addConstr(expr, gb.GRB.LESS_EQUAL, K, "k")
            
            
            counter_const=0
            
            def add_constrained_aux(index,tau=False):
                #init=z.copy()
                #init[-1]=1
                
                weights=get_weights(index)
                
                counter=np.empty((3,weights.shape[0]))
                counter[:]=weights[np.newaxis]
                for i in range(3):
                    counter[i] -= get_weights(index,count_f,i)
                aux = model.addVars(np.arange(bit_mask.shape[0]),#2**neigh.shape[1]),
                                   lb=0,ub=1,
                                   obj=weights,
                                   vtype=gb.GRB.CONTINUOUS)
                model.update()
                for i in range(bit_mask.shape[0]):
                    for j in range(bit_mask.shape[1]):
                        if bit_mask[i,j]:
                            model.addConstr(aux[i]<=interventions[neigh[index,j]])
                        else:
                            model.addConstr(aux[i]<=1-interventions[neigh[index,j]])
                model.addConstr(aux.sum()==1)
                if tau is not False:
                    for i in range(3):
                        model.addConstr(sum(aux[f]*counter[i,f] for f in range(weights.shape[0]))<=tau)
                return aux
            
            aux = list(map(lambda x: add_constrained_aux(x,tau=Tau),range(neigh.shape[0])))
            
                
            model.setObjective(model.getObjective(),gb.GRB.MAXIMIZE)
            model.optimize()
            end = timeSince(start)
            all_times.append(end)
        
            
            if model.status == gb.GRB.Status.OPTIMAL:
                sol = [interventions[i].X for i in range(len(interventions))]
                sol = np.array(sol)
                sol = np.round(sol)
                sol = sol.astype(bool)
                res.loc[ind,'tau'] = Tau
                res.loc[ind,'obv'] = model.objVal
                res.loc[ind,['black','hispanic','white']] = np.sum(A_oh[sol,:],axis=0)
                res.to_csv('./obj_val_sIPM_root_240902.csv')
                ind += 1
            else:
                print('did not work')
                sol = []
            if frac:
                filename = 'results/max_fair_k' + str(K) + '_' + str(Tau) + '_sim' + str(sim) + '_frac_sIPM_root'
                timename = 'results/time_max_k' + str(K) + '_sim' + str(sim) + '_t' + str(t+1) + '_frac_sIPM_root'
            else:
                filename = 'results/max_fair_k' + str(K) + '_' + str(Tau) + '_sim' + str(sim)+ '_sIPM_root'
                timename = 'results/time_max_k' + str(K) + '_sim' + str(sim) + '_t' + str(t+1)+ '_sIPM_root'
            model.write(filename + '.lp')
            np.savez_compressed(filename, sol=sol)
            print('done!')
            print('A dist')
            print(np.sum(A_oh[sol,:],axis=0))
        print(all_times)
        np.savez_compressed(timename, times=all_times, tau=TAUS)#np.linspace(0.04, 0.16, 20))

        print('======================================results========================================')
        if model.status==2:
            res_status = 'Found Optimal Solution'
        print(f'model.status : {res_status} when Tau={Tau}')


all_of_it(5, 1, 1)
