import pandas as pd
import numpy as np
import pdb
import gnureadline

def get_data(sim, minority):
    school_df = pd.read_csv('school_data.csv')

    # 1. load races of all schools (except Other)
    black = school_df["Black"].values.astype(float)
    hisp  = school_df["Hispanic"].values.astype(float)
    white = school_df["White"].values.astype(float)
    n = len(black) # get total number of schools
    a = np.concatenate((black.reshape(n,1), hisp.reshape(n,1), white.reshape(n,1)), axis=1) # (n,d)
    # either keep as fractions or make into 1-hot encoding
    a_oh = np.copy(a)
    a_oh = a_oh / 100.0
    a_oh_bin = np.copy(a)
    a_oh_bin[:] = 0.0
    a_oh_bin[np.arange(n), np.argmax(a,axis=1)] = 1.0

    if minority:
        a_ix = np.argmax(a_oh,axis=1)
        mix = np.concatenate((np.where(a_ix == 0)[0], np.where(a_ix == 1)[0]))
        n = mix.shape[0]
        a_oh = a_oh[mix,:]
        a_oh_bin = a_oh_bin[mix,:]
    # 2. nearby schools interfere with each other, the impact of the interference is based on their inverse distance to each other
    # get coordinates of schools
    x_coord = school_df['X'].values
    y_coord = school_df['Y'].values
    if minority:
        x_coord = x_coord[mix]
        y_coord = y_coord[mix]
    XY = np.concatenate((x_coord.reshape(n,1), y_coord.reshape(n,1)),axis=1)
    # create l2-distance matrix
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            D[i,j] = np.sqrt(np.sum((XY[i,:] - XY[j,:])**2))
    D = D + D.T + np.eye(n)*1e-12 # to avoid divide by 0 warning
    S = 1.0/D
    # set diagonal to 1 because (a) taking classes at your own school is much more convenient and so should upper bound the similarity scores of taking classes somewhere else; (b) all non-diagonal entries in S are less than 1
    np.fill_diagonal(S,1)
    # get the closest [sim] schools, called 'neighbor schools' (+1 because this includes the school itself) 
    neigh = np.zeros((n,sim+1))
    for i in range(n):
        argS = np.argsort(-S[i,:]) # make negative to sort largest to smallest
        S[i,argS[sim+1:]] = 0 # zero out similarity of non-neighbor schools
        neigh[i,:] = argS[0:sim+1] # note the school itself will always occur first in each row of neigh
    
    # 3. get other features
    # x1 - "AP and/or IB"
    x1 = school_df["AP and/or IB"].values
    x1[x1 == 'Yes'] = 1.0
    x1[x1 == 'No']  = 0.0
    if minority:
        x1 = x1[mix]
    x1 = x1.astype(float).reshape(n,1)
    
    # x2 - "Calculus"
    # note: to fit the model we will use information about whether schools offer calculus, this allows us to assess the effect of calculus classes. when allocating interventions we will start with the 0-allocation: assuming no school has calculus classes, and then we will allocate them to assess the allocation procedure under different constraints
    x2 = school_df["Calculus"].values
    x2[x2 == 'Yes'] = 1.0
    x2[x2 == 'No']  = 0.0
    if minority:
        x2 = x2[mix]
    x2 = x2.astype(float).reshape(n,1)
    
    # x3 - "FTE School Counselors"
    x3 = school_df["FTE School Counselors"].values
    if minority:
        x3 = x3[mix]
    x3 = x3.astype(float).reshape(n,1)
    
    # lat/long
    lat = school_df["lat"].values
    if minority:
        lat = lat[mix]
    lat = lat.astype(float).reshape(n,1)
    
    lon = school_df["long"].values
    if minority:
        lon = lon[mix]
    lon = lon.astype(float).reshape(n,1)
    
    # y - "Total_SAT_ACT" / "Total_students"
    y = school_df["Total_SAT_ACT"].values.astype(float) / school_df["Total_students"].values.astype(float)
    if minority:
        y = y[mix]
    y = y.reshape(n,1)
    
    
    # 4. compute maxes and combine with other features
    A = np.max(S * x1.T,axis=1).reshape(n,1) * a_oh
    C = np.max(S * x2.T,axis=1).reshape(n,1) * a_oh
    E = x3 * a_oh
    F = a_oh
    
    X = np.concatenate((A, C, E, F), axis=1)#, G), axis=1)

    feats = np.concatenate((x1, x2, x3), axis=1)
    
    
    # For SIPM
    Cov = np.concatenate((A, E, F), axis=1)
    
    

    # 5. for plotting data on map
    if minority:
        f = open('final_data_interventions_max_frac_minority.csv','w')
    else:
        f = open('final_data_interventions_max_frac.csv','w')

    # x1 - "AP and/or IB"
    # x2 - "Calculus"
    # x3 - "FTE School Counselors"
    # y - "Total_SAT_ACT" / "Total_students"
    # a - race
    # lat/long
    f.write('AP/IB,Calculus,Counselors,FracSAT/ACT,black,Hispanic,white,lat,long,gis_x,gis_y,neigh1,neigh2,neigh3,neigh4,neigh5\n')
    for i in range(n):
        string = ""
        string += str(x1[i,0]) + ","
        string += str(x2[i,0]) + ","
        string += str(x3[i,0]) + ","
        string += str(y[i,0])  + ","
        string += str(int(a_oh_bin[i,0])) + ","
        string += str(int(a_oh_bin[i,1])) + ","
        string += str(int(a_oh_bin[i,2])) + ","
        string += str(lat[i,0])  + ","
        string += str(lon[i,0])  + ","
        string += str(x_coord[i]) + ","
        string += str(y_coord[i]) + ","
        string += str(int(neigh[i,1])) + ","
        string += str(int(neigh[i,2])) + ","
        string += str(int(neigh[i,3])) + ","
        string += str(int(neigh[i,4])) + ","
        string += str(int(neigh[i,5])) + "\n"
        f.write(string)
    f.close()
    return X, y, S, feats, a_oh, a_oh_bin, neigh, Cov, x2



if __name__ == '__main__':
    X, y, S, feats, a_oh, a_oh_bin, neigh, Cov, x2 = get_data(5, 1)
