#!/usr/bin/env python
# coding: utf-8

# In[5]:

import numpy as np
import math
import time
import casadi
import scipy.linalg as sl
from numpy.linalg import matrix_power
from scipy.stats import multivariate_normal
from skimage import measure

class IAIMM_KF_MPC( ):
    def __init__(self, Params):
        self.index_EV = Params['index_EV']
        self.Ts = Params['Ts']
        self.N = Params['N']
        self.Th_MPC = Params['Th_MPC']
        self.Th_QP = Params['Th_QP']
        self.K_Lon_EV = Params['K_Lon_EV']
        self.K_Lat_EV = Params['K_Lat_EV']
        self.l_f = Params['l_f']
        self.l_r = Params['l_r']
        self.N_Lane = Params['N_Lane']
        self.N_M = Params['N_M']
        self.N_M_EV = Params['N_M_EV']
        self.N_Car = Params['N_Car']
        self.L_Center = Params['L_Center']
        self.L_Bound = Params['L_Bound']
        self.DSV = Params['DSV']
        self.DEV = Params['DEV']
        self.Dev = Params['Dev']
        self.SpeedLim = Params['SpeedLim']
        self.Weight= Params['Weight']
        self.w_veh = Params['w_veh']
        self.l_veh = Params['l_veh']
        self.zeta_l = Params['zeta_l']
        self.zeta_w = Params['zeta_w']
        self.zeta_EV = Params['zeta_EV']
        self.H = Params['H']
        self.infinity = Params['infinity']
        self.Models = Params['Models']
        self.K_SCMPC = Params['K_SCMPC']
        self.K_sampling = Params['K_sampling']
        self.std_parameters = Params['std_parameters']
        self.Q1 = Params['Q1']
        self.Q2 = Params['Q2']
        self.Q3 = Params['Q3']
        self.Q4 = Params['Q4']
        self.Q5 = Params['Q5']
        self.Q6 = Params['Q6']
        self.Q7 = Params['Q7']
        self.LongVelProj = self.construct_QP( )
        self.EVplanning = self.contruct_MT_MPC( )
    
    def VelocityTracking(self, x_ini, vx_ref, m, n_step): # velocity tracking model
        """"
        Input: 
                    x_ini is an array without shape, ref is a pure number, m is the pure number
        Output: 
                    F is an array matrix, X_KF is the array matrix
        """
        Ts = self.Ts
        L_Center = self.L_Center
        DSV = self.DSV
        K_Lon = self.K_Lon_EV
        K_Lat = self.K_Lat_EV
        k_lo_1 = K_Lon[0]
        k_lo_2 = K_Lon[1]
        k_la_1 = K_Lat[0]
        k_la_2 = K_Lat[1]
        k_la_3 = K_Lat[2] 
        y_ref = L_Center[m]
        F = np.array([[1, Ts, Ts**2/2, 0, 0, 0],
                      [0, 1-k_lo_1*Ts**2/2, Ts-k_lo_2*(Ts**2)/2, 0, 0, 0],
                      [0, -k_lo_1*Ts, 1-k_lo_2*Ts, 0, 0, 0],
                      [0, 0, 0, 1-k_la_1*(Ts**3)/6, Ts-k_la_2*(Ts**3)/6, Ts**2/2-k_la_3*(Ts**3)/6],
                      [0, 0, 0, -k_la_1*(Ts**2)/2, 1-k_la_2*(Ts**2)/2, Ts-k_la_3*(Ts**2)/2],
                      [0, 0, 0, -k_la_1*Ts, -k_la_2*Ts, 1-k_la_3*Ts]]) # shape = 6 x 6
        E = np.array([0, k_lo_1*(Ts**2)/2*vx_ref, k_lo_1*Ts*vx_ref, (Ts**3)/6*k_la_1*y_ref, (Ts**2)/2*k_la_1*y_ref, Ts*k_la_1*y_ref]) # array no shape
        X_KF = np.zeros((DSV, n_step+1))
        X_KF[:, 0] = x_ini
            
        for i in range(1, n_step+1):
            X_KF[:, i] = (F@X_KF[:, i-1]) + E
            
        return F, X_KF
        
    def LaneTracking(self, initial_y, m): # lane tracking model
        L_Center = self.L_Center
        Ts = self.Ts
        N = self.N
        K_Lat = self.K_Lat_EV
        k_la_1 = K_Lat[0]
        k_la_2 = K_Lat[1]
        k_la_3 = K_Lat[2]
        y_ref = L_Center[m]
        A = np.array([[1-k_la_1*(Ts**3)/6, Ts-k_la_2*(Ts**3)/6, (Ts**2)/2-k_la_3*(Ts**3)/6],
                      [-k_la_1*(Ts**2)/2, 1-k_la_2*(Ts**2)/2, Ts-k_la_3*(Ts**2)/2],
                      [-k_la_1*Ts, -k_la_2*Ts, 1-k_la_3*Ts]])
        B = np.array([(Ts**3)/6*k_la_1, (Ts**2)/2*k_la_1, Ts*k_la_1])*y_ref
        Y = np.zeros((3, N + 1))
        Y[:, 0] = initial_y
        
        for i in range(1, N+1):
            Y[:, i] =  (A@Y[:, i-1]) + B    
        
        return Y[0, :] # returns back longitudinal position, a row
    
    def LookLane(self, y_k): # look for the lane position
        L_Bound = self.L_Bound
        if (y_k <= L_Bound[1]): # the car is on lane 1
            LanePos = 1
        elif (L_Bound[1] < y_k) and (y_k <= L_Bound[2]):
            LanePos = 2
        elif (L_Bound[2] < y_k):
            LanePos = 3
        
        return LanePos
    
    def ScenarioObstacleRealization(self, Obst_k, MU_k, Ref_Speed_All_k, x_EV_k):
        N_Lane = self.N_Lane
        N_Car = self.N_Car
        N_M = self.N_M
        N = self.N
        infinity = self.infinity
        w_veh = self.w_veh
        l_veh = self.l_veh
        zeta_w = self.zeta_w
        zeta_l = self.zeta_l
        K_SCMPC = self.K_SCMPC
        K_sampling = self.K_sampling
        varsigma = 0.0001 # a very small standard deviation value in case sigma is 0
        Synthesized_SV = list( )
        x_position = np.zeros((N_Car, N + 1))
        y_mark_low = np.zeros((N_Car, N + 1))
        y_mark_middle = np.zeros((N_Car, N + 1))
        y_mark_up = np.zeros((N_Car, N + 1))

        for i in range(N_Car):
            if (np.sum(Obst_k[i]) != None) and (Obst_k[i][0, 0] >= x_EV_k): # no obstacle for vehicle i
                Pr = [ ] # save the non-zero probabilities
                model_index = [ ] # save the index for effective models
                if (K_SCMPC != 0) and (i != 3): # SV3 is the car that is braking, so do not sample it
                    for j in range(N_M):
                        if MU_k[i][j] != 0:
                            Pr.append(MU_k[i][j])
                            model_index.append(j)
                    synthesized_sv = np.ones((4, N+1)) # synthesized obstacle, middle_x, middle_y, dx, dy
                    max_y_mark = np.array([None]*(N + 1)) # position
                    mid_y_mark = np.array([None]*(N + 1)) # position
                    min_y_mark = np.array([None]*(N + 1)) # position
                    Index = np.array(np.argsort(Pr)) # index of probability
                    model_index = np.array(model_index)
                    model_index = model_index[Index]
                    model_index = model_index.tolist( )
                    Value = np.sort(Pr) # value of probability from small to large
                    cdf = np.append(0, np.cumsum(Value))
                    Sample = np.random.uniform(0, 1, K_SCMPC)
                    Count = np.array([0]*len(Pr)) # the number of sampled times of the element of Index
                    for j in range(K_SCMPC):
                        for h in range(len(cdf)-1):
                            if (cdf[h] <= Sample[j]) and (Sample[j] <= cdf[h+1]):
                                Count[h] += 1
                    temp_x_po = np.ones((K_SCMPC*K_sampling, N+1)) # longitudinal position
                    temp_y_po = np.ones((K_SCMPC*K_sampling, N+1)) # lateral position
                    Count_sampling = np.append(0, np.cumsum(Count)*K_sampling)
                    state = Obst_k[i][:, 0]
                    j = 0
                    for h in range(len(Pr)):
                        if Count[h] != 0:
                            model_sampling = model_index[h] # the model index which is sampled from
                            for j in range(Count_sampling[h], Count_sampling[h + 1]):
                                vx_ref = Ref_Speed_All_k[i][model_sampling]
                                temp_x_po[j, :], temp_y_po[j, :] = self.SamplingGeneration(model_sampling, vx_ref, state)
                    for j in range(N + 1):
                        x_bar = (np.max(temp_x_po[:, j]) + np.min(temp_x_po[:, j]))/2
                        y_bar = (np.max(temp_y_po[:, j]) + np.min(temp_y_po[:, j]))/2
                        Dx = (np.max(temp_x_po[:, j]) - np.min(temp_x_po[:, j]))/2 + zeta_l*l_veh
                        Dy = (np.max(temp_y_po[:, j]) - np.min(temp_y_po[:, j]))/2 + zeta_w*w_veh
                        synthe = np.array([x_bar, y_bar, Dx, Dy]) # x-center, y-center, x-width, y-width
                        synthesized_sv[:, j] = synthe
                        max_y = synthe[1] + synthe[3] # maximum y boundary
                        min_y = synthe[1] - synthe[3] # minimum y boundary
                        max_y_mark[j] = self.LookLane(max_y)
                        mid_y_mark[j] = self.LookLane(synthe[1])
                        min_y_mark[j] = self.LookLane(min_y)

                elif (K_SCMPC == 0) or (i == 3):
                    synthesized_sv = np.ones((4, N+1)) # synthesized obstacle, middle_x, middle_y, dx, dy
                    max_y_mark = np.array([None]*(N + 1)) # position
                    mid_y_mark = np.array([None]*(N + 1)) # position
                    min_y_mark = np.array([None]*(N + 1)) # position
                    temp_x_po = Obst_k[i][0, :]
                    temp_y_po = Obst_k[i][3, :]
                    for j in range(N + 1):
                        synthe = np.array([temp_x_po[j], temp_y_po[j], zeta_l*l_veh, zeta_w*w_veh]) # x-center, y-center, x-width, y-width
                        synthesized_sv[:, j] = synthe
                        max_y = synthe[1] + synthe[3] # maximum y boundary
                        min_y = synthe[1] - synthe[3] # minimum y boundary
                        max_y_mark[j] = self.LookLane(max_y)
                        mid_y_mark[j] = self.LookLane(synthe[1])
                        min_y_mark[j] = self.LookLane(min_y)
                
                Synthesized_SV.insert(i, synthesized_sv)
                x_position[i, :] = synthesized_sv[0, :] - synthesized_sv[2, :]
                y_mark_up[i, :] = max_y_mark
                y_mark_middle[i, :] = mid_y_mark
                y_mark_low[i, :] = min_y_mark
            
            else: # no obstacle for vehicle i
                Synthesized_SV.append(None)
                x_position[i, :] = np.array([infinity]*(N + 1))
                y_mark_up[i, :] = np.array([0]*(N + 1))
                y_mark_middle[i, :] = np.array([0]*(N + 1))
                y_mark_low[i, :] = np.array([0]*(N + 1))
        
        Temp_up = np.array([infinity]*(N + 1))
        Temp_middle = np.array([infinity]*(N + 1))
        Temp_low = np.array([infinity]*(N + 1))
        X_DV_Lane = [Temp_up, Temp_middle, Temp_low]
        for i in range(N_Lane):
            for k in range(N + 1):
                index_up =     np.where(y_mark_up[:, k] == i+1)
                index_middle = np.where(y_mark_middle[:, k] == i+1)
                index_low =    np.where(y_mark_low[:, k] == i+1)
                if len(index_up[0]) == 0:
                    temp_lane_up = infinity
                else:
                    temp_lane_up = np.min(x_position[:, k][index_up[0]])
                if len(index_middle[0]) == 0:
                    temp_lane_middle = infinity
                else:
                    temp_lane_middle = np.min(x_position[:, k][index_middle[0]])
                if len(index_low[0]) == 0:
                    temp_lane_low = infinity
                else:
                    temp_lane_low = np.min(x_position[:, k][index_low[0]])
                X_DV_Lane[i][k] = np.min([temp_lane_up, temp_lane_middle, temp_lane_low])
        
        return Synthesized_SV, X_DV_Lane
                
    def SamplingGeneration(self, m, vx_ref, x_ini):
        Ts = self.Ts
        L_Center = self.L_Center
        DSV = self.DSV
        Models = self.Models
        std_parameters = self.std_parameters
        N = self.N
        K_set_lon = std_parameters[m][0]
        if (m == 0) or (m == 3) or (m == 6): # lane-keeping mode
            r = np.random.randint(0, len(K_set_lon) - 1)
            K_Lon = K_set_lon[r][0]
            K_Lat = Models[m][1]
        else:
            K_set_lat = std_parameters[m][1] 
            r = np.random.randint(0, len(K_set_lon) - 1)
            K_Lon = K_set_lon[r][0]
            K_Lat = K_set_lat[r][0]
        
        k_lo_1 = K_Lon[0]
        k_lo_2 = K_Lon[1]
        k_la_1 = K_Lat[0]
        k_la_2 = K_Lat[1]
        k_la_3 = K_Lat[2] 
        if (m == 0) or (m == 2):
            y_ref = L_Center[0]
        elif (m == 1) or (m == 3) or (m == 5):
            y_ref = L_Center[1]
        elif (m == 4) or (m == 6):
            y_ref = L_Center[2]
        F = np.array([[1, Ts, Ts**2/2, 0, 0, 0],
                      [0, 1-k_lo_1*Ts**2/2, Ts-k_lo_2*(Ts**2)/2, 0, 0, 0],
                      [0, -k_lo_1*Ts, 1-k_lo_2*Ts, 0, 0, 0],
                      [0, 0, 0, 1-k_la_1*(Ts**3)/6, Ts-k_la_2*(Ts**3)/6, Ts**2/2-k_la_3*(Ts**3)/6],
                      [0, 0, 0, -k_la_1*(Ts**2)/2, 1-k_la_2*(Ts**2)/2, Ts-k_la_3*(Ts**2)/2],
                      [0, 0, 0, -k_la_1*Ts, -k_la_2*Ts, 1-k_la_3*Ts]]) # shape = 6 x 6
        E = np.array([0, k_lo_1*(Ts**2)/2*vx_ref, k_lo_1*Ts*vx_ref, (Ts**3)/6*k_la_1*y_ref, (Ts**2)/2*k_la_1*y_ref, Ts*k_la_1*y_ref]) # array no shape
        X_KF = np.zeros((DSV, N+1))
        X_KF[:, 0] = x_ini
        for i in range(1, N+1):
            X_KF[:, i] = (F@X_KF[:, i-1]) + E
        
        x_samling = X_KF[0, :]
        y_samling = X_KF[3, :]
        
        return X_KF[0, :], X_KF[3, :]
        
    def Define_DV(self, Initial, X_DV_Lane, m):
        N = self.N
        Ts = self.Ts
        Dev = self.Dev
        L_Center = self.L_Center
        zeta_EV = self.zeta_EV
        CL  = self.LookLane(Initial[1] - zeta_EV)
        Phi = Initial[2] # initial heading
        Cy  = Initial[1] # initial lateral position
        X_DV = np.array([None]*(N + 1))
        m += 1 #  now it starts from 1
        
        if CL == 1: # the car is on lane 1
            if m == 1: # it wants to keep on lane 1
                Dy = Cy - L_Center[0]
                if (Phi >= Dev[0] and Dy >= Dev[1]): # the car may go to lane 2, 0 and 1 considered
                    X_DV = np.minimum(X_DV_Lane[0], X_DV_Lane[1])
                else:  # the car will always be on lane 1, 0 considered
                    X_DV = X_DV_Lane[0] 
            elif m == 2: # the car is on lane 1 but it wants to go to lane 2, 0 and 1 considered
                X_DV = np.minimum(X_DV_Lane[0], X_DV_Lane[1])
        elif CL == 2:  # the car is on lane 2
            if m == 2: # it wants to be on lane 2
                Dy = Cy - L_Center[1]
                if (Phi >= Dev[0] and Dy >= Dev[1]):      # the car is possible to go to lane 3, 1 and 2 considered
                    X_DV = np.minimum(X_DV_Lane[1], X_DV_Lane[2])
                elif (Phi <= -Dev[0] and Dy <= -Dev[1]):  # the car is possible to go to lane 1, 0 and 1 considered
                    X_DV = np.minimum(X_DV_Lane[0], X_DV_Lane[1])
                else:   # the car will always be on lane 2, so 0, 1, and 2 considered
                    X_DV = X_DV_Lane[1]
            elif m == 1: # it wants to go to lane 1, 0 and 1 considered
                X_DV = np.minimum(X_DV_Lane[0], X_DV_Lane[1])
            elif m == 3: # it wants to go to lane 3, 1 and 2 considered
                X_DV = np.minimum(X_DV_Lane[1], X_DV_Lane[2])     
        elif CL == 3: # the car is on lane 3
            if m == 3: # it wants to keep on lane 3
                Dy = Cy - L_Center[2]
                if (Phi <= -Dev[0] and Dy <= -Dev[1]): # the car is possible to go to lane 2, 1 and 2 considered
                    X_DV = np.minimum(X_DV_Lane[1], X_DV_Lane[2])
                else: # the car will always be on lane 3, 2 considered
                    X_DV = X_DV_Lane[2]
            elif m == 2: # it wants to go to lane 2, 1 and 2 considered
                X_DV = np.minimum(X_DV_Lane[1], X_DV_Lane[2])
        
        return X_DV
        
    def ProjectSpeed(self, Obst_k, y_k, x_hat_k, RefPrim, MU_k, Ref_Speed_All_k):
        index_EV = self.index_EV
        N_Lane = self.N_Lane
        N_Car = self.N_Car
        N_M = self.N_M
        N_M_EV = self.N_M_EV
        infinity = self.infinity
        N = self.N
        Ts = self.Ts
        DSV = self.DSV
        H = self.H
        K_Lon = self.K_Lon_EV
        K_Lat = self.K_Lat_EV
        
        a = np.array([[1, Ts, (Ts**2)/2], [0, 1-K_Lon[0]*(Ts**2)/2, Ts-K_Lon[1]*(Ts**2)/2], [0, -K_Lon[0]*Ts, 1-K_Lon[1]*Ts]])
        b = np.array([0, K_Lon[0]*(Ts**2)/2, K_Lon[0]*Ts])
    
        # x_hat_k = self.KF(k) # state estimate
        Synthesized_SV, X_DV_Lane = self.ScenarioObstacleRealization(Obst_k, MU_k, Ref_Speed_All_k, y_k[0])
        ProjVal = list( ) # save the projected speed
            
        for i in range(N_M_EV): # output is longitudinal position of obstacle on each lane, column
            if np.sum(x_hat_k[i]) == None: # x_hat_k[i] does not exist
                ProjVal.append(None)
            else: # x_hat_k[i] does exist
                initial_x = x_hat_k[i][0:3] # x part
                initial_y = x_hat_k[i][3::] # y part
                A = list( )
                B = list( )
                sel_x = np.array([[1, 0, 0]])
                A.append(sel_x@b)
                B.append(sel_x@matrix_power(a, 1)@initial_x)
                for j in range(1, N):
                    A.append(sel_x@matrix_power(a, j)@b + A[j - 1])
                    B.append(sel_x@matrix_power(a, j+1)@initial_x)
                SEL = list( )
                X_SV = list( )
                for j in range(N_Car):
                    if j != index_EV: # virtual EV
                        if np.sum(Synthesized_SV[j]) == None:
                            SEL = SEL + [0]*N 
                            X_SV = X_SV + [infinity]*N
                        else:
                            o_SV = Synthesized_SV[j][1]
                            W_SV = Synthesized_SV[j][3]
                            SEL = SEL + self.Sel_Matrix(initial_y, o_SV, W_SV, i)
                            temp_x = Synthesized_SV[j][0, 1::] - Synthesized_SV[j][2, 1::]
                            X_SV = X_SV + temp_x.tolist()     
                SEL = np.diag(SEL)
                X_SV = np.array(X_SV)
                A = np.array(A).reshape(N, 1)
                B = np.array(B).reshape(N, 1)
                A = np.tile(A, (N_Car - 1, 1))
                B = np.tile(B, (N_Car - 1, 1))
                X_SV = X_SV.reshape(N*(N_Car - 1), 1)
                vx_up = self.LongVelProj(A, B, X_SV, RefPrim[i], SEL)
                vx_up = vx_up.__float__()
                if vx_up < 0:
                    vx_up = 0
                ProjVal.append(vx_up)
                
        ProjVal = np.array(ProjVal)

        return ProjVal, X_DV_Lane, Synthesized_SV
        
    def Final_Return(self, k, state_k_loc, state_k_glo, Obst_k, y_k, MU_k, Ref_Speed_All_k):
        Ts = self.Ts
        N = self.N
        N_M_EV = self.N_M_EV
        N_Lane = self.N_Lane
        L_Center = self.L_Center
        Weight = self.Weight
        H = self.H
        SpeedLim = self.SpeedLim
        K_Lon = self.K_Lon_EV
        K_Lat = self.K_Lat_EV

        # define the estimate and primary reference
        if self.LookLane(state_k_loc[1]) == 1:
            x_hat_k = [state_k_glo, state_k_glo, None]
            RefPrim = [state_k_glo[1], state_k_glo[1], None]
        elif self.LookLane(state_k_loc[1]) == 2:
            x_hat_k = [state_k_glo, state_k_glo, state_k_glo]
            RefPrim = [state_k_glo[1], state_k_glo[1], state_k_glo[1]]
        elif self.LookLane(state_k_loc[1]) == 3:
            x_hat_k = [None, state_k_glo, state_k_glo]
            RefPrim = [None, state_k_glo[1], state_k_glo[1]]
        
        # update the primary reference accroding to road speed limit
        for i in range(N_Lane):
            if (np.sum(SpeedLim[i]) != None) and (np.sum(RefPrim[i]) != None): # the lane has nominal speed and the state is not empty 
                RefPrim[i] = SpeedLim[i]
        
        # update the primary reference speed 
        REF, X_DV_Lane, Synthesized_SV_k = self.ProjectSpeed(Obst_k, y_k, x_hat_k, RefPrim, MU_k, Ref_Speed_All_k) 
        # calculate the action cost and model probability
        ActPse = np.array([None]*N_M_EV)
        t = np.arange(0, Ts*(N + 1), Ts, dtype=float)
        L = np.array([None]*N_M_EV)
        c = np.array([1]*N_M_EV)
        
        for i in range(N_M_EV):
            if np.sum(x_hat_k[i]) == None: # if x_hat_k[i] does not exist
                ActPse[i] = None
                L[i] = 0
            else:   
                _, X = self.VelocityTracking(x_hat_k[i], REF[i], i, N)
                ax = X[2, :]*X[2, :]
                ay = X[5, :]*X[5, :]
                ActPse[i] = Weight[0]*np.trapz(ax, t) + Weight[1]*np.trapz(ay, t) + \
                Weight[2]*((REF[i] - x_hat_k[i][1])**2) + Weight[3]*((L_Center[i] - x_hat_k[i][3])**2)
                ActPse[i] = ActPse[i] + 0.0001
                L[i] = 1/np.sqrt(ActPse[i])
            
        temp = c@L
        mu_k = c*L/temp
        m_k = np.argmax(mu_k)
        
        # MPC-based planning
        RefSpeed = REF[m_k]
        Initial = state_k_loc
        Terminal = np.array([L_Center[m_k], RefSpeed])
        X_DV = self.Define_DV(Initial, X_DV_Lane, m_k)
        Initial = casadi.vertcat(Initial)
        Terminal = casadi.vertcat(Terminal)
        X_DV_Casadi = casadi.vertcat(X_DV)
        Traj_k, U_k = self.EVplanning(Initial, Terminal, X_DV_Casadi)
        Traj_k = Traj_k.full( )
        U_k = U_k.full( )
        rho_k = U_k[0, :] # soft variable
        snap_k = U_k[1, :] # longitudinal snap 
        alpha_k = U_k[2, :] # angular acceleration 
        x_pre_k = self.V2G(Traj_k)
        state_k_plus_1_loc = Traj_k[:, 1]
        state_k_plus_1_glo = x_pre_k[:, 1]
        y_k_plus_1 = H@state_k_plus_1_glo
        
        return RefSpeed, L_Center[m_k], mu_k, m_k, x_hat_k, x_pre_k, Traj_k, U_k, state_k_plus_1_loc, state_k_plus_1_glo, y_k_plus_1, Synthesized_SV_k, REF
    
    def Final_Return_Com_Time(self, k, state_k_loc, state_k_glo, Obst_k, y_k, MU_k, Ref_Speed_All_k):
        Ts = self.Ts
        N = self.N
        N_M_EV = self.N_M_EV
        N_Lane = self.N_Lane
        L_Center = self.L_Center
        Weight = self.Weight
        H = self.H
        SpeedLim = self.SpeedLim
        K_Lon = self.K_Lon_EV
        K_Lat = self.K_Lat_EV

        # define the estimate and primary reference
        if self.LookLane(state_k_loc[1]) == 1:
            x_hat_k = [state_k_glo, state_k_glo, None]
            RefPrim = [state_k_glo[1], state_k_glo[1], None]
        elif self.LookLane(state_k_loc[1]) == 2:
            x_hat_k = [state_k_glo, state_k_glo, state_k_glo]
            RefPrim = [state_k_glo[1], state_k_glo[1], state_k_glo[1]]
        elif self.LookLane(state_k_loc[1]) == 3:
            x_hat_k = [None, state_k_glo, state_k_glo]
            RefPrim = [None, state_k_glo[1], state_k_glo[1]]
        
        # update the primary reference accroding to road speed limit
        for i in range(N_Lane):
            if (np.sum(SpeedLim[i]) != None) and (np.sum(RefPrim[i]) != None): # the lane has nominal speed and the state is not empty 
                RefPrim[i] = SpeedLim[i]
        
        # update the primary reference speed 
        REF, X_DV_Lane, Synthesized_SV_k = self.ProjectSpeed(Obst_k, y_k, x_hat_k, RefPrim, MU_k, Ref_Speed_All_k) 
        # calculate the action cost and model probability
        ActPse = np.array([None]*N_M_EV)
        t = np.arange(0, Ts*(N + 1), Ts, dtype=float)
        L = np.array([None]*N_M_EV)
        c = np.array([1]*N_M_EV)
        
        for i in range(N_M_EV):
            if np.sum(x_hat_k[i]) == None: # if x_hat_k[i] does not exist
                ActPse[i] = None
                L[i] = 0
            else:   
                _, X = self.VelocityTracking(x_hat_k[i], REF[i], i, N)
                ax = X[2, :]*X[2, :]
                ay = X[5, :]*X[5, :]
                ActPse[i] = Weight[0]*np.trapz(ax, t) + Weight[1]*np.trapz(ay, t) + \
                Weight[2]*((REF[i] - x_hat_k[i][1])**2) + Weight[3]*((L_Center[i] - x_hat_k[i][3])**2)
                ActPse[i] = ActPse[i] + 0.0001
                L[i] = 1/np.sqrt(ActPse[i])
            
        temp = c@L
        mu_k = c*L/temp
        m_k = np.argmax(mu_k)
        
        # MPC-based planning
        RefSpeed = REF[m_k]
        Initial = state_k_loc
        Terminal = np.array([L_Center[m_k], RefSpeed])
        X_DV = self.Define_DV(Initial, X_DV_Lane, m_k)
        Initial = casadi.vertcat(Initial)
        Terminal = casadi.vertcat(Terminal)
        X_DV_Casadi = casadi.vertcat(X_DV)
        start_ocp = time.perf_counter( )
        Traj_k, U_k = self.EVplanning(Initial, Terminal, X_DV_Casadi)
        end_ocp = time.perf_counter( )
        ocp_time = end_ocp - start_ocp
        Traj_k = Traj_k.full( )
        U_k = U_k.full( )
        rho_k = U_k[0, :] # soft variable
        snap_k = U_k[1, :] # longitudinal snap 
        alpha_k = U_k[2, :] # angular acceleration 
        x_pre_k = self.V2G(Traj_k)
        state_k_plus_1_loc = Traj_k[:, 1]
        state_k_plus_1_glo = x_pre_k[:, 1]
        y_k_plus_1 = H@state_k_plus_1_glo
        
        return RefSpeed, L_Center[m_k], mu_k, m_k, x_hat_k, x_pre_k, Traj_k, U_k, state_k_plus_1_loc, state_k_plus_1_glo, y_k_plus_1, Synthesized_SV_k, REF, ocp_time
    
    def V2G(self, local_state):
        DSV = self.DSV
        l_f = self.l_f
        l_r = self.l_r
        dimension = local_state.ndim
        if dimension == 1: # a single vector
            beta = l_r/(l_r + l_f)*local_state[6]
            global_state = np.array([local_state[0], local_state[3]*np.cos(local_state[2] + beta), 
                                     local_state[4]*np.cos(local_state[2] + beta), 
                                     local_state[1], local_state[3]*np.sin(local_state[2] + beta), 
                                     local_state[4]*np.sin(local_state[2] + beta)])
        else:
            n = local_state.shape[1]
            global_state = np.zeros((DSV, n))
            for i in range(n):
                beta = l_r/(l_r + l_f)*local_state[6, i]
                global_state[:, i] = np.array([local_state[0, i], local_state[3, i]*np.cos(local_state[2, i] + beta), 
                                               local_state[4, i]*np.cos(local_state[2, i]+ beta), 
                                               local_state[1, i], local_state[3, i]*np.sin(local_state[2, i] + beta),
                                               local_state[4, i]*np.sin(local_state[2, i]) + beta])
        return global_state
            
    def Sel_Matrix(self, initial_y, o_SV, W_SV, m):
        Ts = self.Ts
        N = self.N
        w_veh = self.w_veh
        y_EV = self.LaneTracking(initial_y, m)
        y_EV = y_EV[1:] # from k + 1 to k + N
        o_SV = o_SV[1:] # from k + 1 to k + N
        W_SV = W_SV[1:] # from k + 1 to k + N
        SEL = [0]*N
        for i in range(N):
            if np.abs(y_EV[i] - o_SV[i]) < (w_veh/2 + W_SV[i]):
                SEL[i] = 1
            
        return SEL
    
    def construct_QP(self):
        Ts = self.Ts
        N = self.N
        Th_QP = self.Th_QP
        N_Car = self.N_Car
        l_veh = self.l_veh
        opti = casadi.Opti( )
        H = opti.parameter(N*(N_Car - 1), N*(N_Car - 1))
        X_SV = opti.parameter(N*(N_Car - 1), 1)
        A = opti.parameter(N*(N_Car - 1), 1)
        B = opti.parameter(N*(N_Car - 1), 1)
        v_pri = opti.parameter( )
        v_up = opti.variable( )
        X_EV = A*v_up + B
        D_Error = X_SV - X_EV
        Safe_D = Th_QP*v_up*np.ones((N*(N_Car-1), 1)) + l_veh/2
        J = (v_pri - v_up)**2
        opti.minimize(J)
        opti.subject_to(H@Safe_D <= H@D_Error)
        opts = {"ipopt.warm_start_init_point": "yes",
                "ipopt.print_level": 0,
                "ipopt.linear_solver": "ma57",
                "print_time": False}
                #'ipopt.max_iter': 100}
        opti.solver('ipopt', opts)
        
        return opti.to_function('f', [A, B, X_SV, v_pri, H], [v_up])
    
    def contruct_MT_MPC(self):
        N = self.N
        Nx = self.DEV
        Ts = self.Ts
        Th_MPC = self.Th_MPC
        l_veh = self.l_veh
        Q1 = self.Q1
        Q2 = self.Q2
        Q3 = self.Q3
        Q4 = self.Q4
        Q5 = self.Q5
        Q6 = self.Q6
        Q7 = self.Q7
        
        opti = casadi.Opti( )
        X = opti.variable(Nx, N+1)         # states, from k to k+N + 1
        Opt_variable = opti.variable(3, N) # Opt_variable, from k to k + N
        rho   = Opt_variable[0, :].T       # slack variable
        snap  = Opt_variable[1, :].T                 # longitudinal snap 
        alpha = Opt_variable[2, :].T                 # angular acceleration
        Terminal = opti.parameter(2, 1) # terminal reference state
        Initial  = opti.parameter(Nx, 1) # initial state
        X_DV     = opti.parameter(N+1, 1)   # the most direct obstacles
        
        # model constraints, RK4
        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.vehicle_model(X[:, k], snap[k], alpha[k])
            k2 = self.vehicle_model(X[:, k] + Ts / 2 * k1, snap[k], alpha[k])
            k3 = self.vehicle_model(X[:, k] + Ts / 2 * k2, snap[k], alpha[k])
            k4 = self.vehicle_model(X[:, k] + Ts * k3, snap[k], alpha[k])
            x_next = X[:, k] + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next)
        
        # state collection
        x = X[0, 1::].T # longitudinal position
        y = X[1, 1::].T # lateral position
        v = X[3, 1::].T # longitudinal speed
        a = X[4, 1::].T # longitudinal acceleration
        delta = X[6, 1::].T # front tire angle
        y_error = y[-1] - Terminal[0] # lateral position error
        v_error = v[-1] - Terminal[1] # longitudinal speed error
        
        # stage constraint
        opti.subject_to(0 <= v)
        opti.subject_to(opti.bounded(-6, a, 6))
        opti.subject_to(opti.bounded(-0.8, delta, 0.8))
        
        # collision avoidance constraints
        opti.subject_to(v*Th_MPC - rho <= X_DV[1::] - x - l_veh/2)
        opti.subject_to(0 <= rho)
        
        # objective function
        J = snap.T@Q1@snap + alpha.T@Q2@alpha + a.T@Q3@a + delta.T@Q4@delta + y_error@Q5@y_error + v_error@Q6@v_error + rho.T@Q7@rho
        
        opti.minimize(J)
        
        # solution
        opts = {"ipopt.warm_start_init_point": "yes",
                "ipopt.print_level": 0,
                "ipopt.linear_solver": "ma57",
                "print_time": False}
                #'ipopt.max_iter': 100}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [Initial, Terminal, X_DV], [X, Opt_variable])
        
    def vehicle_model(self, w, snap, alpha):
        l_f = self.l_f
        l_r = self.l_r
        
        x_dot = w[3]
        y_dot = w[3]*w[2] + l_r/(l_r + l_f)*w[3]*w[6]
        phi_dot = 1/(l_r + l_f)*w[3]*w[6]
        v_dot = w[4]
        a_dot = w[5]
        j_dot = snap
        delta_dot = w[7]
        omiga_dot = alpha
        
        return casadi.vertcat(x_dot, y_dot, phi_dot, v_dot, a_dot, j_dot, delta_dot, omiga_dot)