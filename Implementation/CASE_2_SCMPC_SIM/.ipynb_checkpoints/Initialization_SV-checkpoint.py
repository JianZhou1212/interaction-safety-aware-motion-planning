#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import math
import matplotlib.pyplot as plt
import casadi
import scipy.linalg as sl
from numpy.linalg import matrix_power

class Initialization_SV( ):
    def __init__(self, Params):
        
        self.Ts = Params['Ts']
        self.N = Params['N']
        self.Models = Params['Models']
        self.N_Lane = Params['N_Lane']
        self.N_M      = Params['N_M']
        self.N_Car    = Params['N_Car']
        self.L_Bound  = Params['L_Bound']
        self.L_Center = Params['L_Center']
        self.DSV = Params['DSV']
        self.H   = Params['H']
        
    def Initialize_MU_M_P(self, X_State, index_EV): # state fusion steps of IMM-KF & define the primary reference speed
        N_M = self.N_M
        N = self.N
        N_Car = self.N_Car
        H = self.H
        Models = self.Models
        DSV = self.DSV
        L_Bound = self.L_Bound
        L_Center = self.L_Center
        
        X_State_0 = X_State[0] # level 3
        MU_0 = list( ) # level 2
        M_0 = list( )   # level 2
        X_Hat_0 = list( ) # level  2
        Y_0 = list( ) # level 2
        Y_1 = list( ) # level 2
        X_Pre_0 = list( ) # level 2, the predicted trajectory
        X_Po_All_0 = list( ) # level 2, the prediction of all possible trajectories
        X_Var_0 = list( ) # level 2, the variance of all possible x trajectories of all cars
        Y_Var_0 = list( ) # level 2, the variance of all possible y trajectories of all cars
        p_m = np.diag(np.array([1, 1, 1, 1, 1, 1]))*1e-6 # level 4
        p_0 = list( ) # level 3
        P_0 = list( ) # level 2
        REF_Speed_0 = list( ) # level 2, reference speed of each car
        REF_Lane_0 = list( ) # level 2, reference lane of each car
        REF_Speed_All_0 = list( ) # level 2, reference speed of all modes of each car
        
        # initialize m, mu, and x_hat
        for i in range(N_Car):
            if i == index_EV:
                MU_0.append(None)
                M_0.append(None)
                X_Hat_0.append(None)
                P_0.append(None)
                REF_Speed_0.append(None)
                REF_Lane_0.append(None)
                REF_Speed_All_0.append(None)
            else:
                if (L_Bound[0] <= X_State[0][i][3]) and (X_State[0][i][3] < L_Bound[1]): # the car is on lane 1
                    mu_0 = np.array([0.51, 0.49, 0, 0.0, 0, 0, 0])
                    m_0 = np.argmax(mu_0)
                    MU_0.append(mu_0)
                    M_0.append(m_0)
                    x_hat_m = X_State_0[i] # level 4
                    x_hat_0 = [x_hat_m, x_hat_m, None, None, None, None, None] # level 3
                    p_0 = [p_m, p_m, None, None, None, None, None] # level 3
                    ref_speed_all_0 = [x_hat_m[1], x_hat_m[1], None, None, None, None, None]
                    X_Hat_0.append(x_hat_0) # level 2
                    P_0.append(p_0) # level 2
                    REF_Speed_0.append(ref_speed_all_0[m_0])
                    REF_Lane_0.append(L_Center[0])
                    REF_Speed_All_0.append(ref_speed_all_0)
                elif (L_Bound[1] <= X_State[0][i][3]) and (X_State[0][i][3] < L_Bound[2]):  # the car is on lane 2
                    mu_0 = np.array([0, 0, 0.33, 0.34, 0.33, 0, 0])
                    m_0 = np.argmax(mu_0)
                    MU_0.append(mu_0)
                    M_0.append(m_0)
                    x_hat_m = X_State_0[i] # level 4
                    x_hat_0 = [ None, None, x_hat_m, x_hat_m, x_hat_m, None, None] # level 3
                    p_0 = [None, None, p_m, p_m, p_m, None, None] # level 3
                    ref_speed_all_0 = [ None, None, x_hat_m[1], x_hat_m[1], x_hat_m[1], None, None] # level 3
                    X_Hat_0.append(x_hat_0) # level 2
                    P_0.append(p_0) # level 2
                    REF_Speed_0.append(ref_speed_all_0[m_0])
                    REF_Lane_0.append(L_Center[1])
                    REF_Speed_All_0.append(ref_speed_all_0)
                else: # the car is on lane 3
                    mu_0 = np.array([0, 0, 0, 0, 0, 0.49, 0.51])
                    m_0 = np.argmax(mu_0)
                    MU_0.append(mu_0)
                    M_0.append(m_0)
                    x_hat_m = X_State_0[i] # level 4
                    x_hat_0 = [None, None, None, None, None, x_hat_m, x_hat_m] # level 3
                    p_0 = [None, None, None, None, None, p_m, p_m] # level 3
                    ref_speed_all_0 = [None, None, None, None, None, x_hat_m[1], x_hat_m[1]] # level 3
                    X_Hat_0.append(x_hat_0) # level 2
                    P_0.append(p_0) # level 2
                    REF_Speed_0.append(ref_speed_all_0[m_0])
                    REF_Lane_0.append(L_Center[2])
                    REF_Speed_All_0.append(ref_speed_all_0)
        
        # initialize y and x_pre
        for i in range(N_Car):
            if i == index_EV:
                Y_0.append(None)
                Y_1.append(None)
                X_Pre_0.append(None)
            else:
                K_Lon = Models[M_0[i]][0]
                K_Lat = Models[M_0[i]][1]
                y_0 = H@X_State_0[i] # level 2
                x_pre_0 = self.VelocityTracking(X_State_0[i], X_State_0[i][1], M_0[i], N, K_Lon, K_Lat)
                y_1 = H@x_pre_0[:, 1] # level 3
                Y_0.append(y_0)
                Y_1.append(y_1)
                X_Pre_0.append(x_pre_0)  
        
        # initialize the X_Po_All
        for i in range(N_Car):
            if i == index_EV:
                X_Po_All_0.append(None)
            else:
                temp_tra = list( )
                temp_x_var = list( )
                temp_y_var = list( )
                for j in range(N_M):
                    if np.sum(X_Hat_0[i][j]) == None:
                        temp_tra.append(None)
                        temp_x_var.append(None)
                        temp_y_var.append(None)
                    else:
                        K_Lon = Models[j][0]
                        K_Lat = Models[j][1]
                        temp_tra.append(self.VelocityTracking(X_Hat_0[i][j], X_Hat_0[i][j][1], j, N, K_Lon, K_Lat))
                        temp_x_var.append(np.array([0]*(N+1)))
                        temp_y_var.append(np.array([0]*(N+1)))
                X_Po_All_0.append(temp_tra)
                
        return MU_0, M_0, Y_0, Y_1, X_Hat_0, P_0, X_Pre_0, X_Po_All_0, REF_Speed_0, REF_Lane_0, REF_Speed_All_0
            
    def VelocityTracking(self, x_ini, ref, m, n_step, K_Lon, K_Lat): # velocity tracking model, deterministic
        Ts = self.Ts
        L_Center = self.L_Center
        DSV = self.DSV
        k_lo_1 = K_Lon[0]
        k_lo_2 = K_Lon[1]
        k_la_1 = K_Lat[0]
        k_la_2 = K_Lat[1]
        k_la_3 = K_Lat[2]
        vx_ref = ref 
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
        E = np.array([0, k_lo_1*(Ts**2)/2*vx_ref, k_lo_1*Ts*vx_ref, \
                      (Ts**3)/6*k_la_1*y_ref, (Ts**2)/2*k_la_1*y_ref, Ts*k_la_1*y_ref]) # array no shape
        
        X_KF = np.zeros((DSV, n_step+1))
        X_KF[:, 0] = x_ini
            
        for i in range(1, n_step+1):
            X_KF[:, i] = (F@X_KF[:, i-1]) + E
            
        return X_KF

