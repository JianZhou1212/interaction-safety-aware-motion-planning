#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import math
import matplotlib.pyplot as plt
import casadi
import scipy.linalg as sl
from numpy.linalg import matrix_power
import scipy.stats as stats

class Driver_Model( ):
    def __init__(self, Params):
        self.Ts = Params['Ts']
        self.DSV = Params['DSV']
        self.H = Params['H']
        self.L_Center = Params['L_Center']
        self.N = Params['N']
        self.SpeedLim = Params['SpeedLim']
        self.L_Bound = Params['L_Bound']

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
        mu_x, sigma_x = 0, 1
        lower_x, upper_x = mu_x - 2*sigma_x, mu_x + 2*sigma_x  
        x = stats.truncnorm((lower_x - mu_x)/sigma_x, (upper_x - mu_x)/sigma_x, loc = mu_x, scale = sigma_x)
        
        mu_y, sigma_y = 0, 0.2
        lower_y, upper_y = mu_y - 2*sigma_y, mu_y + 2*sigma_y  
        y = stats.truncnorm((lower_y - mu_y)/sigma_y, (upper_y - mu_y)/sigma_y, loc = mu_y, scale = sigma_y)

        for i in range(1, n_step+1):
            X_KF[:, i] = (F@X_KF[:, i-1]) + E #+ np.array([x.rvs(1)[0], 0, 0, y.rvs(1)[0], 0, 0])
            
        return X_KF
    
    def Final_Return(self, k, X_State_LC, K_Lon, K_Lat):
        H = self.H
        N = self.N
        SpeedLim = self.SpeedLim
        L_Bound = self.L_Bound
        
        y_k = X_State_LC[k][3]
        if (y_k <= L_Bound[1]): 
            m_decision = None
        elif (L_Bound[1] < y_k) and (y_k <= L_Bound[2]):
            m_decision = 3
        elif (L_Bound[2] < y_k):
            m_decision = 5
            
        x_pre_k = self.VelocityTracking(X_State_LC[k], SpeedLim[1], m_decision, N, K_Lon, K_Lat)
        x_state_k_plus_1 = x_pre_k[:, 1]
        y_k_plus_1 = H@x_state_k_plus_1
    
        return y_k_plus_1, x_state_k_plus_1
                    

    
