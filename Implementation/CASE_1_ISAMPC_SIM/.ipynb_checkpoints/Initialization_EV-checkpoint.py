#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import math
import matplotlib.pyplot as plt
import casadi
import scipy.linalg as sl
from numpy.linalg import matrix_power

class Initialization_EV( ):
    def __init__(self, Params, state_0_glo, state_0_loc):
        
        self.Ts = Params['Ts']
        self.N = Params['N']
        self.Th_MPC = Params['Th_MPC']
        self.l_f = Params['l_f']
        self.l_r = Params['l_r']
        self.N_Lane = Params['N_Lane']
        self.N_M_EV = Params['N_M_EV']
        self.N_Car = Params['N_Car']
        self.L_Center = Params['L_Center']
        self.L_Bound = Params['L_Bound']
        self.DSV = Params['DSV']
        self.DEV = Params['DEV']
        self.infinity = Params['infinity']
        self.SpeedLim = Params['SpeedLim']
        self.Weight= Params['Weight']
        self.l_veh = Params['l_veh']
        self.H = Params['H']
        self.Q_Initial = Params['Q_Initial']
        self.EVplanning = self.contruct_MT_MPC( )
    
    def LookLane(self, y_k): # look for the lane position
        L_Bound = self.L_Bound
        if (L_Bound[0] <= y_k) and (y_k <= L_Bound[1]): # the car is on lane 1
            LanePos = 1
        elif (L_Bound[1] < y_k) and (y_k <= L_Bound[2]):
            LanePos = 2
        elif (L_Bound[2] < y_k) and (y_k <= L_Bound[3]):
            LanePos = 3
        
        return LanePos
    
    def Initialization_MPC(self, state_0_glo, state_0_loc, X_State_0, index_EV, X_Pre_0): # state fusion steps of IMM-KF & define the primary reference speed
        """"
        Input: 
                    state_0_glo is the initial global state, state_0_loc is the initial local state
        Output: 
                    mu_0, m_0,  x_hat_0, x_pre_0, state_1_loc, state_1_glo, y_0,  y_1
        """
        N_M_EV = self.N_M_EV
        N = self.N
        N_Car = self.N_Car
        L_Center = self.L_Center
        N_Lane = self.N_Lane
        SpeedLim = self.SpeedLim
        infinity = self.infinity
        H = self.H
        
        # initialize mu, m, and x_hat, and RefPrim
        if self.LookLane(state_0_glo[3]) == 1:
            mu_0 = np.array([0.6, 0.4, 0])
            m_0 = np.argmax(mu_0)
            x_hat_0 = [state_0_glo, state_0_glo, None]
            RefPrim = [state_0_glo[1], state_0_glo[1], None]
        elif self.LookLane(state_0_glo[3]) == 2:
            mu_0 = np.array([0.3, 0.4, 0.3])
            m_0 = np.argmax(mu_0)
            x_hat_0 = [state_0_glo, state_0_glo, state_0_glo]
            RefPrim = [state_0_glo[1], state_0_glo[1], state_0_glo[1]]
        else:
            mu_0 = np.array([0, 0.4, 0.6])
            m_0 = np.argmax(mu_0)
            x_hat_0 = [None, state_0_glo, state_0_glo]
            RefPrim = [None, state_0_glo[1], state_0_glo[1]]
                
        # update RefPrim
        for i in range(N_M_EV):
            if (np.sum(SpeedLim[i]) != None) and (np.sum(RefPrim[i]) != None): # the lane has nominal speed and the state is not empty 
                RefPrim[i] = SpeedLim[i]
        
        # do the prediction and get y_1, and state of 1
        RefSpeed = RefPrim[m_0]
        Initial = state_0_loc
        Terminal = np.array([L_Center[m_0], RefSpeed]) # y, v
        X_DV = np.array([infinity]*(N + 1))
        Initial = casadi.vertcat(Initial)
        Terminal = casadi.vertcat(Terminal)
        X_DV = casadi.vertcat(X_DV)
        Traj_0, U_0 = self.EVplanning(Initial, Terminal, X_DV)
        Traj_0 = Traj_0.full()
        U_0 = U_0.full()
        rho = U_0[0, :] # soft variable
        snap  = U_0[1, :]  # longitudinal snap 
        alpha = U_0[2, :] # angular acceleration
        
        x_pre_0 = self.V2G(Traj_0)
        state_1_loc = Traj_0[:, 1]
        state_1_glo = x_pre_0[:, 1]
        y_0 = H@state_0_glo
        y_1 = H@state_1_glo
        
        # compute the convex hull for initial
        Lead_full_0 = list( )
        Car_index_lane_0 = list( )
        Temp = infinity*np.ones((N_Lane, N_Car))
        
        for i in range(N_Car):
            if i != index_EV:
                if np.sum(X_State_0[i]) != None:
                    for j in range(N_Lane):
                        if self.LookLane(X_State_0[i][3]) == (j + 1): # pay attention, the lane mark is 1, 2, 3
                            Temp[j, i] = X_State_0[i][0]
        
        for i in range(N_Car):
            Lead_full_0.insert(i, None)
        
        X_DV = [infinity]*(N + 1)
                    
        return mu_0, m_0, x_hat_0, x_pre_0, Traj_0, U_0, state_1_loc, state_1_glo, y_0,  y_1, Lead_full_0, RefSpeed, RefPrim
    
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
            
    def contruct_MT_MPC(self):
        N = self.N
        Nx = self.DEV
        Ts = self.Ts
        Th_MPC = self.Th_MPC
        l_veh = self.l_veh
        Q_Initial = self.Q_Initial
        Q1 = Q_Initial[0]
        Q2 = Q_Initial[1]
        Q3 = Q_Initial[2]
        Q4 = Q_Initial[3]
        Q5 = Q_Initial[4]
        Q6 = Q_Initial[5]
        Q7 = Q_Initial[6]
        
        opti = casadi.Opti( )
        X = opti.variable(Nx, N+1)  # states, from k to k+Np + 1
        Opt_variable = opti.variable(3, N)     # control input, from k to k + N
        rho = Opt_variable[0, :].T # soft variable
        snap  = Opt_variable[1, :].T  # longitudinal snap 
        alpha = Opt_variable[2, :].T # angular acceleration
        Terminal = opti.parameter(2, 1)  # reference state
        Initial  = opti.parameter(Nx, 1)  # initial state
        X_DV = opti.parameter(N + 1, 1) # the leading car status
        
        # model constraint RK4
        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.vehicle_model(X[:, k], snap[k], alpha[k])
            k2 = self.vehicle_model(X[:, k] + Ts / 2 * k1, snap[k], alpha[k])
            k3 = self.vehicle_model(X[:, k] + Ts / 2 * k2, snap[k], alpha[k])
            k4 = self.vehicle_model(X[:, k] + Ts * k3, snap[k], alpha[k])
            x_next = X[:, k] + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next)
       
        # state allocation
        x = X[0, 1::].T # longitudinal position
        y = X[1, 1::].T # lateral position
        v = X[3, 1::].T # longitudinal speed
        a = X[4, 1::].T # longitudinal acceleration
        delta = X[6, 1::].T # front tire angle
        y_error = y[-1] - Terminal[0] # terminal lateral position error
        v_error = v[-1] - Terminal[1] # terminal longitudinal speed error

        # input constraints
        opti.subject_to(0 <= v)
        opti.subject_to(opti.bounded(-8, a, 8))
        opti.subject_to(opti.bounded(-0.8, delta, 0.8))

        # collision avoidance constraints
        opti.subject_to(v*Th_MPC - rho <= X_DV[1::] - x - l_veh/2)
        opti.subject_to(0 <= rho)
        
        # objective function
        J = snap.T@Q1@snap + alpha.T@Q2@alpha + a.T@Q3@a + delta.T@Q4@delta + y_error@Q5@y_error + v_error@Q6@v_error + rho.T@Q7@rho
        
        # solving
        opti.minimize(J)
        opts = {"ipopt.warm_start_init_point": "yes",
                "ipopt.print_level": 0,
                "ipopt.linear_solver": "ma57",
                "print_time": False}
                #'ipopt.max_iter': 100}
        opti.solver('ipopt', opts)

        return opti.to_function('f',[Initial, Terminal, X_DV],[X, Opt_variable])
        
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

