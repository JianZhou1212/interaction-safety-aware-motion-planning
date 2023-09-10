import numpy as np
import math
import matplotlib.pyplot as plt
import casadi
import scipy.linalg as sl
from numpy.linalg import matrix_power
import pdb

class IAIMM_KF( ): # The IMM-KF for motion prediction
    def __init__(self, Params):
        self.Ts             = Params['Ts']
        self.N              = Params['N']
        self.Models         = Params['Models']
        self.std_parameters = Params['std_parameters']
        self.N_Lane       = Params['N_Lane']
        self.N_M          = Params['N_M']
        self.N_Car        = Params['N_Car']
        self.L_Width      = Params['L_Width']
        self.L_Bound      = Params['L_Bound']
        self.L_Center     = Params['L_Center']
        self.DSV          = Params['DSV']
        self.infinity     = Params['infinity']
        self.SpeedLim     = Params['SpeedLim']
        self.Q            = Params['Q']
        self.R            = Params['R']
        self.Weight       = Params['Weight']
        self.w_veh        = Params['w_veh']
        self.l_veh        = Params['l_veh']
        self.H            = Params['H']
        self.K_sampling   = Params['K_sampling']
        self.LongVelProj  = self.construct_QCQP( )
        self.LongSampSymb = self.LongSampling( )
        self.LatSampSymb  = self.LatrSampling( )
        
    def Fusion_Prim_Speed(self, mu_k_1, x_hat_k_1, y_pos_k_1, y_pos_k, p_k_1): # state fusion steps of IMM-KF & define the primary reference speed of each mode of each SV
        DSV = self.DSV
        N_M = self.N_M
        SpeedLim = self.SpeedLim

        L_pos_k_1 = self.LookLane(y_pos_k_1)    
        L_pos_k = self.LookLane(y_pos_k)        
        Pr = self.ProTrans(L_pos_k_1, L_pos_k)  
        c = list( )
        x_bar = list( )
        p_bar = list( )
        
        for i in range(N_M):
            c.append(mu_k_1@Pr[:, i])
        c = np.array(c)
        
        for i in range(N_M):
            if c[i] != 0:
                temp_x = np.array([0]*DSV)
                for j in range(N_M):
                    if mu_k_1[j] != 0:
                        temp_x = temp_x + Pr[j, i]*mu_k_1[j]/c[i]*x_hat_k_1[j]
                    else:
                        temp_x = temp_x + 0
                x_bar.append(temp_x) 
            else:
                x_bar.append(None)
                
        for i in range(N_M):
            if c[i] != 0:
                temp_p = np.zeros((DSV, DSV))
                for j in range(N_M):
                    if mu_k_1[j] != 0:
                        X_k_k = x_bar[i] - x_hat_k_1[j] 
                        temp_p = temp_p + Pr[j, i]*mu_k_1[j]/c[i]*(p_k_1[j] +  (np.array([X_k_k]).T)@(np.array([X_k_k])))
                    else:
                        temp_p = temp_p + 0
                    p_bar.append(temp_p)
            else:
                p_bar.append(None)
                              
        RefPrim = list( )
        for i in range(N_M):
            if (np.sum(x_bar[i])== None):
                RefPrim.append(None)
            else:
                RefPrim.append(x_bar[i][1])
        RefPrim = np.array(RefPrim)
        
        for i in range(N_M):
            if (i == 0) or (i == 2): 
                SpeedLim_Lane = SpeedLim[0]
            elif (i == 1) or (i == 3) or (i == 5): 
                SpeedLim_Lane = SpeedLim[1]
            elif (i == 4) or (i == 6): 
                SpeedLim_Lane = SpeedLim[2]
            if (SpeedLim_Lane != None) and (RefPrim[i] != None): 
                RefPrim[i] = SpeedLim_Lane
             
        return c, x_bar, p_bar, RefPrim 
    
    def LookLane(self, y_k): # check lane index according to the current lateral position
        L_Bound = self.L_Bound
        if (y_k <= L_Bound[1]): 
            LanePos = 1
        elif (L_Bound[1] < y_k) and (y_k <= L_Bound[2]):
            LanePos = 2
        elif (L_Bound[2] < y_k):
            LanePos = 3
        
        return LanePos
    
    def KalmanFilter(self, mu_k_1, x_hat_k_1, y_k, y_pos_k_1, y_pos_k, p_k_1): # Run regular Kalman Filter at current step
        N_M = self.N_M
        Models = self.Models                                                           
        Q = self.Q
        R = self.R
        H= self.H
        DSV = self.DSV
        
        c, x_bar, p_bar, RefPrim = self.Fusion_Prim_Speed(mu_k_1, x_hat_k_1, y_pos_k_1, y_pos_k, p_k_1)
        x_hat_k = list( ) 
        p_k = list( )     
        y_tilde_k = list( )
        s_k = list( )      
        
        for i in range(N_M):
            if (np.sum(x_bar[i]) == None):
                x_hat_k.append(None)
                p_k.append(None)
                y_tilde_k.append(None)
                s_k.append(None)
            else:
                K_Lon = Models[i][0]
                K_Lat = Models[i][1]
                F, X_KF = self.VelocityTracking(x_bar[i], RefPrim[i], i, 1, K_Lon, K_Lat)
                x_hat_k_k_1 = X_KF[:, 1] 
                p_k_k_1 = F@p_bar[i]@(F.T) + Q   
                y_tilde = y_k - H@x_hat_k_k_1    
                s = H@p_k_k_1@(H .T)+ R  
                k_k = p_k_k_1@H.T@np.linalg.pinv(s) 
                x_hat_k.append((x_hat_k_k_1 + k_k@y_tilde))
                p_k.append((np.eye(DSV) - k_k@H)@p_k_k_1)   
                y_tilde_k.append(y_tilde)
                s_k.append(s)
            
        return x_hat_k, p_k, y_tilde_k, s_k, c, RefPrim
            
    def VelocityTracking(self, x_ini, vx_ref, m, n_step, K_Lon, K_Lat): # velocity tracking model
        Ts = self.Ts
        L_Center = self.L_Center
        DSV = self.DSV
        if (m == 0) or (m == 2):
            y_ref = L_Center[0]
        elif (m == 1) or (m == 3) or (m == 5):
            y_ref = L_Center[1]
        elif (m == 4) or (m == 6):
            y_ref = L_Center[2]
            
        F = np.array([[1, Ts, Ts**2/2, 0, 0, 0],
                      [0, 1-K_Lon[0]*Ts**2/2, Ts-K_Lon[1]*(Ts**2)/2, 0, 0, 0],
                      [0, -K_Lon[0]*Ts, 1-K_Lon[1]*Ts, 0, 0, 0],
                      [0, 0, 0, 1-K_Lat[0]*(Ts**3)/6, Ts-K_Lat[1]*(Ts**3)/6, Ts**2/2-K_Lat[2]*(Ts**3)/6],
                      [0, 0, 0, -K_Lat[0]*(Ts**2)/2, 1-K_Lat[1]*(Ts**2)/2, Ts-K_Lat[2]*(Ts**2)/2],
                      [0, 0, 0, -K_Lat[0]*Ts, -K_Lat[1]*Ts, 1-K_Lat[2]*Ts]]) # shape = 6 x 6
        E = np.array([0, K_Lon[0]*(Ts**2)/2*vx_ref, K_Lon[0]*Ts*vx_ref, \
                      (Ts**3)/6*K_Lat[0]*y_ref, (Ts**2)/2*K_Lat[0]*y_ref, Ts*K_Lat[0]*y_ref]) # array no shape
        X_KF = np.zeros((DSV, n_step+1))
        X_KF[:, 0] = x_ini   
        for i in range(1, n_step+1):
            X_KF[:, i] = (F@X_KF[:, i-1]) + E
            
        return F, X_KF
        
    def LaneTracking(self, initial_y, m, K_Lat): # lane tracking model
        L_Center = self.L_Center
        Ts = self.Ts
        N = self.N
        if (m == 0) or (m == 2):
            y_ref = L_Center[0]
        elif (m == 1) or (m == 3) or (m == 5):
            y_ref = L_Center[1]
        elif (m == 4) or (m == 6):
            y_ref = L_Center[2]
            
        A = np.array([[1-K_Lat[0]*(Ts**3)/6, Ts-K_Lat[1]*(Ts**3)/6, (Ts**2)/2-K_Lat[2]*(Ts**3)/6],
                     [-K_Lat[0]*(Ts**2)/2, 1-K_Lat[1]*(Ts**2)/2, Ts-K_Lat[2]*(Ts**2)/2],
                     [-K_Lat[0]*Ts, -K_Lat[1]*Ts, 1-K_Lat[2]*Ts]])
        B = np.array([(Ts**3)/6*K_Lat[0], (Ts**2)/2*K_Lat[0], Ts*K_Lat[0]])*y_ref
        Y = np.zeros((3, N + 1))
        Y[:, 0] = initial_y
        for i in range(1, N+1):
            Y[:, i] =  (A@Y[:, i-1]) + B
            
        return Y[0, :] 
            
        
    def ProjectSpeed(self, Obst_k, x_hat_k, RefPrim, car_index): # Update the reference speed of each mode of each SV using optimization
        N_M = self.N_M
        N = self.N
        Ts = self.Ts
        infinity = self.infinity
        Models = self.Models
        DSV = self.DSV
        H = self.H
        N_Car = self.N_Car
        
        ProjVal = list( ) 
        for i in range(N_M): 
            if np.sum(x_hat_k[i]) == None: 
                ProjVal.append(None)
            else:                         
                K_Lon = Models[i][0]
                K_Lat = Models[i][1]
                a = np.array([[1, Ts, (Ts**2)/2], [0, 1-K_Lon[0]*(Ts**2)/2, Ts-K_Lon[1]*(Ts**2)/2], [0, -K_Lon[0]*Ts, 1-K_Lon[1]*Ts]])
                b = np.array([0, K_Lon[0]*(Ts**2)/2, K_Lon[0]*Ts])
                initial_x = x_hat_k[i][0:3] 
                initial_y = x_hat_k[i][3::] 
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
                    if j != car_index:
                        if np.sum(Obst_k[j]) == None:
                            SEL = SEL + [0]*N 
                            X_SV = X_SV + [infinity]*N
                        else:
                            SEL = SEL + self.Sel_Matrix_Diag(K_Lat, initial_y, Obst_k[j][3], i)
                            X_SV = X_SV + Obst_k[j][0, 1::].tolist()
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

        return ProjVal
        
    def Final_Return(self, k, MU, X_Hat, P, Y, Obst_k, car_index): # Return computation results
        Ts = self.Ts
        N = self.N
        N_M = self.N_M
        L_Center = self.L_Center
        Q = self.Q
        Weight = self.Weight
        H = self.H
        Models = self.Models
        DSV = self.DSV
        mu_k_1 = MU[k - 1][car_index]       
        x_hat_k_1 = X_Hat[k - 1][car_index] 
        p_k_1 = P[k - 1][car_index]       
        y_pos_k = Y[k][car_index][-1]     
        y_pos_k_1 = Y[k-1][car_index][-1] 
        y_k = Y[k][car_index]
        ActPse = np.array([None]*N_M)
        x_hat_k, p_k, y_tilde_k, s_k, c, RefPrim = self.KalmanFilter(mu_k_1, x_hat_k_1, y_k, y_pos_k_1, y_pos_k, p_k_1)
        REF = self.ProjectSpeed(Obst_k, x_hat_k, RefPrim, car_index) 
        t = np.arange(0, Ts*(N + 1), Ts, dtype = float)
        for i in range(N_M):
            if np.sum(x_hat_k[i]) == None: 
                ActPse[i] = None
            else:
                K_Lon = Models[i][0]
                K_Lat = Models[i][1] 
                _, X = self.VelocityTracking(x_hat_k[i], REF[i], i, N, K_Lon, K_Lat)
                ax = X[2, :]*X[2, :]
                ay = X[5, :]*X[5, :]
                
                if (i == 0) or (i == 2):
                    y_ref = L_Center[0]
                elif (i == 1) or (i == 3) or (i == 5):
                    y_ref = L_Center[1]
                elif (i == 4) or (i == 6):
                    y_ref = L_Center[2]
                    
                ActPse[i] = Weight[0]*np.trapz(ax, t) + Weight[1]*np.trapz(ay, t) + \
                Weight[2]*((REF[i] - x_hat_k[i][1])**2) + Weight[3]*((y_ref - x_hat_k[i][3])**2)
                ActPse[i] = ActPse[i] + 0.0001
            
        L = np.array([None]*N_M)
        y_tilde_aug = list( )
        s_aug = list( )
        for i in range(N_M):
            if c[i] == 0:
                y_tilde_aug.append(None)
                s_aug.append(None)
                L[i] = 0
            else:
                y_tilde_aug.append(np.append(y_tilde_k[i], np.sqrt(ActPse[i])))
                s_aug.append(sl.block_diag(s_k[i], ActPse[i]*0.1))
                L[i] = np.exp(-1/2*y_tilde_aug[i]@np.linalg.pinv(s_aug[i])@y_tilde_aug[i])/np.sqrt(np.linalg.det(2*math.pi*s_aug[i]))
        temp = c@L
        mu_k = c*L/temp
        m_k = np.argmax(mu_k)
        x_state_k = np.array([0]*DSV)
        for i in range(N_M):
            if (np.sum(x_hat_k[i]) != None):
                x_state_k = x_state_k + x_hat_k[i]*mu_k[i]
                
        _, x_pre_k = self.VelocityTracking(x_state_k, REF[m_k], m_k, N, Models[m_k][0], Models[m_k][1]) 
        
        if (m_k == 0) or (m_k == 2):
            ref_lane = L_Center[0]
        elif (m_k == 1) or (m_k == 3) or (m_k == 5):
            ref_lane = L_Center[1]
        elif (m_k == 4) or (m_k == 6):
            ref_lane = L_Center[2]
        y_k_plus_1 = H@x_pre_k[:, 1] 
        
        x_po_all_k = list( )
        x_var_k = list( ) 
        y_var_k = list( ) 
        for i in range(N_M):
            if np.sum(x_hat_k[i]) == None:
                temp_tra = None
                Var_y = None
                Var_x = None
            else:
                K_Lon = Models[i][0]
                K_Lat = Models[i][1]
                _, temp_tra = self.VelocityTracking(x_hat_k[i], REF[i], i, N, K_Lon, K_Lat)
                
                x_ini = x_hat_k[i][0:3]
                x_ref = REF[i]
                y_ini = x_hat_k[i][3::]
                if (i == 0) or (i == 2):
                    y_ref = L_Center[0]
                elif (i == 1) or (i == 3) or (i == 5):
                    y_ref = L_Center[1]
                elif (i == 4) or (i == 6):
                    y_ref = L_Center[2]
                    
                Var_y, Var_x = self.EstimateUncertainty(y_k[2], i, x_ini, x_ref, y_ini, y_ref) 
                
            x_po_all_k.append(temp_tra)
            x_var_k.append(Var_x)
            y_var_k.append(Var_y)

        return REF[m_k], ref_lane, mu_k, m_k, x_hat_k, p_k, x_state_k, x_pre_k, y_k_plus_1, REF, x_po_all_k, x_var_k, y_var_k
    
    def LatrSampling(self): # Parametermize the lateral lane-tracking model for sampling to estimate the standard deviation
        Ts = self.Ts
        N = self.N
        y_ini = casadi.SX.sym('y_ini', 3, 1)
        y_ref = casadi.SX.sym('y_ref')
        k_lat = casadi.SX.sym('k_lat', 3, 1)
        A = casadi.SX.zeros((3, 3))
        B = casadi.SX.zeros((3, 1))
        A[0, 0] = 1-k_lat[0]*(Ts**3)/6
        A[0, 1] = Ts-k_lat[1]*(Ts**3)/6
        A[0, 2] = (Ts**2)/2-k_lat[2]*(Ts**3)/6
        A[1, 0] = -k_lat[0]*(Ts**2)/2
        A[1, 1] = 1-k_lat[1]*(Ts**2)/2
        A[1, 2] = Ts-k_lat[2]*(Ts**2)/2
        A[2, 0] = -k_lat[0]*Ts
        A[2, 1] = -k_lat[1]*Ts
        A[2, 2] = 1-k_lat[2]*Ts
        B[0, 0] = (Ts**3)/6*k_lat[0]*y_ref
        B[1, 0] = (Ts**2)/2*k_lat[0]*y_ref
        B[2, 0] = Ts*k_lat[0]*y_ref
        Y = casadi.SX.zeros((3, N + 1))
        Y[:, 0] = y_ini
        
        for i in range(1, N + 1):
            Y[:, i] =  (A@Y[:, i-1]) + B
        
        lateral_position = Y[0, :]
        
        return casadi.Function('lat_f', [y_ini, y_ref, k_lat], [lateral_position])
    
    def LongSampling(self): # Parametermize the longitudinal velocity-tracking model for sampling to estimate the standard deviation
        Ts = self.Ts
        N = self.N

        x_ini = casadi.SX.sym('x_ini', 3, 1)
        x_ref = casadi.SX.sym('x_ref')
        k_lon = casadi.SX.sym('k_lon', 2, 1)
        A = casadi.SX.zeros((3, 3))
        B = casadi.SX.zeros((3, 1))
        A[0, 0] = 1
        A[0, 1] = Ts
        A[0, 2] = (Ts**2)/2
        A[1, 0] = 0
        A[1, 1] = 1-k_lon[0]*Ts**2/2
        A[1, 2] = Ts-k_lon[1]*(Ts**2)/2
        A[2, 0] = 0
        A[2, 1] = -k_lon[0]*Ts
        A[2, 2] = 1-k_lon[1]*Ts
        B[0, 0] = 0
        B[1, 0] = k_lon[0]*(Ts**2)/2*x_ref
        B[2, 0] = k_lon[0]*Ts*x_ref
        X = casadi.SX.zeros((3, N + 1))
        X[:, 0] = x_ini
        
        for i in range(1, N + 1):
            X[:, i] =  (A@X[:, i-1]) + B
        
        long_position = X[0, :]
        
        return casadi.Function('lon_f', [x_ini, x_ref, k_lon], [long_position])
    
    def EstimateUncertainty(self, y_k, m, x_ini, x_ref, y_ini, y_ref): # estimate standard deviation in prediction horizon (sampling-based approach)
        N = self.N
        L_Bound = self.L_Bound
        std_parameters = self.std_parameters
        K_sampling = self.K_sampling
        if (m == 0) or (m == 3) or (m == 6): 
            var_y = np.array([std_parameters[m][1][0]**2]*(N + 1))
            var_y[0] = 0
            K_set_lon = std_parameters[m][0]
            x_observation = np.ones((K_sampling, N + 1))
            for i in range(K_sampling):
                r = np.random.randint(0, len(K_set_lon) - 1)
                k_lon = K_set_lon[r][0]
                x_long = self.LongSampSymb(x_ini, x_ref, k_lon)
                x_observation[i, :] = x_long.full()
            var_x = np.var(x_observation, axis = 0)
        else: 
            if m == 1: 
                if y_k <= L_Bound[1]: 
                    K_set_lon = std_parameters[m][0]
                    K_set_lat = std_parameters[m][1]
                    x_observation = np.ones((K_sampling, N + 1))
                    y_observation = np.ones((K_sampling, N + 1))
                    for i in range(K_sampling):
                        r = np.random.randint(0, len(K_set_lon) - 1)
                        k_lon = K_set_lon[r][0]
                        k_lat = K_set_lat[r][0]
                        x_long    = self.LongSampSymb(x_ini, x_ref, k_lon)
                        y_lateral = self.LatSampSymb(y_ini, y_ref, k_lat)
                        x_observation[i, :] = x_long.full()
                        y_observation[i, :] = y_lateral.full()
                    var_x = np.var(x_observation, axis = 0)
                    var_y = np.var(y_observation, axis = 0)
                else: 
                    var_y = np.array([std_parameters[3][1][0]**2]*(N + 1)) 
                    var_y[0] = 0
                    K_set_lon = std_parameters[m][0]
                    x_observation = np.ones((K_sampling, N + 1))
                    for i in range(K_sampling):
                        r = np.random.randint(0, len(K_set_lon) - 1)
                        k_lon = K_set_lon[r][0]
                        x_long = self.LongSampSymb(x_ini, x_ref, k_lon)
                        x_observation[i, :] = x_long.full()
                    var_x = np.var(x_observation, axis = 0)
            if m == 2:
                if L_Bound[1] <= y_k: 
                    K_set_lon = std_parameters[m][0]
                    K_set_lat = std_parameters[m][1]
                    x_observation = np.ones((K_sampling, N + 1))
                    y_observation = np.ones((K_sampling, N + 1))
                    for i in range(K_sampling):
                        r = np.random.randint(0, len(K_set_lon) - 1)
                        k_lon = K_set_lon[r][0]
                        k_lat = K_set_lat[r][0]
                        x_long    = self.LongSampSymb(x_ini, x_ref, k_lon)
                        y_lateral = self.LatSampSymb(y_ini, y_ref, k_lat)
                        x_observation[i, :] = x_long.full()
                        y_observation[i, :] = y_lateral.full()
                    var_x = np.var(x_observation, axis = 0)
                    var_y = np.var(y_observation, axis = 0)
                else: 
                    var_y = np.array([std_parameters[0][1][0]**2]*(N + 1)) 
                    var_y[0] = 0
                    K_set_lon = std_parameters[m][0]
                    x_observation = np.ones((K_sampling, N + 1))
                    for i in range(K_sampling):
                        r = np.random.randint(0, len(K_set_lon) - 1)
                        k_lon = K_set_lon[r][0]
                        x_long = self.LongSampSymb(x_ini, x_ref, k_lon)
                        x_observation[i, :] = x_long.full()
                    var_x = np.var(x_observation, axis = 0)
            if m == 4:
                if y_k <= L_Bound[2]: 
                    K_set_lon = std_parameters[m][0]
                    K_set_lat = std_parameters[m][1]
                    x_observation = np.ones((K_sampling, N + 1))
                    y_observation = np.ones((K_sampling, N + 1))
                    for i in range(K_sampling):
                        r = np.random.randint(0, len(K_set_lon) - 1)
                        k_lon = K_set_lon[r][0]
                        k_lat = K_set_lat[r][0]
                        x_long = self.LongSampSymb(x_ini, x_ref, k_lon)
                        y_lateral = self.LatSampSymb(y_ini, y_ref, k_lat)
                        x_observation[i, :] = x_long.full()
                        y_observation[i, :] = y_lateral.full()
                    var_x = np.var(x_observation, axis = 0)
                    var_y = np.var(y_observation, axis = 0)
                else:
                    var_y = np.array([std_parameters[6][1][0]**2]*(N + 1)) 
                    var_y[0] = 0
                    K_set_lon = std_parameters[m][0]
                    x_observation = np.ones((K_sampling, N + 1))
                    for i in range(K_sampling):
                        r = np.random.randint(0, len(K_set_lon) - 1)
                        k_lon = K_set_lon[r][0]
                        x_long = self.LongSampSymb(x_ini, x_ref, k_lon)
                        x_observation[i, :] = x_long.full()
                    var_x = np.var(x_observation, axis = 0)
            if m == 5: 
                if L_Bound[2] <= y_k: 
                    K_set_lon = std_parameters[m][0]
                    K_set_lat = std_parameters[m][1]
                    x_observation = np.ones((K_sampling, N + 1))
                    y_observation = np.ones((K_sampling, N + 1))
                    for i in range(K_sampling):
                        r = np.random.randint(0, len(K_set_lon) - 1)
                        k_lon = K_set_lon[r][0]
                        k_lat = K_set_lat[r][0]
                        x_long = self.LongSampSymb(x_ini, x_ref, k_lon)
                        y_lateral = self.LatSampSymb(y_ini, y_ref, k_lat)
                        x_observation[i, :] = x_long.full()
                        y_observation[i, :] = y_lateral.full()
                    var_x = np.var(x_observation, axis = 0)
                    var_y = np.var(y_observation, axis = 0)
                else: 
                    var_y = np.array([std_parameters[3][1][0]**2]*(N + 1))
                    var_y[0] = 0
                    K_set_lon = std_parameters[m][0]
                    x_observation = np.ones((K_sampling, N + 1))
                    for i in range(K_sampling):
                        r = np.random.randint(0, len(K_set_lon) - 1)
                        k_lon = K_set_lon[r][0]
                        x_long = self.LongSampSymb(x_ini, x_ref, k_lon)
                        x_observation[i, :] = x_long.full()
                    var_x = np.var(x_observation, axis = 0)
           
        return var_y, var_x
        
    def ProTrans(self, Pos_k_1, Pos_k): # define the probability transformation matrix in IAIMM-KF, you can manually design it
        N_M = self.N_M
        
        Pr = np.zeros((N_M, N_M))
        if (Pos_k_1 == 1) and (Pos_k == 1):    # the car is in Lane 1
            Pr_active = np.array([[0.6, 0.4],  # m0 → m0, m1
                                  [0.4, 0.6]]) # m1 → m0, m1
            Pr[0:2, 0:2] = Pr_active
                
        elif (Pos_k_1 == 1) and (Pos_k == 2):         # from lane 1 to lane 2
            Pr_active = np.array([[0.3, 0.5, 0.2],    # m0 → m2, m3, m4
                                  [0.25, 0.5, 0.25]]) # m1 → m2, m3, m4
            Pr[0:2, 2:5] = Pr_active
        
        elif (Pos_k_1 == 2) and (Pos_k == 1):   # jump from lane 2 to lane 1
            Pr_active = np.array([[0.6, 0.4],   # m2 → m0, m1 
                                  [0.6, 0.4],   # m3 → m0, m1
                                  [0.6, 0.4]])  # m4 → m0, m1 
            Pr[2:5, 0:2] = Pr_active
        
        elif (Pos_k_1 == 2) and (Pos_k == 2):         # the car is is in Lane 2
            Pr_active = np.array([[0.5, 0.25, 0.25],   # m2  → m2, m3, m4
                                  [0.25, 0.5, 0.25],   # m3  → m2, m3, m4
                                  [0.25, 0.25, 0.5]])  # m4  → m2, m3, m4
            Pr[2:5, 2:5] = Pr_active
                
        elif (Pos_k_1 == 2) and (Pos_k == 3):  # jump from lane 2 to lane 3
            Pr_active = np.array([[0.4, 0.6],  # m2 → m5, m6
                                  [0.4, 0.6],  # m3 → m5, m6
                                  [0.4, 0.6]]) # m4 → m5, m6
            Pr[2:5, 5::] = Pr_active
        
        elif (Pos_k_1 == 3) and (Pos_k == 2):         # jump from lane 3 to lane 2
            Pr_active = np.array([[0.25, 0.5, 0.25],  # m5 → m2, m3, m4
                                  [0.3, 0.4, 0.3]])   # m6 → m2, m3, m4
            Pr[5::, 2:5] = Pr_active          
        
        elif (Pos_k_1 == 3) and (Pos_k == 3):   # k-1 and k are in Lane 3
            Pr_active = np.array([[0.7, 0.3],   # m5 → m5, m6
                                  [0.3, 0.7]])  # m6 → m5, m6
            Pr[5::, 5::] = Pr_active
        
        return Pr
    
    def Sel_Matrix_Diag(self, K_Lat, initial_y, y_SV, m): # Find the time steps over the horizon where the EV prediction does not have collision with SV occupancy, given the initial state and mode (candidate maneuver)
        N = self.N
        w_veh = self.w_veh
        y_EV = self.LaneTracking(initial_y, m, K_Lat)
        y_EV = y_EV[1:] 
        y_SV = y_SV[1:]
        SEL = [0]*N 
        for i in range(N):
            if np.abs(y_EV[i] - y_SV[i]) <= w_veh:
                SEL[i] = 1
            
        return SEL
    
    def construct_QCQP(self): # The QCQP problem for computing the collision-free reference speed of each mode of each SV (Note in the original method it was an mixted-interger programming)
        N = self.N
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
        D_Error = np.multiply(D_Error, D_Error)
        Safe_D = np.ones((N*(N_Car-1), 1))*(l_veh**2)
        J = (v_pri - v_up)**2
        opti.minimize(J)
        opti.subject_to(H@Safe_D <= H@D_Error)
        opts = {"ipopt.warm_start_init_point": "yes",
                "ipopt.print_level": 0,
                "ipopt.linear_solver": "ma57", # You can comment this line of you do not have ma57
                "print_time": False}
        opti.solver('ipopt', opts)
        
        return opti.to_function('f', [A, B, X_SV, v_pri, H], [v_up])