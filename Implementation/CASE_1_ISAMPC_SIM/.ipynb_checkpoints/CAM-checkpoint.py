import numpy as np
from numpy.linalg import matrix_power

class CAM( ): # Constant acceleration model for modeling SV3 in Case 1 and 2
    def __init__(self, Params):
        
        self.Ts       = Params['Ts']
        self.Models   = Params['Models']
        self.DSV      = Params['DSV']
        self.H        = Params['H']
        self.L_Center = Params['L_Center']
        self.acc      = Params['acc']
        self.N        = Params['N']
        self.N_M      = Params['N_M']
        
    def Constant_Acc(self, x0, N_S): 
        Ts = self.Ts
        acc = self.acc
        DSV = self.DSV
        x_state = np.ones((DSV, N_S+1))
        x_state[:, 0] = x0
        for i in range(N_S+1):
            if i == 0:
                x_state[:, 0] = x0
            else:
                x_state[3:5, i] = x_state[3:5, i-1] 
                x_state[2, i] = acc
                x_state[1, i] = x_state[1, i-1] + acc*Ts
                x_state[0, i] = x_state[0, i-1] + x_state[1, i-1]*Ts + 1/2*acc*Ts**2
                if x_state[1, i] < 0:
                    acc = 0
                    x_state[2, i] = acc
                    x_state[1, i] = x_state[1, i-1] + acc*Ts
                    x_state[0, i] = x_state[0, i-1] + x_state[1, i-1]*Ts + 1/2*acc*Ts**2
        
        return x_state
    
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
                      [0, 0, 0, -k_la_1*Ts, -k_la_2*Ts, 1-k_la_3*Ts]]) 
        E = np.array([0, k_lo_1*(Ts**2)/2*vx_ref, k_lo_1*Ts*vx_ref, \
                      (Ts**3)/6*k_la_1*y_ref, (Ts**2)/2*k_la_1*y_ref, Ts*k_la_1*y_ref]) # array no shape
        
        X_KF = np.zeros((DSV, n_step+1))
        X_KF[:, 0] = x_ini
            
        for i in range(1, n_step+1):
            X_KF[:, i] = (F@X_KF[:, i-1]) + E
            
        return X_KF
    
    def Final_Return(self, k, X_Hat, Y, car_index): # Return computation results
        H = self.H
        N = self.N
        N_M = self.N_M
        Models = self.Models
        L_Center = self.L_Center
        
        x_hat_k = list( )
        x_hat_k_1 = X_Hat[k - 1][car_index] 
        
        mu_k = np.array([0, 0, 0, 0, 0, 0, 1])
        REF_Speed = [None, None, None, None, None, None, 0]
        m_k = 6
        
        for i in range(N_M):
            if mu_k[i] == 0: 
                x_hat_k.append(None)
            else:
                Temp = self.Constant_Acc(x_hat_k_1[i], 1)
                x_hat_k.append(Temp[:, 1])
        
        p_k = None
        x_state_k = x_hat_k[m_k]
        x_pre_k = self.Constant_Acc(x_state_k, N)
        y_k_plus_1 = H@x_pre_k[:, 1] 
        
        x_po_all_k = list( )
        x_var_k = list( ) 
        y_var_k = list( )
        
        for i in range(N_M):
            if np.sum(x_hat_k[i]) == None:
                temp_tra = None
                temp_x_var = None
                temp_y_var = None
            else:
                if i == m_k:
                    temp_tra = x_pre_k
                else:
                    K_Lon = Models[i][0]
                    K_Lat = Models[i][1]
                    temp_tra = self.VelocityTracking(x_hat_k[i], x_hat_k[i][1], i, N, K_Lon, K_Lat)
                
                temp_x_var = np.array([0]*(N+1))
                temp_y_var = np.array([0]*(N+1))
                    
            x_po_all_k.append(temp_tra)
            x_var_k.append(temp_x_var)
            y_var_k.append(temp_y_var)
                
        return REF_Speed[m_k], L_Center[2], REF_Speed, mu_k, m_k, x_hat_k, p_k, x_state_k, x_pre_k, y_k_plus_1, x_po_all_k, x_var_k, y_var_k
                    

    
