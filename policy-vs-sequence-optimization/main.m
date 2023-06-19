clc
clear
close all
T = 0.25;
K_lon = [0 0.3847 0.8663];
K_lat = [0.5681 1.4003 1.7260];

parameters.L_width = 4;
parameters.x0_e = 0;
parameters.y0_e = 2;
parameters.vx0_e = 20;
parameters.vy0_e = 0;
parameters.vx_ref_e = 20;
parameters.y_ref_e = parameters.y0_e;
parameters.ax_bound = 5;
parameters.ay_bound = 1;
parameters.x0_s = 0;
parameters.y0_s = 6;
parameters.vx0_s = 20;
parameters.vy0_s = 0;
parameters.ax0_s = 0;
parameters.ay0_s = 0;
parameters.K_lon = [0 0.3847 0.8663];
parameters.K_lat = [0.5681 1.4003 1.7260];
parameters.T = T;
parameters.N = 20;
parameters.A = [1 T T^2/2 0 0 0;
                0 1 T 0 0 0;
                0 0 1 0 0 0;
                0 0 0 1 T T^2/2;
                0 0 0 0 1 T;
                0 0 0 0 0 1];
parameters.B = [T^3/6 0;
                T^2/2 0;
                T 0;
                0 T^3/6;
                0 T^2/2;
                0 T];
parameters.A_ev = [0 1 0 0;
                   0 0 0 0;
                   0 0 0 1;
                   0 0 0 0];
parameters.B_ev = [0 0;
                   1 0;
                   0 0;
                   0 1];
parameters.K = [0 0.3847 0.8663 0 0 0;
                0 0 0 0.5681 1.4003 1.7260];
parameters.ref_1 = [0 parameters.vx0_s 0 2 0 0]';
parameters.ref_2 = [0 parameters.vx0_s 0 6 0 0]';
parameters.l_veh = 4;
parameters.w_veh = 2;
parameters.infx = 10000;
parameters.N_bar = 10;

k_max = 35;
h = 1;
for p = 0.1:0.1:0.5

x_current_sv = [parameters.x0_s parameters.vx0_s parameters.ax0_s parameters.y0_s parameters.vy0_s parameters.ay0_s];
x_current_ev_seq = [parameters.x0_e parameters.vx0_e parameters.y0_e parameters.vy0_e];
x_current_ev_pol = x_current_ev_seq;

State_EV_sequence = ones(4, k_max + 1);
State_EV_sequence(:, 1) = x_current_ev_seq;
U_EV_sequence = ones(2, k_max);
J_EV_sequence = ones(1, k_max);

State_EV_policy = ones(4, k_max + 1);
State_EV_policy(:, 1) = x_current_ev_pol;
U_EV_policy = ones(2, k_max);
J_EV_policy = ones(1, k_max);

State_SV = ones(6, k_max + 1);
State_SV(:, 1) = x_current_sv;

SV = SVModel(parameters);
EV = EVControl(parameters);

Time_sequence = ones(1, k_max);
Time_policy = ones(1, k_max);

p1 = ones(1, k_max);
p2 = ones(1, k_max);

for k = 1:k_max
    if k <= 5
        p1(k) = 0;
        p2(k) = 1;
    elseif (6 <= k) && (k <= 25)
        p1(k) = p;
        p2(k) = 1 - p1(k);
    else
        p1(k) = 0;
        p2(k) = 1;
    end
end


for k = 1:k_max
    [x_next_sv, x_sv_prediction_1, x_sv_prediction_2] = SV.solve(x_current_sv);
    x_current_sv = x_next_sv;
    State_SV(:, k + 1) = x_current_sv;
    
    
    [x_next_ev_seq, u_current_ev_seq, u_opt_ev_seq, x_opt_ev_seq, J_opt_ev_seq, toc_seq] = EV.solve_sequence(x_current_ev_seq, x_sv_prediction_1, x_sv_prediction_2, p1(k), p2(k));
   
    Time_sequence(k) = toc_seq;
    x_current_ev_seq = x_next_ev_seq;
    State_EV_sequence(:, k + 1) = x_current_ev_seq;
    U_EV_sequence(:, k) = u_current_ev_seq;
    J_EV_sequence(k) = J_opt_ev_seq;

    [x_next_ev_pol, u_current_ev_pol, u_mode_1_opt_ev_pol, x_mode_1_opt_ev_pol, u_mode_2_opt_ev_pol, x_mode_2_opt_ev_pol, J_opt_ev_pol, toc_pol] = EV.solve_policy(x_current_ev_pol, x_sv_prediction_1, x_sv_prediction_2, p1(k), p2(k));

    Time_policy(k) = toc_pol;
    x_current_ev_pol = x_next_ev_pol;
    State_EV_policy(:, k + 1) = x_current_ev_pol;
    U_EV_policy(:, k) = u_current_ev_pol;
    J_EV_policy(k) = J_opt_ev_pol;

end
t = (0:1:k_max)*T;
fprintf('Average solution time for sequence optimization is %.2f sec.\n', mean(Time_sequence));
fprintf('Average solution time for policy optimization is %.2f sec.\n', mean(Time_policy));

Result.p1 = p1;
Result.p2 = p2;
Result.State_SV = State_SV;
Result.State_EV_sequence = State_EV_sequence;
Result.State_EV_policy = State_EV_policy;
Result.U_EV_sequence = U_EV_sequence;
Result.U_EV_policy = U_EV_policy;
Result.J_EV_sequence = J_EV_sequence;
Result.J_EV_policy = J_EV_policy;
Result.Com_Time_sequence = mean(Time_sequence);
Result.Com_Time_policy = mean(Time_policy);

filename = sprintf('%d_Result',h);
save(filename, 'Result');

h = h + 1;

end

    



