classdef EVControl < handle

    properties (SetAccess = public)
        A;
        B;
        N;
        vx_ref;
        y_ref;
        ax_bound;
        ay_bound;
        l_veh;
        w_veh;
        T;
        infx;
        N_bar;

        optimizer_sequence;
        optimizer_policy;
    end
    
    methods (Access = public)
        function obj = EVControl(parameter)

            obj.A = parameter.A_ev;
            obj.B = parameter.B_ev;
            obj.N = parameter.N;
            obj.vx_ref = parameter.vx_ref_e;
            obj.y_ref = parameter.y_ref_e;
            obj.ax_bound = parameter.ax_bound;
            obj.ay_bound = parameter.ay_bound;
            obj.l_veh = parameter.l_veh;
            obj.w_veh = parameter.w_veh;
            obj.T = parameter.T;
            obj.infx= parameter.infx;
            obj.N_bar = parameter.N_bar;

            obj.optimizer_sequence = obj.optimizesequence( );
            obj.optimizer_policy = obj.optimizepolicy( );

        end

        function [x_next, u_current, u_opt, x_opt, J_opt, t1] = solve_sequence(obj, x_ini, x_sv_prediction_1, x_sv_prediction_2, p1, p2)

            if p1 == 0
                x_DV = x_sv_prediction_2(1, 2:end);
                y_DV = x_sv_prediction_2(4, 2:end);
                x_aug = obj.l_veh*ones(1, obj.N);
                y_aug = obj.w_veh*ones(1, obj.N);
            else
                x_DV = (x_sv_prediction_1(1, 2:end) + x_sv_prediction_2(1, 2:end))/2;
                y_DV = (x_sv_prediction_1(4, 2:end) + x_sv_prediction_2(4, 2:end))/2;
                x_aug = abs(x_sv_prediction_1(1, 2:end) - x_sv_prediction_2(1, 2:end))/2 + obj.l_veh;
                y_aug = abs(x_sv_prediction_1(4, 2:end) - x_sv_prediction_2(4, 2:end))/2 + obj.w_veh;
            end


            t0 = cputime;
            [u_opt, x_opt, J_opt] = obj.optimizer_sequence(x_ini, x_DV, y_DV, x_aug, y_aug);
            t1 = cputime-t0;

            u_opt  = u_opt.full();
            x_opt  = x_opt.full();
            J_opt = J_opt.full();

            x_next    = x_opt(:, 2);
            u_current = u_opt(:, 1);

        end

        function [x_next, u_current, u_mode_1_opt, x_mode_1_opt, u_mode_2_opt, x_mode_2_opt, J_opt, t1] = solve_policy(obj, x_ini, x_sv_prediction_1, x_sv_prediction_2, p1, p2)

            if p1 == 0
                x_DV_1 = x_sv_prediction_1(1, 2:end) + obj.infx;
                y_DV_1 = x_sv_prediction_1(4, 2:end) + obj.infx;
                x_DV_2 = x_sv_prediction_2(1, 2:end);
                y_DV_2 = x_sv_prediction_2(4, 2:end);
                x_aug_1 = obj.l_veh*ones(1, obj.N);
                y_aug_1 = obj.w_veh*ones(1, obj.N);
                x_aug_2 = obj.l_veh*ones(1, obj.N);
                y_aug_2 = obj.w_veh*ones(1, obj.N);
            else 
                x_DV_1 = x_sv_prediction_1(1, 2:end);
                y_DV_1 = x_sv_prediction_1(4, 2:end);
                x_DV_2 = x_sv_prediction_2(1, 2:end);
                y_DV_2 = x_sv_prediction_2(4, 2:end);
                x_aug_1 = obj.l_veh*ones(1, obj.N);
                y_aug_1 = obj.w_veh*ones(1, obj.N);
                x_aug_2 = obj.l_veh*ones(1, obj.N);
                y_aug_2 = obj.w_veh*ones(1, obj.N);
            end


            t0 = cputime;
            [u_mode_1_opt, x_mode_1_opt, u_mode_2_opt, x_mode_2_opt, J_opt] = obj.optimizer_policy(x_ini, x_DV_1, y_DV_1, x_aug_1, y_aug_1, x_DV_2, y_DV_2, x_aug_2, y_aug_2, p1, p2);
            t1 = cputime-t0;

            u_mode_1_opt  = u_mode_1_opt.full();
            x_mode_1_opt  = x_mode_1_opt.full();
            u_mode_2_opt  = u_mode_2_opt.full();
            x_mode_2_opt  = x_mode_2_opt.full();
            J_opt = J_opt.full();

            x_next    = x_mode_1_opt(:, 2);
            u_current = u_mode_1_opt(:, 1);

        end

        function x_dot = EVModel(obj, x, u)
            x_dot = obj.A*x + obj.B*u;
        end

        function optimizer = optimizesequence(obj)

            opti = casadi.Opti(); 
            x = opti.variable(4, obj.N + 1); 
            u = opti.variable(2, obj.N); 
 
            x_ini = opti.parameter(4, 1);
            x_DV = opti.parameter(1, obj.N);
            y_DV = opti.parameter(1, obj.N);
            x_aug = opti.parameter(1, obj.N);
            y_aug = opti.parameter(1, obj.N);
 
            opti.subject_to(x(:, 1) == x_ini);
            for k=1:obj.N 
               k1 = obj.EVModel(x(:,k),         u(:,k));
               k2 = obj.EVModel(x(:,k)+obj.T/2*k1, u(:,k));
               k3 = obj.EVModel(x(:,k)+obj.T/2*k2, u(:,k));
               k4 = obj.EVModel(x(:,k)+obj.T*k3,   u(:,k));
               x_next = x(:,k) + obj.T/6*(k1+2*k2+2*k3+k4); 
               opti.subject_to(x(:,k+1)==x_next); 
            end

            opti.subject_to(-((x(1, 2:end) - x_DV)./x_aug).^2 - ((x(3, 2:end) - y_DV)./y_aug).^2 <= -1);
            opti.subject_to(1 <= x(3, 2:end) <= 3);

            opti.subject_to(-(obj.ax_bound) <= u(1, :) <= (obj.ax_bound)); 
            opti.subject_to(-(obj.ay_bound) <= u(2, :) <= (obj.ay_bound)); 

            J = (x(2, 2:end) - obj.vx_ref)*(x(2, 2:end) - obj.vx_ref)' + (x(3, 2:end) - obj.y_ref)*(x(3, 2:end) - obj.y_ref)' + u(1, :)*u(1, :)' + u(2, :)*u(2, :)';

            opti.minimize(J);

            options = struct;
            %options.ipopt.linear_solver = 'ma57';
            options.ipopt.max_iter = 2000;
            options.ipopt.print_level = 0;
            opti.solver('ipopt', options); 
            optimizer = opti.to_function('f', {x_ini, x_DV, y_DV, x_aug, y_aug}, {u, x, J});
        end

        function optimizer = optimizepolicy(obj)

            opti = casadi.Opti(); 
 
            u_mode_1 = opti.variable(2, obj.N);
            u_mode_2 = opti.variable(2, obj.N);

            x_mode_1 = opti.variable(4, obj.N + 1);
            x_mode_2 = opti.variable(4, obj.N + 1);
 
            x_ini = opti.parameter(4, 1);
            x_DV_1 = opti.parameter(1, obj.N);
            y_DV_1 = opti.parameter(1, obj.N);
            x_aug_1 = opti.parameter(1, obj.N);
            y_aug_1 = opti.parameter(1, obj.N);
            x_DV_2 = opti.parameter(1, obj.N);
            y_DV_2 = opti.parameter(1, obj.N);
            x_aug_2 = opti.parameter(1, obj.N);
            y_aug_2 = opti.parameter(1, obj.N);
            p1 = opti.parameter( );
            p2 = opti.parameter( );

            for k = 1:obj.N_bar
                opti.subject_to(u_mode_1(:, k) == u_mode_2(:, k));
            end

            opti.subject_to(x_mode_1(:, 1) == x_ini);
            opti.subject_to(x_mode_2(:, 1) == x_ini);
            for k=1:obj.N
               k1 = obj.EVModel(x_mode_1(:,k),            u_mode_1(:,k));
               k2 = obj.EVModel(x_mode_1(:,k)+obj.T/2*k1, u_mode_1(:,k));
               k3 = obj.EVModel(x_mode_1(:,k)+obj.T/2*k2, u_mode_1(:,k));
               k4 = obj.EVModel(x_mode_1(:,k)+obj.T*k3,   u_mode_1(:,k));
               x_next = x_mode_1(:,k) + obj.T/6*(k1+2*k2+2*k3+k4); 
               opti.subject_to(x_mode_1(:,k+1)==x_next); 
            end

            for k=1:obj.N
               k1 = obj.EVModel(x_mode_2(:,k),            u_mode_2(:,k));
               k2 = obj.EVModel(x_mode_2(:,k)+obj.T/2*k1, u_mode_2(:,k));
               k3 = obj.EVModel(x_mode_2(:,k)+obj.T/2*k2, u_mode_2(:,k));
               k4 = obj.EVModel(x_mode_2(:,k)+obj.T*k3,   u_mode_2(:,k));
               x_next = x_mode_2(:,k) + obj.T/6*(k1+2*k2+2*k3+k4); 
               opti.subject_to(x_mode_2(:,k+1)==x_next); 
            end

            opti.subject_to(1 <= (((x_mode_1(1, 2:end) - x_DV_1)./x_aug_1).^2 + ((x_mode_1(3, 2:end) - y_DV_1)./y_aug_1).^2));
            opti.subject_to(1 <= (((x_mode_1(1, 2:end) - x_DV_2)./x_aug_2).^2 + ((x_mode_1(3, 2:end) - y_DV_2)./y_aug_2).^2));
            opti.subject_to(1 <= x_mode_1(3, 2:end) <= 3);
            opti.subject_to(1 <= x_mode_2(3, 2:end) <= 3);

            opti.subject_to(-(obj.ax_bound) <= u_mode_1(1, :) <= (obj.ax_bound)); 
            opti.subject_to(-(obj.ay_bound) <= u_mode_1(2, :) <= (obj.ay_bound)); 

            opti.subject_to(-(obj.ax_bound) <= u_mode_2(1, :) <= (obj.ax_bound)); 
            opti.subject_to(-(obj.ay_bound) <= u_mode_2(2, :) <= (obj.ay_bound)); 

            J1 = (x_mode_1(2, 2:end) - obj.vx_ref)*(x_mode_1(2, 2:end) - obj.vx_ref)' + (x_mode_1(3, 2:end) - obj.y_ref)*(x_mode_1(3, 2:end) - obj.y_ref)' + u_mode_1(1, :)*u_mode_1(1, :)' + u_mode_1(2, :)*u_mode_1(2, :)';
            J2 = (x_mode_2(2, 2:end) - obj.vx_ref)*(x_mode_2(2, 2:end) - obj.vx_ref)' + (x_mode_2(3, 2:end) - obj.y_ref)*(x_mode_2(3, 2:end) - obj.y_ref)' + u_mode_2(1, :)*u_mode_2(1, :)' + u_mode_2(2, :)*u_mode_2(2, :)';
            J = p1*J1 + p2*J2;

            opti.minimize(J);

            options = struct;
            %options.ipopt.linear_solver = 'ma57';
            options.ipopt.max_iter = 2000;
            options.ipopt.print_level = 0;
            opti.solver('ipopt', options); 
            optimizer = opti.to_function('f', {x_ini, x_DV_1, y_DV_1, x_aug_1, y_aug_1, x_DV_2, y_DV_2, x_aug_2, y_aug_2,p1, p2}, {u_mode_1, x_mode_1, u_mode_2, x_mode_2, J});
        end

    end
end