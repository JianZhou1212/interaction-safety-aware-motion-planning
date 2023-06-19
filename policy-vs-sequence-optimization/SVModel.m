classdef SVModel < handle

    properties (SetAccess = public)
      A;
      B;
      K
      ref_1;
      ref_2;
      N;
    end
    
    methods (Access = public)
        function obj = SVModel(parameter)
            obj.A = parameter.A;
            obj.B = parameter.B;
            obj.K = parameter.K;
            obj.ref_1 = parameter.ref_1;
            obj.ref_2 = parameter.ref_2;
            obj.N = parameter.N;

        end

        function [x_next, x_mode_1, x_mode_2] = solve(obj, x_current)
            x_mode_1 = ones(6, obj.N + 1);
            x_mode_2 = ones(6, obj.N + 1);

            x_mode_1(:, 1) = x_current;
            x_mode_2(:, 1) = x_current;

            for i = 1:1:obj.N
                x_i = x_mode_1(:, i);
                x_i_next = (obj.A - obj.B*obj.K)*x_i + obj.B*obj.K*obj.ref_1;
                x_mode_1(:, i + 1) = x_i_next;
            end


            for i = 1:1:obj.N
                x_i = x_mode_2(:, i);
                x_i_next = (obj.A - obj.B*obj.K)*x_i + obj.B*obj.K*obj.ref_2;
                x_mode_2(:, i + 1) = x_i_next;
            end

            x_next = x_mode_2(:, 2);
            
        end

    end
end