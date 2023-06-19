%%
clc
clear
close all
Result_1 = load('1_Result.mat');
Result_2 = load('2_Result.mat');
Result_3 = load('3_Result.mat');
Result_4 = load('4_Result.mat');
Result_5 = load('5_Result.mat');
k_max = 35;
t = (0:1:35)*0.25;
%
figure(1)
for h = 1:1:(k_max + 1)
   
    heading_sv = atan(Result_1.Result.State_SV(5, h)/Result_1.Result.State_SV(2, h));
    [x_sv, y_sv] = VehicleShape(Result_1.Result.State_SV(1, h), Result_1.Result.State_SV(4, h), heading_sv, 4, 2);
    hold on
    h_sv = plot(x_sv, y_sv, 'k');

    heading_ev = atan(Result_1.Result.State_EV_sequence(4, h)/Result_1.Result.State_EV_sequence(2, h));
    [x_ev, y_ev] = VehicleShape(Result_1.Result.State_EV_sequence(1, h), Result_1.Result.State_EV_sequence(3, h), heading_ev, 4, 2);
    hold on
    h_ev = plot(x_ev, y_ev, 'b');
end
plot(Result_1.Result.State_SV(1, :), 0*ones(1, length(Result_1.Result.State_SV(1, :))), 'k', 'linewidth', 3);
hold on
plot(Result_1.Result.State_SV(1, :), 4*ones(1, length(Result_1.Result.State_SV(1, :))), 'k--', 'linewidth', 3);
hold on
plot(Result_1.Result.State_SV(1, :), 8*ones(1, length(Result_1.Result.State_SV(1, :))), 'k', 'linewidth', 3);
legend([h_sv, h_ev], {'SV', 'EV'}, 'Interpreter','latex');
%xlim([0.0 - 5, max(State_Car(1, :)) + 5]);
ylim([0, 8]);
xlabel('X [m]', 'Interpreter', 'latex');
ylabel('Y [m]', 'Interpreter', 'latex');
set(gca,'FontName','Times New Roman','FontSize',15);
set(gca,'LooseInset',get(gca,'TightInset'));
set(gcf,'unit','centimeters','position',[3 5 20 4.5]);
set(gcf, 'PaperSize', [12 4]);
set(gca, 'ygrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
set(gca, 'xgrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
grid off
box on
exportgraphics(gcf,'Fig_SV_EV_Path.pdf','ContentType','vector');

figure(2)
plot(t, Result_1.Result.State_EV_sequence(2, :), 'k', 'linewidth', 2.5);
hold on
plot(t, Result_1.Result.State_EV_policy(2, :), 'm', 'linewidth', 2.5);
hold on
plot(t, Result_2.Result.State_EV_policy(2, :), 'g', 'linewidth', 2.5);
hold on
plot(t, Result_3.Result.State_EV_policy(2, :), 'c', 'linewidth', 2.5);
hold on
plot(t, Result_4.Result.State_EV_policy(2, :), 'r', 'linewidth', 2.5);
hold on
plot(t, Result_5.Result.State_EV_policy(2, :), 'b', 'linewidth', 2.5);
legend('Sequence', 'Policy (p = 0.1)', 'Policy (p = 0.2)', 'Policy (p = 0.3)', 'Policy (p = 0.4)', 'Policy (p = 0.5)', 'Interpreter', 'latex', 'NumColumns', 1);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('Long. Speed [m/s]', 'Interpreter', 'latex');
set(gca,'FontName','Times New Roman','FontSize',15);
set(gca,'LooseInset',get(gca,'TightInset'));
set(gcf,'unit','centimeters','position',[3 5 20 6.5]);
set(gcf, 'PaperSize', [12 4]);
set(gca, 'ygrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
set(gca, 'xgrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
grid off
box on
exportgraphics(gcf,'Fig_EV_Speed.pdf','ContentType','vector');

figure(3)
plot(t(1:end-1), sqrt(Result_1.Result.U_EV_sequence(1, :).^2 + Result_1.Result.U_EV_sequence(2, :).^2), 'k', 'linewidth', 2.5);
hold on
plot(t(1:end-1), sqrt(Result_1.Result.U_EV_policy(1, :).^2 + Result_1.Result.U_EV_policy(2, :).^2), 'm', 'linewidth', 2.5);
hold on
plot(t(1:end-1), sqrt(Result_2.Result.U_EV_policy(1, :).^2 + Result_2.Result.U_EV_policy(2, :).^2), 'g', 'linewidth', 2.5);
hold on
plot(t(1:end-1), sqrt(Result_3.Result.U_EV_policy(1, :).^2 + Result_3.Result.U_EV_policy(2, :).^2), 'c', 'linewidth', 2.5);
hold on
plot(t(1:end-1), sqrt(Result_4.Result.U_EV_policy(1, :).^2 + Result_4.Result.U_EV_policy(2, :).^2), 'r', 'linewidth', 2.5);
hold on
plot(t(1:end-1), sqrt(Result_5.Result.U_EV_policy(1, :).^2 + Result_5.Result.U_EV_policy(2, :).^2), 'b', 'linewidth', 2.5);
legend('Sequence', 'Policy (p = 0.1)', 'Policy (p = 0.2)', 'Policy (p = 0.3)', 'Policy (p = 0.4)', 'Policy (p = 0.5)', 'Interpreter', 'latex', 'NumColumns', 1);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('Net Acc. [${\rm m/s^2}$]', 'Interpreter', 'latex');
set(gca,'FontName','Times New Roman','FontSize',15);
set(gca,'LooseInset',get(gca,'TightInset'));
set(gcf,'unit','centimeters','position',[3 5 20 6.5]);
set(gcf, 'PaperSize', [12 4]);
set(gca, 'ygrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
set(gca, 'xgrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
grid off
box on
exportgraphics(gcf,'Fig_EV_Acc.pdf','ContentType','vector');

figure(4)
plot(t(1:end-1), Result_1.Result.J_EV_sequence, 'k', 'linewidth', 2.5);
hold on
plot(t(1:end-1), Result_1.Result.J_EV_policy, 'm', 'linewidth', 2.5);
hold on
plot(t(1:end-1), Result_2.Result.J_EV_policy, 'g', 'linewidth', 2.5);
hold on
plot(t(1:end-1), Result_3.Result.J_EV_policy, 'c', 'linewidth', 2.5);
hold on
plot(t(1:end-1), Result_4.Result.J_EV_policy, 'r', 'linewidth', 2.5);
hold on
plot(t(1:end-1), Result_5.Result.J_EV_policy, 'b', 'linewidth', 2.5);
legend('Sequence', 'Policy (p = 0.1)', 'Policy (p = 0.2)', 'Policy (p = 0.3)', 'Policy (p = 0.4)', 'Policy (p = 0.5)', 'Interpreter', 'latex', 'NumColumns', 1);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('Optimal Cost [--]', 'Interpreter', 'latex');
set(gca,'FontName','Times New Roman','FontSize',15);
set(gca,'LooseInset',get(gca,'TightInset'));
set(gcf,'unit','centimeters','position',[3 5 20 6.5]);
set(gcf, 'PaperSize', [12 4]);
set(gca, 'ygrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
set(gca, 'xgrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
grid off
box on
exportgraphics(gcf,'Fig_EV_Cost.pdf','ContentType','vector');

figure(5)
plot(t(1:end-1), Result_1.Result.p2, 'm', 'linewidth', 2.5);
hold on
plot(t(1:end-1), Result_2.Result.p2, 'g', 'linewidth', 2.5);
hold on
plot(t(1:end-1), Result_3.Result.p2, 'c', 'linewidth', 2.5);
hold on
plot(t(1:end-1), Result_4.Result.p2, 'r', 'linewidth', 2.5);
hold on
plot(t(1:end-1), Result_5.Result.p2, 'b', 'linewidth', 2.5);
legend('Prob. LC = 0.1', 'Prob. LC = 0.2', 'Prob. LC = 0.3', 'Prob. LC = 0.4', 'Prob. LC = 0.5', 'Interpreter', 'latex', 'NumColumns', 1);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('Probability [-]', 'Interpreter', 'latex');
set(gca,'FontName','Times New Roman','FontSize',15);
set(gca,'LooseInset',get(gca,'TightInset'));
set(gcf,'unit','centimeters','position',[3 5 20 6.5]);
set(gcf, 'PaperSize', [12 4]);
set(gca, 'ygrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
set(gca, 'xgrid', 'on', 'GridColor', [0.75 0.75 0.75], 'LineWidth', 1);
grid off
box on
exportgraphics(gcf,'Fig_EV_Prob.pdf','ContentType','vector');
