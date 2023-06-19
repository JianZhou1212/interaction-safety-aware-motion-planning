# Code for Interaction-Aware Motion Planning for Autonomous Vehicles with Multi-Modal Obstacle Uncertainties in Multi-Vehicle Scenarios

(For any questions please contact jian.zhou@liu.se)

The current version contains the implementation for the comparison between **Sequence-based Optimization** and **Policy-based Optimization** [1], [2].

The full code of the paper will be published if the paper can be accepted.

## Comparison between Sequence-based and Policy-based Optimizations
The implementation is performed by MATLAB, to run the code you need to download CasADi (https://web.casadi.org/).


The scenario is designed in Fig. 1, where the SV is predicted to have both lane-changing probability and lane-keeping probability by the EV. In actuality the SV is always keeping the lane, the EV is controlled by an MPC controller which tracks a constant velocity and keeps the lane. If the SV is predicted to change the lane, the EV, being controlled by the MPC, needs to decelerate to avoid a collision with the SV over the prediction horizon.
![alt](policy-vs-sequence-optimization/Fig_SV_EV_Path.png)
<center><font size=2> Fig. 1 Scenario </font></center>

We compare the results of the two methods with respect to different lane-changing probabilities of the SV. The probabilities are set manually here, while in a comprehensive motion planner, it is predicted by a motion prediction module. We set the predicted lane-changing probabilities of the SV in five different cases as below:
![alt](policy-vs-sequence-optimization/Fig_EV_Prob.png)
<center><font size=2> Fig.2 Predicted lane-changing probability of the SV. </font></center>

According to the probability output, we get the motion-planning results of the EV:
![alt](policy-vs-sequence-optimization/Fig_EV_Speed.png)
<center><font size=2> Fig.3 Speed of the EV. </font></center>

![alt](policy-vs-sequence-optimization/Fig_EV_Acc.png)
<center><font size=2> Fig.4 Net acceleration of the EV. </font></center>

![alt](policy-vs-sequence-optimization/Fig_EV_Cost.png)
<center><font size=2> Fig.5 Optimal cost of two methods. </font></center>

The computation time of the sequence-based approach with p = 0.1 to 0.5 is 0.06 s, 0.07 s, 0.05 s, 0.07 s, 0.05 s, respectively.

The computation time of the policy-based approach with p = 0.1 to 0.5 is 0.11 s, 0.12 s, 0.09 s, 0.12 s, 0.09 s, respectively.

From this case study, we can conclude that:
(1) The policy-based approach is generally less conservative than the sequence-based approach, particularly when one of the probabilities is predicted very small. (2) The policy-based approach takes a longer time to solve the problem, and it is easy to infer that the computation time increases when the number of SVs and the number of modes are increased.

## Reference
[1] Batkovic, Ivo, et al. "A robust scenario MPC approach for uncertain multi-modal obstacles." IEEE Control Systems Letters 5.3 (2021): 947-952.

[2] Nair, Siddharth H., et al. "Stochastic mpc with multi-modal predictions for traffic intersections." 2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2022.

