# Code for Interaction-Aware Motion Planning for Autonomous Vehicles with Multi-Modal Obstacle Uncertainties in Multi-Vehicle Scenarios

This version contains the implementation for the comparison between **Sequence-based Optimization** and **Policy-based Optimization** [1], [2].

Full code will be published if the paper can be accepted.

## Comparison between Sequence-based and Policy-based Optimizations
The scenario is designed as below, the SV is predicted to have both lane-changing probability and lane-keeping probability by the EV. In actuality the SV is always keeping the lane, the EV is controlled by an MPC controller which tracks a constant velocity and keeps the lane. If the SV is predicted to change the lane, the EV, being controlled by the MPC, needs to decelerate to avoid a collision with the SV over the prediction horizon.
![alt](policy-vs-sequence-optimization/Fig_SV_EV_Path.png "Fig.1 Scenario.")

We compare the results of the two methods with respect to different lane-changing probabilities of the SV. The probabilities are set manually while in a comprehensive motion planner, it is predicted by a motion prediction module. We set the predicted lane-changing of the SV in five different cases as below:
![alt](policy-vs-sequence-optimization/Fig_EV_Prob.png "Fig.2 Predicted lane-changing probability of the SV.")



## Reference
[1] Batkovic, Ivo, et al. "A robust scenario MPC approach for uncertain multi-modal obstacles." IEEE Control Systems Letters 5.3 (2021): 947-952.

[2] Nair, Siddharth H., et al. "Stochastic mpc with multi-modal predictions for traffic intersections." 2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2022.

