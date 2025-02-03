A New framework for Bayesian registration

In this project, we adopt the linear representation of the time warping function to propose a new prior within the Bayesian registration. 

This repository contains 12 main scripts and 3 function scripts: 
1. main_br_pw_iso.m
Bayesian registration with Gaussian process prior with isotropic-like covariance for pairwise case.

2. main_br_pw_diag.m
Bayesian registration with Gaussian process prior with diagonal-like covariance for pairwise case.

3. main_Mul_pw_iso.m
Bayesian registration with Gaussian process prior with isotropic-like covariance for multiple case.

4. main_Mul_pw_diag.m
Bayesian registration with Gaussian process prior with diagonal-like covariance for multiple case.

5. main_clr_vs_srvf.m 
Compare the resampling results obtained using CLR-based and SRVF-based methods

6. main_signal_est.m
Get signal estimation using L2 mean

7. main_two_vs_one.m
Showcase Bayesian registration can get more comprehensive exploration of potention warpings, and apply K-means clustering to cluster the warping result.

8. main_three_vs_one_control_mean.m
showcase we can target the peak to align with by adjusting the prior in Bayesian registration.

9. main_three_vs_one_NON_Gaussian.m
showcase the effectiveness of the non-Gaussian prior in Bayesian registration.


Files below are the other necessary functions used in the main scripts. 

1. cal_joint_ratio_clr_corr.m
compute the \rho which is used to decide whether we accept the new proposal within MH or not for pairwise case. 

1. cal_joint_ratio_clr_mulf_corr.m
compute the \rho which is used to decide whether we accept the new proposal within MH or not for multiple case. 

