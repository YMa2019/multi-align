function [prob, SSE_diff] = cal_joint_ratio_clr_corr(sigma1, q1, q2, t, phi_new, phi_old)
    N = length(t);
    temp_in1 = cumtrapz(t, exp(phi_new))./trapz(t, exp(phi_new));
    gamma_in1 = round(temp_in1/temp_in1(end)*(N-1))+1;
    gam1_dev = exp(phi_new)./trapz(t, exp(phi_new));
    SSE_new = (norm(q1 - q2(gamma_in1).*sqrt(gam1_dev')))^2;

    temp_in2 = cumtrapz(t, exp(phi_old))./trapz(t,exp(phi_old));
    gamma_in2 = round(temp_in2/temp_in2(end)*(N-1))+1;
    gam2_dev = exp(phi_old)./trapz(t, exp(phi_old));
    SSE_old = (norm(q1 - q2(gamma_in2).*sqrt(gam2_dev')))^2;
    
    SSE_diff = SSE_new - SSE_old;

    prob = exp(-1/(2*sigma1^2)*SSE_diff);
end