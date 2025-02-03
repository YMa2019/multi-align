function [prob, SSE_diff] = cal_joint_ratio_clr_mulf_corr(sigma1, q, phi, t, qm_new, qm_old)
    [M, N] = size(q);
    SSE_new = 0;  
    SSE_old = 0; 
    for m = 1: M
        phi_temp = phi(m,:);
        q_m = q(m,:);
        temp_in1 = cumtrapz(t, exp(phi_temp))./trapz(t, exp(phi_temp));
%         temp_in1 = round(temp_in1/temp_in1(end)*(N-1))+1;
        temp_in1 = (temp_in1 - min(temp_in1))/(max(temp_in1) - min(temp_in1));
        gam_dev = exp(phi_temp)./trapz(t, exp(phi_temp));

%         SSE_new = SSE_new + (norm(qm_new - q_m(temp_in1).*sqrt(gam_dev')))^2;  
%         SSE_old = SSE_old + (norm(qm_old - q_m(temp_in1).*sqrt(gam_dev')))^2; 
        SSE_new = SSE_new + (norm(qm_new - interp1(t, q_m, temp_in1).*sqrt(gam_dev')))^2;  
        SSE_old = SSE_old + (norm(qm_old - interp1(t, q_m, temp_in1).*sqrt(gam_dev')))^2; 
    end

    SSE_diff = SSE_new - SSE_old;
    prob = exp(-1/(2*sigma1^2)*SSE_diff);
end