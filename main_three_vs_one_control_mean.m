%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
N = 201;

seed = rand*1000;
rng("default");

t1 = linspace(-3,3,N);

a = 0.5;
time_gap = 1/(N-1);

for j =1:N 
    f(j) = exp(-(t1(j)-1.8).^2/0.2) + exp(-(t1(j)-0).^2/0.2)+ exp(-(t1(j)+1.8).^2/0.2);
end

f = f - interp1([-3 3], [f(1) f(end)],t1);

a = [-0.5, 2];
t = linspace(0,1,N);

gamma1 = t;
f1 = interp1(t,f,gamma1);

for j =1:N 
    g(j) = exp(-(t1(j)-0.2).^2/0.2);
end
f2 = interp1(t,g,t);

figure(1);clf;
plot(t, f1, t, f2)

f1 = f1';
f2 = f2';

q1 = sign(gradient(f1)/time_gap).*sqrt(abs(gradient(f1)/time_gap));
q2 = sign(gradient(f2)/time_gap).*sqrt(abs(gradient(f2)/time_gap));

%define the covariance for the gamma function
mu = zeros(1, N);
mu = 2*(t-0.5); %-2 0 2.5

% Define the value of pho
pho = 0.999;  % You can change this to your desired value

f_cov = ones(1, N);
Cr = diag(f_cov);

% Fill in the matrix
for i = 1:N
    for j = 1:N
        if i ~= j
            Cr(i, j) = pho^(abs(i-j));
        end
    end
end
Cr = Cr*2;

figure(1001); clf;
imagesc(Cr);
try chol(Cr)
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end

inv_cr = inv(Cr);
figure(1002); clf;
imagesc(inv_cr);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Dynamic programming to match f1 and f2 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma_t = DynamicProgrammingQ(q2', q1', 0, 0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% bayesian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. initiate phi 

phi_int = mvnrnd(mu, Cr, 1);
phi_int = phi_int-trapz(t,phi_int);

figure(101); clf;
plot(t, phi_int);
figure(102); clf;
gamma_in = cumtrapz(t, exp(phi_int))./trapz(t, exp(phi_int));
gamma_in = (gamma_in-min(gamma_in))/(max(gamma_in)-min(gamma_in)); 
plot(t, gamma_in);

%2. Initiate sigma1
sigma1_int = 5;

%3: update g and sigma1
J = 10000;
phi_set = [];
phi_set(1,:) = phi_int;

%set the parameters for the pCN-mixture
betals = [0.5, linspace(0.001, 0.01, 4)];
probabilities = [0.6, repmat(0.1, 1, 4)];


for j = 1: J
    %propose new phi
    kesi = mvnrnd(mu, Cr, 1);
    kesi = kesi-trapz(t,kesi);
    beta = randsample(betals, 1,true, probabilities);
    phi_new = phi_set(j,:)*sqrt(1-beta^2) + beta*kesi;
    phi_new = phi_new-trapz(t,phi_new);
    
    % calculate MCMC acceptance ratio
    [lossratio, sse_diff(j)] = cal_joint_ratio_clr_corr(sigma1_int, q1, q2, t, phi_new, phi_set(j,:));
    lossratio_(j) = lossratio;
    lamd_p = min(1, lossratio);

    if rand()<lamd_p
        phi_set(j+1,:) = phi_new;
    else
        phi_set(j+1,:) = phi_set(j,:);
    end
    
    temp_t = cumtrapz(t,exp(phi_set(j+1,:)))./trapz(t, exp(phi_set(j+1,:)),2);
    temp_t = round(temp_t/temp_t(end)*(N-1))+1;
    gam1_dev = exp(phi_set(j+1,:))./trapz(t, exp(phi_set(j+1,:)),2);
    SSE = (norm(q2 - q1(temp_t).*sqrt(gam1_dev')))^2;
    sse_(j) = SSE;

end

% plot the sample gamma
gamma_new = [];
cnt = 1;
for j = 1:J+1
%     phi_set(j,:) = phi_set(j,:)-log(trapz(t,exp(phi_set(j,:))));
    gamma_new(j,:) = cumtrapz(t, exp(phi_set(j,:)))./trapz(t, exp(phi_set(j,:)));
    gamma_new(j,:) = (gamma_new(j,:)-min(gamma_new(j,:)))/(max(gamma_new(j,:))-min(gamma_new(j,:)));
end


fgamma = gamma_new(J/2+1:10:J+1,:);
fphi_set = phi_set(J/2+1:10:J+1,:);

eps = 1e-5;
learn_rate = 0.1;
sample_mean = mean(fphi_set,1);
gamma_mean = cumsum(exp(sample_mean))./sum(exp(sample_mean));
gamma_mean = (gamma_mean-min(gamma_mean))/(max(gamma_mean)-min(gamma_mean));





figure(1); clf;
hold on;
% plot(t,gamma_new);
% plot(t,fgamma,'Color', [0.8 0.8 0.8]);
plot(t,fgamma,'g');
% plot(t,gamma_t,'k','LineWidth',2);
plot(t,gamma_mean,'r','LineWidth',2);
% plot(t,gamma_mean','b--','LineWidth',2);
lsize = 16; % Label fontsize
nsize = 18; % Axis fontsize
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors     = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width      = 12;
opts.height     = 10;
opts.fontType   = 'Times';
% 

f2_gamma_t = interp1(t,f2,gamma_t);
f2_gamma_mt = interp1(t,f2,gamma_mean);
figure(2);clf;
% plot(t,f1,'b.','LineWidth', 1.5);
hold on;
% plot(t,f2,'g.','LineWidth', 1.5);
% plot(t, f2_gamma_t, 'k','LineWidth', 1.5)
plot(t, f2_gamma_mt, 'c','LineWidth', 1.5)
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors     = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width      = 12;
opts.height     = 10;
opts.fontType   = 'Times';

figure(3);clf;
plot(t,q1,'b.','LineWidth', 1.5);
hold on;
plot(t,q2,'g.','LineWidth', 1.5);

figure(5);clf;
subplot(2,1,1);
plot(sse_)
subplot(2,1,2);
hist(sse_);


