%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
N = 101;

seed = rand*1000;
rng(994);%756，644

t = linspace(0,1,N);
f1 = @(x) 6*(0.8).^(20*x).*cos(10*pi.*x-pi/4);
f2 = @(x) 6*(0.8).^(20*x).*sin(10*pi.*x);
time_gap = 1/(N-1);
a = [-0.5, 2];
f1 = f1(t)';


% % set fixed gamma2
gamma2 = (exp(a(2)*t)-1)/(exp(a(2))-1);
gamma_true = interp1(gamma2, t, t);
f2 = 1*interp1(t,f2(t),gamma2);
f2 = f2';
figure(1); clf;
plot(t, f1, t, f2);
q1 = sign(gradient(f1)/time_gap).*sqrt(abs(gradient(f1)/time_gap));
q2 = sign(gradient(f2)/time_gap).*sqrt(abs(gradient(f2)/time_gap));

%define the covariance for the gamma function
mu = zeros(1, N);
% f_cov = ones(1, N);
f_cov= [5*ones(1, (N-1)/2+1), 0.1*ones(1, (N-1)/2-0)];
Cr1 = diag(f_cov);
sigma_kernel = 8;
kernel_size = 51; % Adjust the size as needed
[X, Y] = meshgrid(-(kernel_size-1)/2:(kernel_size-1)/2, -(kernel_size-1)/2:(kernel_size-1)/2);
gaussian_kernel = exp(-(X.^2 + Y.^2) / (2 * sigma_kernel^2));
Cr2 = 1*conv2(Cr1, gaussian_kernel, 'same'); %change to 0.01, we can get second-optimal
Cr = Cr2;
[V, D, U] = svd(Cr);
Cr = V * D * V';


figure(1001); clf;
imagesc(Cr);
colorbar;
axis equal;
set(gca,'YDir','normal')
xlim([0 100])
ylim([0 100])
xticks([0 20 40 60 80 100]);
xticklabels([0 0.2 0.4 0.6 0.8 1]);
yticks([0 20 40 60 80 100]);
yticklabels([0 0.2 0.4 0.6 0.8 1]);
set(gca, 'Fontsize', 18,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors     = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width      = 12;
opts.height     = 10;
opts.fontType   = 'Times';
% try chol(Cr)
%     disp('Matrix is symmetric positive definite.')
% catch ME
%     disp('Matrix is not symmetric positive definite')
% end

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
phi_int = zeros(1, N);

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
betals = [0.5, linspace(0.001, 0.1,9)];
probabilities = repmat(0.1, 1, 10);

cnt = 1;
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
        cnt = cnt + 1;
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

for j = 1:J+1
    gamma_new(j,:) = cumtrapz(t, exp(phi_set(j,:)))./trapz(t, exp(phi_set(j,:)));
    gamma_new(j,:) = (gamma_new(j,:)-min(gamma_new(j,:)))/(max(gamma_new(j,:))-min(gamma_new(j,:)));
end


fgamma = gamma_new(J/2+1:10:J+1,:);
fphi_set = phi_set(J/2+1:10:J+1,:);

eps = 1e-5;
learn_rate = 0.1;
sample_mean = mean(fphi_set,1);
gamma_mean = cumtrapz(t, exp(sample_mean))./trapz(t, exp(sample_mean));
gamma_mean = (gamma_mean-min(gamma_mean))/(max(gamma_mean)-min(gamma_mean));





figure(1); clf;
% plot(t,gamma_new);
plot(t,fgamma,'Color', [0.7 0.7 0.7],'LineWidth', 1);
hold on;
% plot(t,gamma_true ,'k','LineWidth',2);
plot(t,gamma_t,'k','LineWidth',2);
plot(t,gamma_mean','r','LineWidth',2);
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
figure(4); clf;
plot(t,fphi_set,'Color', [0.7 0.7 0.7],'LineWidth', 1);
hold on;
plot(t,mean(fphi_set),'r','LineWidth', 2);
% pbaspect([1 1 1]);
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


%plot aligned function
f2_gamma_mean = interp1(t,f2,gamma_mean);
for i = 1: length(fgamma)
    f2_gamma_t(i,:) = interp1(t,f2,fgamma(i,:));
end 
f2_gamma_dp = interp1(t,f2,gamma_t);
max_f2_gamma = max(f2_gamma_t);
min_f2_gamma = min(f2_gamma_t);
mean_f2_gamma = mean(f2_gamma_t);
figure(2);clf;
plot(t,f1,'b.-','LineWidth', 1.5);
hold on
patch([t, fliplr(t)], [min_f2_gamma, fliplr(max_f2_gamma)],[0.7 0.7 0.7], 'EdgeColor',[0.7 0.7 0.7]);
plot(t,f2_gamma_t,'color',[0.7 0.7 0.7],'LineWidth', 1.5);
plot(t,f1,'b.-','LineWidth', 1.5);
plot(t,f2,'g.-','LineWidth', 1.5);
plot(t, mean_f2_gamma, 'r','LineWidth', 1.5)
legend({'','$f_1$','$f_2$'},'Interpreter','latex','Box','off', 'Fontsize', 30)
% plot(t, f2_gamma_dp, 'k','LineWidth', 1.5)
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

