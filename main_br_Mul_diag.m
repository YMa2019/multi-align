%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all;

see = rand*1000;
rng(0);
nsize = 18;
% load('simu_data.mat');
% load('growth_male_vel.mat');
% load('growth_female_vel.mat');
load('growth.mat');
figure(5);clf;
plot(mydata.age,mydata.hgtf,'LineWidth', 1.5);
pbaspect([1 1 1]);
xlim([1,18]);
% ylim([0,25]);
xticks([1 3 6 9 12 15 18]);
xticklabels([1 3 6 9 12 15 18]);
ylabel("Height (cm)")
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

f_org = f;
[N_org M] = size(f);
N = 201;
t_org = linspace(0, 1, N_org);
t = linspace(0, 1, N);

f = zeros(N, M);
windowSize = 5; % Adjust the window size as needed
for m = 1:M
    f(:,m) = interp1(t_org, f_org(:,m), t);
    f(:,m) = smooth(f(:,m), windowSize);
end

g = f';

figure(1000001);clf
plot(t, g,'Color',[0.8, 0.8, 0.8])
hold on;
plot(t, mean(g, 1),'k','LineWidth', 1.5);
y1 = mean(g, 1)+2*std(g, 0, 1);
y2 = mean(g, 1)-2*std(g, 0, 1);
plot(t, y1,'Color', [155,226,250]/255,'LineWidth', 1);
plot(t, y1,'b +','LineWidth', 1,'MarkerIndices',1:10:length(y1), 'MarkerSize',5,'MarkerMode','manual');
plot(t, y2,'Color', [155,226,250]/255,'LineWidth', 1);
plot(t, y2,'b _','LineWidth', 1,'MarkerIndices',1:10:length(y1), 'MarkerSize',5,'MarkerMode','manual');


time_gap = 1/(N-1);
for m = 1:M
    qn(m,:) = sign(gradient(f(:,m))/time_gap).*sqrt(abs(gradient(f(:,m))/time_gap));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Dynamic programming to match f
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [fn, gam, q] = mainWarpingWrapper(t, f, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bayesian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = zeros(1, N);
mu_star = 0*(t-0.5);
temp_star = cumtrapz(exp(mu_star))./trapz(exp(mu_star));
gamm_star = (temp_star-min(temp_star))/(max(temp_star)-min(temp_star));
% Define the value of pho
pho = 0.999;  % You can change this to your desired value
% Initialize an empty matrix of size (n+1) x (n+1)

fq_cov = ones(1, N);
Cq = diag(fq_cov);
% Fill in the matrix
for i = 1:N
    for j = 1:N
        if i ~= j
            Cq(i, j) = pho^(abs(i-j));
        end
    end
end

Cq = Cq * 5;


%define the covariance for the gamma function
f_cov= [0.001*ones(1, (N-1)/2+21), 20*ones(1, (N-1)/2-20)];
% f_cov= [0.1*ones(1, (N-1)/2+21), 5*ones(1, (N-1)/2-20)];
% f_cov= [0.001*ones(1, (N-1)/2+11), 15*ones(1, (N-1)/2-10)];
Cr1 = diag(f_cov);
sigma_kernel = 8;
kernel_size = 51; % Adjust the size as needed
[X, Y] = meshgrid(-(kernel_size-1)/2:(kernel_size-1)/2, -(kernel_size-1)/2:(kernel_size-1)/2);
gaussian_kernel = exp(-(X.^2 + Y.^2) / (2 * sigma_kernel^2));
Cr2 = 0.001*conv2(Cr1, gaussian_kernel, 'same'); %change to 0.09, we can get sup-optimal
% Cr2 = 0.01*conv2(Cr1, gaussian_kernel, 'same'); %change to 0.09, we can get sup-optimal
Cr = Cr2*1;
[V, D, U] = svd(Cr);
neg_eigenvalues = find(diag(D) < 0);
D(neg_eigenvalues, neg_eigenvalues) = D(neg_eigenvalues, neg_eigenvalues)*-1;
Cr = V * D * V';
figure(1002); clf;
imagesc(Cr);
colorbar;
axis equal;
set(gca,'YDir','normal')
xlim([0 200])
ylim([0 200])
% xticks([0 40 80 120 160 200]);
% xticklabels([0 0.2 0.4 0.6 0.8 1]);
xticks([0 2 5 8 11 14 17]/17*200);
xticklabels([1 3 6 9 12 15 18]);
% yticks([0 40 80 120 160 200]);
% yticklabels([0 0.2 0.4 0.6 0.8 1]);
yticks([0 2 5 8 11 14 17]/17*200);
yticklabels([1 3 6 9 12 15 18]);
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
% 1. initiate g 

% phi_int = mvnrnd(mu, Cr, M);
% for i = 1: M
%     phi_int(i,:) = phi_int(i,:)-trapz(t,phi_int(i,:));
% end 
phi_int = zeros(M,N);

%2. Initiate sigma1
sigma1_int = 1;

%3. Initiate q*
qm_int = mean(qn);
% qm_int = qn(6,:);

qm_set = [];
qm_set(1,:) = qm_int;

%4: update g and sigma1
J = 2000;
phi_set = [];
phi_set(1,:,:) = phi_int;

%set the parameters for the pCN-mixture
betals = [0.5, linspace(0.001, 0.1,9)];
% betalq = [0.5, linspace(0.001, 0.1,9)];
betalq = [0.5, linspace(0.001, 0.1,9)];
probabilities = repmat(0.1, 1, 10);

for j = 1 : J
    for m = 1:M
        %propose new phi
        kesi = mvnrnd(mu, Cr, 1);
        kesi = kesi-trapz(t,kesi);
        beta = randsample(betals, 1,true, probabilities);
        phi_new = squeeze(phi_set(j, m,:))*sqrt(1-beta^2) + beta*kesi';
        phi_new = phi_new-trapz(t,phi_new);
        
        % calculate MCMC acceptance ratio
        [lossratio, ] = cal_joint_ratio_clr_corr(sigma1_int, qm_set(j,:), qn(m,:), t, phi_new, squeeze(phi_set(j, m,:)));
        lamd_p = min(1, lossratio);
    
        if rand()<lamd_p
            phi_set(j+1,m,:) = phi_new;
        else
            phi_set(j+1,m,:) = squeeze(phi_set(j, m,:));
        end

    end
  
    %%center gamma and q*
    psi_mean = mean(squeeze(phi_set(j+1,:,:)), 1);
    gm_mean = cumtrapz(t, exp(psi_mean))./trapz(t, exp(psi_mean));
    gm_mean = (gm_mean-gm_mean(1))/(gm_mean(end)-gm_mean(1));
    gm_mean_inv = interp1(gm_mean, t, t)';
    comp_star = interp1(t, gm_mean_inv, gamm_star);
    gm_mean_inv_dev = gradient(gm_mean_inv, 1/(N-1));
    comp_star_dev = gradient(comp_star, 1/(N-1));
    qm_set(j,:) = interp1(t, qm_set(j,:), comp_star)'.*sqrt(comp_star_dev');
    for m = 1:M
        temp = squeeze(phi_set(j+1,m,:))';
        temp1 = cumtrapz(t, exp(temp))./trapz(t, exp(temp));
        gamma_set(m,:) = (temp1-temp1(1))/(temp1(end)-temp1(1));
        gamma_set(m,:) = interp1(t, gamma_set(m,:), comp_star);
        temp2 = gradient(gamma_set(m,:), 1/(N-1));
        phi_set(j+1,m,:) = log(temp2) - trapz(t, log(temp2));
    end
    %propose new qm*
    kesiqm = mvnrnd(mu, Cq, 1);
    betaqm = randsample(betalq, 1, true, probabilities);
    qm_new = qm_set(j,:)*sqrt(1-betaqm^2) + betaqm*kesiqm;
    [lossratio_q, ] = cal_joint_ratio_clr_mulf_corr(sigma1_int, qn, squeeze(phi_set(j+1,:,:)), t, qm_new, qm_set(j,:));
    lamd_q = min(1, lossratio_q);

    if rand()<lamd_q
        qm_set(j+1,:) = qm_new;
    else
        qm_set(j+1,:) = qm_set(j, :);
    end
end

%% plot the sample gamma
gamma_new = [];

for j = 1: J+1
    for m = 1:M
        temp = squeeze(phi_set(j, m,:));
        gamma_new(j, m,:) = cumsum(exp(temp))./sum(exp(temp));
    end
end

fgamma = gamma_new(J/2+1:10:J+1,:,:);
fphi = phi_set(J/2+1:10:J+1,:,:);

gamma_mean = [];
for m = 1:M
    phi_temp = squeeze(fphi(:,m,:));
    sample_mean = mean(phi_temp, 1);
    sample_mean = cumsum(exp(sample_mean))./sum(exp(sample_mean));
    gamma_mean(m,:) = (sample_mean-min(sample_mean))/(max(sample_mean)-min(sample_mean));
end

figure(50); clf;
plot(t,gamma_mean,'LineWidth',1.5);
lsize = 16; % Label fontsize
nsize = 18; % Axis fontsize
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 2 5 8 11 14 17]/17);
xticklabels([1 3 6 9 12 15 18]);
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
figure(51); clf;
subplot(1,2,1)
plot(t,squeeze(fgamma(:,1,:)),'LineWidth',1);
% plot(t,gamma_mean,'LineWidth',2);
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
subplot(1,2,2)
plot(t,squeeze(fgamma(:,2,:)),'LineWidth',1);
% plot(t,gamma_mean,'LineWidth',2);
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

for m = 1:M
    f1_gamma_t(m,:) = interp1(t,f(:,m),gamma_mean(m,:));
end

figure(52);clf;
plot(t,f,'LineWidth', 1.5);
pbaspect([1 1 1]);
xlim([0,1]);
ylim([0,25]);
xticks([0 2 5 8 11 14 17]/17);
xticklabels([1 3 6 9 12 15 18]);
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


iso = load("female_iso.mat");

figure(53);clf;
plot(t, f1_gamma_t,'LineWidth', 1.5);
hold on
% plot(t, mean(g, 1),'r','LineWidth', 1.5);
% plot(t, mean(iso.f1_gamma_t, 1),'b','LineWidth', 1.5);
plot(t, mean(f1_gamma_t, 1),'k','LineWidth', 2);
y1 = mean(f1_gamma_t, 1)+2*std(f1_gamma_t, 1);
y2 = mean(f1_gamma_t, 1)-2*std(f1_gamma_t, 1);
% plot(t, y1,'Color', [155,226,250]/255,'LineWidth', 1);
% plot(t, y1,'b +','LineWidth', 1,'MarkerIndices',1:10:length(y1), 'MarkerSize',5,'MarkerMode','manual');
% plot(t, y2,'Color', [155,226,250]/255,'LineWidth', 1);
% plot(t, y2,'b _','LineWidth', 1,'MarkerIndices',1:10:length(y1), 'MarkerSize',5,'MarkerMode','manual');
pbaspect([1 1 1]);
xlim([0,1]);
ylim([0, 30])
xticks([0 2 5 8 11 14 17]/17);
xticklabels([1 3 6 9 12 15 18]);
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


for m = 1:M
    f1_gamma_t2(m,:) = interp1(t,f(:,m),squeeze(fgamma(end,m,:)));
end

figure(54);clf;
% plot(t,f,'LineWidth', 1.5);
hold on;
plot(t,f1_gamma_t2,'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);

figure(55);clf;
imagesc(f1_gamma_t2(:,40:end))

figure(56);clf;
hold on;
plot(t, f1_gamma_t2([1:19 21:end],:),'Color', [0.8 0.8 0.8]);
plot(t, f1_gamma_t2(20,:),'k','LineWidth', 2);

iso = load('female_iso.mat');
y_iso = iso.f1_gamma_t;
y1_iso = mean(y_iso, 1)+2*std(y_iso, 1);
y2_iso = mean(y_iso, 1)-2*std(y_iso, 1);
figure(57); clf;
plot(t, mean(f1_gamma_t, 1),'b','LineWidth', 1.5);
hold on
plot(t, y1,'Color', [155,226,250]/255,'LineWidth', 1);
plot(t, y2,'Color', [155,226,250]/255,'LineWidth', 1);
plot(t, mean(y_iso, 1),'k','LineWidth', 1.5);
plot(t, y1_iso,'k--','LineWidth', 1);
plot(t, y2_iso,'k--','LineWidth', 1);
pbaspect([1 1 1]);
xlim([0,1]);
ylim([0, 25])
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