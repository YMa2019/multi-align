%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
N = 201;

seed = rand*1000;
rng("default");

t = linspace(0,1,N);
t1 = linspace(-3,3,N);
time_gap = 1/(N-1);
for j =1:N 
    g(j) = exp(-(t1(j)-0.2).^2/0.2);
%     g(j) = exp(-(t(j)-0.53).^2*180);
end
f2 = interp1(t,g,t);

U1 = sqrt(3)*2*(t-0.5);
h1 = -1.0*U1;
h2 = -0.2*U1;
h3 = 1.0*U1;

gamma1 = cumtrapz(exp(h1))./trapz(exp(h1));
gamma1 = (gamma1-min(gamma1))/(max(gamma1)-min(gamma1));
gamma2 = cumtrapz(exp(h2))./trapz(exp(h2));
gamma2 = (gamma2-min(gamma2))/(max(gamma2)-min(gamma2));
gamma3 = cumtrapz(exp(h3))./trapz(exp(h3));
gamma3 = (gamma3-min(gamma3))/(max(gamma3)-min(gamma3));

f1_1 = interp1(t,f2,gamma1);
f1_2 = interp1(t,f2,gamma2);
f1_3 = interp1(t,f2,gamma3);
f1 = f1_1 + f1_2*0.5 + f1_3;

figure(1);clf;
plot(t, f1, t, f2)

f1 = f1';
f2 = f2';

q1 = sign(gradient(f1)/time_gap).*sqrt(abs(gradient(f1)/time_gap));
q2 = sign(gradient(f2)/time_gap).*sqrt(abs(gradient(f2)/time_gap));

%define the covariance for the gamma function
mu = zeros(1, N);


% Define the value of pho
pho = 0.999;  % You can change this to your desired value
% Initialize an empty matrix of size (n+1) x (n+1)

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
Cr = Cr*8;

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
%%%  Gaussian bayesian
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
sigma1_int = 2;

%3: update g and sigma1
J = 10000;
phi_set = [];
phi_set(1,:) = phi_int;

%set the parameters for the pCN-mixture


% betals = [0.9, linspace(0.001, 0.08, 7)]; %742
% probabilities = [0.3, repmat(0.1, 1, 7)];

% betals = [0.6, linspace(0.001, 0.008, 6)];
% probabilities = [0.4, repmat(0.1, 1, 6)];

betals = [0.9,  linspace(0.001, 0.09, 7)];
probabilities = [0.3,  repmat(0.1, 1, 7)];

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

sample_mean = mean(phi_set,1);
gamma_mean = cumsum(exp(sample_mean))./sum(exp(sample_mean));
gamma_mean = (gamma_mean-min(gamma_mean))/(max(gamma_mean)-min(gamma_mean));

fgamma = gamma_new(2:2:J+1,:);
fphi_set = phi_set(2:2:J+1,:);

sample_mean = mean(fphi_set,1);
gamma_mean = cumsum(exp(sample_mean))./sum(exp(sample_mean));
gamma_mean = (gamma_mean-min(gamma_mean))/(max(gamma_mean)-min(gamma_mean));

figure(1); clf;
hold on;
plot(t,gamma_new);
plot(t,gamma_t,'k','LineWidth',2);
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
figure(2);clf;
plot(t,f1,'b.','LineWidth', 1.5);
hold on;
plot(t,f2,'g.','LineWidth', 1.5);
plot(t, f2_gamma_t, 'r','LineWidth', 1.5)

figure(3);clf;
plot(t,q1,'b.','LineWidth', 1.5);
hold on;
plot(t,q2,'g.','LineWidth', 1.5);

figure(5);clf;
subplot(2,1,1);
plot(sse_)
subplot(2,1,2);
hist(sse_);


% Perform k-means clustering
seed = rand*1000;
rng(419);
K = 3;
[idx, centers] = kmeans(phi_set, K);
% [idx, centers] = kmeans(fphi_set, K);
for i = 1: K
    centers_g(i,:) = cumtrapz(t, exp(centers(i,:)))/trapz(t, exp(centers(i,:)));
end

figure(11); clf;
plot(t,gamma_new(idx==2,:),'Color', [0.8 0.8 0.8]);
hold on;
plot(t,gamma_new(idx==1,:),'y');
plot(t,gamma_new(idx==3,:),'g');
% plot(t,fgamma(idx==1,:)','Color', [0.8 0.8 0.8]);
% plot(t,fgamma(idx==2,:),'y');
% plot(t,fgamma(idx==3,:),'g');
plot(t, centers_g(2,:),'m','LineWidth',2)
plot(t, centers_g(1,:),'c','LineWidth',2)
plot(t, centers_g(3,:),'r','LineWidth',2)
% plot(t,gamma_t,'k--','LineWidth',2);
% plot(t,gamma_t,'m--','LineWidth',2);
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



f2_gamma_t1 = interp1(t,f2,centers_g(1,:));
f2_gamma_t2 = interp1(t,f2,centers_g(2,:));
f2_gamma_t3 = interp1(t,f2,centers_g(3,:));
figure(22);clf;
% plot(t,f1,'b.-','LineWidth', 1.5);
% plot(t,f2,'g.-','LineWidth', 1.5);
plot(t, f2_gamma_t2, 'm','LineWidth', 1.5)
hold on;
plot(t, f2_gamma_t1, 'c','LineWidth', 1.5)
plot(t, f2_gamma_t3,'r', 'LineWidth',1.5)
% plot(t, f2_gamma_t, 'k--','LineWidth', 1.5)
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
%PCA
time_diff = time_gap;
v = phi_set;

figure(4); clf;
plot(t, v, 'Color', [0.5 0.5 0.5]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Non-Gaussian bayesian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
N = 201;

seed = rand*1000;
rng("default");

t = linspace(0,1,N);
t1 = linspace(-3,3,N);
time_gap = 1/(N-1);
for j =1:N 
    g(j) = exp(-(t1(j)-0.2).^2/0.2);
end
f2 = interp1(t,g,t);

U1 = sqrt(3)*2*(t-0.5);
h1 = -1.0*U1;
h2 = -0.1*U1;
h3 = 1.0*U1;

gamma1 = cumtrapz(exp(h1))./trapz(exp(h1));
gamma1 = (gamma1-min(gamma1))/(max(gamma1)-min(gamma1));
gamma2 = cumtrapz(exp(h2))./trapz(exp(h2));
gamma2 = (gamma2-min(gamma2))/(max(gamma2)-min(gamma2));
gamma3 = cumtrapz(exp(h3))./trapz(exp(h3));
gamma3 = (gamma3-min(gamma3))/(max(gamma3)-min(gamma3));

f1_1 = interp1(t,f2,gamma1);
f1_2 = interp1(t,f2,gamma2);
f1_3 = interp1(t,f2,gamma3);
f1 = f1_1 + f1_2*0.5 + f1_3;

figure(1);clf;
plot(t, f1, t, f2)

f1 = f1';
f2 = f2';


q1 = sign(gradient(f1)/time_gap).*sqrt(abs(gradient(f1)/time_gap));
q2 = sign(gradient(f2)/time_gap).*sqrt(abs(gradient(f2)/time_gap));

%define the covariance for the gamma function
mu = zeros(1, N);


% Define the value of pho
pho = 0.999;  % You can change this to your desired value
% Initialize an empty matrix of size (n+1) x (n+1)

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
Cr = Cr*8;

figure(1001); clf;
imagesc(Cr);
try chol(Cr);
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
%define the covariance for the gamma function
mu = zeros(1, N);
gamma_mu = cumsum(exp(mu))./sum(exp(mu));
gamma_mu = (gamma_mu-min(gamma_mu))/(max(gamma_mu)-min(gamma_mu)); 

dn = 1;
U1 = sqrt(3)*2*(t-0.5);
% 1. initiate phi 
a1 = -1.5;
b1 = -0.5;

a2 = 0.5;
b2 = 1.5;

intervals = [[a1, b1]; [a2, b2]];
p = rand;
if p <= 0.5
    a = intervals(1, 1);
    b = intervals(1, 2);
else
    a = intervals(2, 1);
    b = intervals(2, 2);
end

phi_int = mu;
coeff(1, 1) = a + (b - a) * rand();


for j = 1:dn
    phi_int = phi_int + coeff(j, 1)*U1(j,:);
end
phi_int = phi_int-log(trapz(t,exp(phi_int)));

figure(101); clf;
plot(t, phi_int);
figure(102); clf;
gamma_in = cumsum(exp(phi_int))./sum(exp(phi_int));
gamma_in = (gamma_in-min(gamma_in))/(max(gamma_in)-min(gamma_in)); 
plot(t, gamma_in);

%2. Initiate sigma1
sigma1_int = 3;

%3: update g and sigma1
J = 10000;
coeff_set = zeros(5, J);
phi_set = [];
coeff_set(:,1) = coeff;
phi_set(1,:) = phi_int;

u_coeff = [0, 0, 0, 0, 0];
sigma_coeff = [0, sqrt(0.4)/3, sqrt(0.3)/3, sqrt(0.2)/3/3, sqrt(0.15)/4];
cnt = 0; 
for j = 1: J
    %propose new phi   
    phi_new = mu;
    for jj = 1: dn
        if jj == 1
            coeff_new(jj) = proposal_distribution(coeff_set(jj,j), 1, 1, a1, a2, b1, b2);
        else
            coeff_new(jj) = proposal_distribution2(coeff_set(jj,j), 0.01, 1, u_coeff(jj), sigma_coeff(jj));
        end 
    end

    for jj = 1: dn        
        phi_new = phi_new + coeff_new(jj)*U1(jj,:);
    end
    phi_new = phi_new-log(trapz(t,exp(phi_new)));
    % calculate MCMC acceptance ratio
    prior = 1;
    if jj>1
        for jj = 2: dn
            prior = prior * normpdf(coeff_new(jj),u_coeff(jj),sigma_coeff(jj))/normpdf(coeff_set(jj,j),u_coeff(jj),sigma_coeff(jj));
        end
    end
    if (coeff_new(1) >= a1 && coeff_new(1) <= b1) || (coeff_new(1) >= a2 && coeff_new(1) <= b2)
        [lossratio, sse_diff(j)] = cal_joint_ratio_clr(sigma1_int, q1, q2, t, phi_new, phi_set(j,:));
        lossratio_(j) = lossratio*prior;
        lamd_p = min(1, lossratio*prior);
    else
        lamd_p = 0;
    end

    if rand()<lamd_p
        phi_set(j+1,:) = phi_new;
        coeff_set(:,j+1) =coeff_new;
        cnt = cnt + 1;
    else
        phi_set(j+1,:) = phi_set(j,:);
        coeff_set(:,j+1) =coeff_set(:,j);
    end
    
    temp_t = cumtrapz(t,exp(phi_set(j+1,:)))./trapz(t, exp(phi_set(j+1,:)),2);
    temp_t = round(temp_t/temp_t(end)*(N-1))+1;
    gam1_dev = exp(phi_set(j+1,:));
    SSE = (norm(q2 - q1(temp_t).*sqrt(gam1_dev')))^2;
    sse_(j) = SSE;

end

% plot the sample gamma
gamma_new = [];
for j = 1:J+1
    gamma_new(j,:) = cumsum(exp(phi_set(j,:)))./sum(exp(phi_set(j,:)));
    gamma_new(j,:) = (gamma_new(j,:)-min(gamma_new(j,:)))/(max(gamma_new(j,:))-min(gamma_new(j,:)));
end


sample_mean = mean(phi_set,1);
gamma_mean = cumsum(exp(sample_mean))./sum(exp(sample_mean));
gamma_mean = (gamma_mean-min(gamma_mean))/(max(gamma_mean)-min(gamma_mean));

fgamma = gamma_new(2:10:J+1,:);
fphi_set = phi_set(2:10:J+1,:);

sample_mean = mean(fphi_set,1);
gamma_mean = cumsum(exp(sample_mean))./sum(exp(sample_mean));
gamma_mean = (gamma_mean-min(gamma_mean))/(max(gamma_mean)-min(gamma_mean));



figure(1); clf;
hold on;
plot(t,gamma_new);
% plot(t,fgamma);
plot(t,gamma_t,'k','LineWidth',2);
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
f2_gamma_b = interp1(t,f2,gamma_mean);
figure(2);clf;
hold on;
plot(t,f1,'b','LineWidth', 1.5);
plot(t,f2,'g','LineWidth', 1.5);
plot(t, f2_gamma_t, 'k -.','LineWidth', 1.5)
plot(t, f2_gamma_b, 'r -.','LineWidth', 1.5)

figure(3);clf;
plot(t,q1,'b.','LineWidth', 1.5);
hold on;
plot(t,q2,'g.','LineWidth', 1.5);

figure(5);clf;
subplot(2,1,1);
plot(sse_)
subplot(2,1,2);
hist(sse_);



% Perform k-means clustering
K = 2;
[idx, centers] = kmeans(phi_set, K);
% [idx, centers] = kmeans(fphi_set, K);
for i = 1: K
    centers_g(i,:) = cumtrapz(t, exp(centers(i,:)))/trapz(t, exp(centers(i,:)));
end

figure(11); clf;
plot(t,gamma_new(idx==1,:),'Color', [0.8 0.8 0.8]);
hold on;
plot(t,gamma_new(idx==2,:),'y');
% plot(t, gamma_mu, 'g','LineWidth',2);
% plot(t,fgamma(idx==1,:)','Color', [0.8 0.8 0.8]);
% plot(t,fgamma(idx==2,:),'y');
plot(t, centers_g(1,:),'m','LineWidth',2)
plot(t, centers_g(2,:),'c','LineWidth',2)
% plot(t,gamma_t,'r--','LineWidth',2);
% plot(t,gamma_t,'m--','LineWidth',2);
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

figure(12); clf;
hold on;
plot(t,phi_set(idx==1,:),'Color', [0.8 0.8 0.8]);
plot(t,phi_set(idx==2,:),'y');
plot(t, centers(1,:),'m','LineWidth',2)
plot(t, centers(2,:),'c','LineWidth',2)
plot(t, mu, 'g','LineWidth',2)
lsize = 16; % Label fontsize
nsize = 18; % Axis fontsize
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

f2_gamma_t1 = interp1(t,f2,centers_g(1,:));
f2_gamma_t2 = interp1(t,f2,centers_g(2,:));
figure(22);clf;
% plot(t,f1,'b','LineWidth', 1.5);
% hold on;
% plot(t,f2,'g','LineWidth', 1.5);
% legend({'$f_1$','$f_2$'},'Interpreter','latex','Box','off', 'Fontsize', 30)
plot(t, f2_gamma_t1, 'm','LineWidth', 1.5)
hold on;
plot(t, f2_gamma_t2, 'c','LineWidth', 1.5)
% plot(t, f2_gamma_t, 'r--','LineWidth', 1.5)
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


figure(101); clf;
plot(coeff_set(1,:))