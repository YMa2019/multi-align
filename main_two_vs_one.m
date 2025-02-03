%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
N = 101;

seed = rand*1000;
rng("default");

t1 = linspace(-3,3,N);
t = linspace(0,1,N);
a = 0.5;
time_gap = 1/(N-1);

for j =1:N 
%     f(j) = 1.5*exp(-(t1(j)-1.7).^2/2) + 1.5*exp(-(t1(j)+1.7).^2/2);
    f(j) = 0.86*exp(-(t(j)-0.25).^2*80) + 0.86*exp(-(t(j)-0.75).^2*80);
end

% f = f - interp1([-3 3], [f(1) f(end)],t1);

a = [-0.5, 2];


gamma1 = t;
f1 = interp1(t,f,gamma1);

for j =1:N 
%     g(j) = exp(-t1(j).^2/2);
    g(j) = exp(-(t(j)-0.5).^2*20);
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
% mu = -2*t;
% f_cov = ones(1, N);
% % f_cov= [2*ones(1, (N-1)/2+1), ones(1, (N-1)/2)];
% Cr = diag(f_cov);
% sigma_kernel = 1.0;
% kernel_size = 11; % Adjust the size as needed
% [X, Y] = meshgrid(-(kernel_size-1)/2:(kernel_size-1)/2, -(kernel_size-1)/2:(kernel_size-1)/2);
% gaussian_kernel = exp(-(X.^2 + Y.^2) / (2 * sigma_kernel^2));
% Cr = conv2(Cr, gaussian_kernel, 'same');


% Define the value of pho
pho = 0.999;  % You can change this to your desired value
% Initialize an empty matrix of size (n+1) x (n+1)

f_cov = ones(1, N);
% f_cov= [20*ones(1, (N-1)/2+1), 0.1*ones(1, (N-1)/2)];
Cr = diag(f_cov);
% Cr = zeros(N, N);

% Fill in the matrix
for i = 1:N
    for j = 1:N
        if i ~= j
            Cr(i, j) = pho^(abs(i-j));
        end
    end
end

Cr = Cr*10;
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
% A = rand(N, N);
% 
% % Form a symmetric matrix by multiplying A with its transpose
% Cr = A * A';
% 
% % Ensure the matrix is positive definite by adjusting eigenvalues
% min_eigenvalue = min(eig(Cr));
% if min_eigenvalue <= 0
%     Cr = Cr + eye(N) * (-min_eigenvalue + 1e-5);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Dynamic programming to match f1 and f2 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma_t = DynamicProgrammingQ(q2', q1', 0, 0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% bayesian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. initiate phi 
% svd of Cr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% bayesian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. initiate phi 
% svd of Cr
[U, S, V] = svd(Cr);
S = S*time_gap;

dn = 5;
for j = 1:dn
    U1(j,:) = U(:,j);
    U1(j,:) = U1(j,:)/sqrt(trapz(t, U1(j,:).^2)); 
end

phi_int = zeros(1, N);

% for j = 1:dn
%     phi_int = phi_int + normrnd(0, sqrt(S(j,j)))*U1(j,:);
% end
% phi_int = phi_int-trapz(t,phi_int);

% phi_int = mvnrnd(mu, Cr, 1);
% phi_int = phi_int-trapz(t,phi_int);

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
betals = [linspace(0.001, 1,10)];
probabilities = repmat(0.1, 1, 10);
% betals = [0.5, linspace(0.001, 0.5,9)];
% probabilities = repmat(0.1, 1, 10);

for j = 1: J
    %propose new phi
    kesi = mvnrnd(mu, Cr, 1);
%     kesi = zeros(1, N);
%     for jj = 1:dn
%         kesi = kesi + normrnd(0, sqrt(S(jj,jj)))*U1(jj,:);
%     end
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
    gamma_new(j,:) = cumtrapz(t, exp(phi_set(j,:)))./trapz(t, exp(phi_set(j,:)));
    gamma_new(j,:) = (gamma_new(j,:)-min(gamma_new(j,:)))/(max(gamma_new(j,:))-min(gamma_new(j,:)));
end


fgamma = gamma_new(100:50:J+1,:);
fphi_set = phi_set(100:50:J+1,:);

sample_mean = mean(phi_set,1);
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
% 
figure (8);clf;
for i =1:N
    eigenvalue(i) = S(i,i);
end
plot(cumsum(eigenvalue(1:50))/sum(eigenvalue),'linewidth', 1.5);


%%
% Perform k-means clustering
K = 2;
% [idx, centers] = kmeans(phi_set, K);
[idx, centers] = kmeans(fphi_set, K);
for i = 1: K
    centers_g(i,:) = cumtrapz(t, exp(centers(i,:)))/trapz(t, exp(centers(i,:)));
end

figure(11); clf;

% plot(t,gamma_new(idx==1,:),'Color', [0.8 0.8 0.8]);
% plot(t,gamma_new(idx==2,:),'y');
plot(t,fgamma(idx==2,:)','Color', [0.8 0.8 0.8]);
hold on;
plot(t,fgamma(idx==1,:),'y');
plot(t, centers_g(2,:),'m','LineWidth',2)
plot(t, centers_g(1,:),'c','LineWidth',2)
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

% plot(t,phi_set(idx==1,:),'Color', [0.8 0.8 0.8]);
% plot(t,phi_set(idx==2,:),'y');
plot(t,fphi_set(idx==2,:)','Color', [0.8 0.8 0.8]);
hold on;
plot(t,fphi_set(idx==1,:),'y');
plot(t, centers(2,:),'m','LineWidth',2)
plot(t, centers(1,:),'c','LineWidth',2)
lsize = 16; % Label fontsize
nsize = 18; % Axis fontsize
pbaspect([1 1 1]);
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
plot(t,f1,'b--','LineWidth', 1.5);
hold on;
plot(t,f2,'g--','LineWidth', 1.5);
% plot(t, f2_gamma_t, 'r--','LineWidth', 1.5)
plot(t, f2_gamma_t1, 'm','LineWidth', 1.5)
plot(t, f2_gamma_t2, 'c','LineWidth', 1.5)

legend({'$f_1$','$f_2$','$f_2\circ \gamma$','$f_2\circ \gamma$'},'Interpreter','latex','Box','off')
pbaspect([1 1 1]);
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