% Generate the time warping functions without prior information using
% Algorithm 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 1: Create Fourier basis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
%addpath('supplement\');
rng(20230410); % change the seed
d = 100;
t = linspace(0,1,d);
n = 150;
for k =1:d
f1(2*k-1,:) = sqrt(2)*sin(2*k*pi*t) ;
f1(2*k,:) = sqrt(2)*cos(2*k*pi*t);
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Step 2: Generate the simple warpings
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_gen = zeros(n,d);
basis = sqrt(3)*2*(t-0.5);
% % 1). Gaussian process
% temp = normrnd(0,1,[n,1]).*basis;
% x_gen = temp;

% 2). Gaussian process
a1 = -3;
b1 = -1.0;

a2 = 1.0;
b2 = 3;
intervals = [[a1, b1]; [a2, b2]];
for i = 1: n
    p = rand;
    if p <= 0.5
        a = intervals(1, 1);
        b = intervals(1, 2);
    else
        a = intervals(2, 1);
        b = intervals(2, 2);
    end
    temp = (a + (b - a) * rand()).*basis;
    x_gen(i,:) = temp-log(trapz(t,exp(temp)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 3: Plot the simulated time warping functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lsize = 16; % Label fontsize
nsize = 18; % Axis fontsize
figure (1);clf;
plot(t, x_gen,'linewidth', 1); % plot the stochastic process in H(0, 1)
xticks([0 0.2 0.4 0.6 0.8 1]);
ylim([-10,10]);
title('Given data in H(0,1)');
% xlim([0,1]);
pbaspect([1 1 1]);
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width = 12;
opts.height = 10;
opts.fontType = 'Times';
% apply the inverse-clr transfromation to get warping function
temp = exp(x_gen)./(trapz(t,exp(x_gen),2));
gam_gen= cumsum(temp,2)./sum(temp,2);
figure (2);clf;
plot(t, gam_gen,'linewidth', 1);% plot the simulated warping function
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
% title('Given data in \Gamma_1');
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width = 12;
opts.height = 10;
opts.fontType = 'Times';
%% model observation in the CLR framework
% estimate mean and covariance from the given sample (we can direclty use
% functions in H(0,1), which is one-to-one to functions in \Gamma_1) 
mu = mean(x_gen);
Sigma = cov(x_gen);
figure(3);
imagesc(Sigma);
title('Cov in H(0,1)')
set(gca, 'Fontsize', 20);
% resampling using the estimated covariance
x_clr = mvnrnd(mu, Sigma, n);
figure (4);clf;
plot(t, x_clr,'linewidth', 1); % plot the stochastic process in H(0, 1)
xticks([0 0.2 0.4 0.6 0.8 1]);
ylim([-10,10]);
% xlim([0,1]);
title('Resample in H(0,1)');
pbaspect([1 1 1]);
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width = 12;
opts.height = 10;
opts.fontType = 'Times';
% apply the inverse-clr transfromation to get warping function
gam_clr = exp(x_clr)./(trapz(t,exp(x_clr),2));
gam_clr= cumsum(gam_clr,2)./sum(gam_clr,2);
figure (5);clf;
plot(t, gam_clr,'linewidth', 1);% plot the simulated warping function
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
title('Resample in \Gamma_1 via CLR');
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width = 12;
opts.height = 10;
opts.fontType = 'Times';

%%% resample using FPCA
[U, S, V] = svd(Sigma);
time_diff = mean(diff(t));
S = S*time_diff;
dn =1;
for j = 1:dn
    U1(j,:) = U(:,j);
    U1(j,:) = U1(j,:)/sqrt(trapz(t, U1(j,:).^2)); 
    coeff(j,:) = trapz(t, (x_gen-mu).*U1(j,:),2);
    pd(j,:) = fitdist(coeff(j,:)','Kernel');
end

%use pca for reconstruction get the coeff hist 
figure(6);clf;
h = histogram(coeff(1,:),20);
h(1).FaceColor = [0 .5 .5];
pbaspect([1 1 1]);
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

x_new2 = zeros(n,d);
a1_min = min(coeff);
a1_max = max(coeff(coeff < 0));
a2_min = min(coeff(coeff > 0));
a2_max = max(coeff);
new_a = rand(n/2,1)'*(a1_max-a1_min)+a1_min;
new_a = [new_a, rand(n/2,1)'*(a2_max-a2_min) + a2_min];
for k =1: dn
    temp = new_a'.*U1(k,:);
    x_new2 = x_new2 +temp;
end
x_new2 = x_new2 +mu;

theta_B3 = exp(x_new2)./(trapz(t,exp(x_new2),2));
xnew_theta2= cumsum(theta_B3,2)./sum(theta_B3,2);
% xnew_theta2=normalize(xnew_theta2','range');
figure (7);clf;
plot(t, xnew_theta2);
% title('Resample in \Gamma_1 via CLR FPCA');
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
%% model observation in the SRVF framework
% compute the SRVF
x_srvf = sqrt(gradient(gam_gen));
x_srvf = x_srvf./sqrt(trapz(t, x_srvf.^2, 2));
% Inv-Exp map to the tangent space at gamma_id
mu_srvf = ones(1,d);
for i=1:n
len = acos(trapz(t, mu_srvf.*x_srvf(i,:)));
vec(i,:) = (len/sin(len))*(x_srvf(i,:) - cos(len)*mu_srvf);
end
% estimate mean and covariance 
mu_tg = mean(vec);
Sigma_tg = cov(vec);
figure(8);
imagesc(Sigma_tg); 
title('Cov in SRVF tangent')
set(gca, 'Fontsize', 20);
% resampling using the estimated covariance
vec_tg = mvnrnd(mu_tg, Sigma_tg, n);
% Exp map to the sphere and warping 
for i = 1:n
lenm(i) = sqrt(trapz(t, vec_tg(i,:).^2)); 
vec_exp(i,:) = cos(lenm(i))*mu_srvf + (sin(lenm(i))/lenm(i))*vec_tg(i,:);
end
gam_srvf = cumsum(vec_exp.^2,2)./sum(vec_exp.^2,2);
figure (9);clf;
plot(t, gam_srvf,'linewidth', 1);% plot the simulated warping function
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
title('Resample in \Gamma_1 via SRVF');
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width = 12;
opts.height = 10;
opts.fontType = 'Times';

%%% resample using FPCA
[U_tg, S_tg, V_tg] = svd(Sigma_tg);
S_tg = S_tg*time_diff;

for j = 1:dn
    U1_tg(j,:) = U_tg(:,j);
    U1_tg(j,:) = U1_tg(j,:)/sqrt(trapz(t, U1_tg(j,:).^2)); 
    coeff_tg(j,:) = trapz(t, (vec-mu_tg).*U1_tg(j,:),2);
    pd_tg(j,:) = fitdist(coeff_tg(j,:)','Kernel');
end

%Plot principal direction 
figure(101);clf;
plot(t, U1_tg)

figure(102);clf;
plot(t, U1)

%use pca for reconstruction get the coeff hist 
figure(10);clf;
h = histogram(coeff_tg(1,:),20);
h(1).FaceColor = [0 .5 .5];
pbaspect([1 1 1]);
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



%%resampling 
cnt = 1;
while cnt<n
    vec_tg2 = 0;
    for i = 1:dn
    vec_tg2 = vec_tg2+ random(pd_tg(i,:),[1,1]).*U1_tg(i,:);
    end
    vec_tg2 = vec_tg2+mu_tg;
    lenm2 = sqrt(trapz(t, vec_tg2.^2)); 
    temp2 =  cos(lenm2)*mu_srvf + (sin(lenm2)/lenm2)*vec_tg2;
    if all(temp2>0)
        vec_exp2(cnt,:) = temp2;
        cnt = cnt + 1;
    end
end
gam_srvf2 = cumsum(vec_exp2.^2,2)./sum(vec_exp2.^2,2);
figure (110);clf;
plot(t, vec_exp2,'linewidth', 1);% plot the simulated warping function


figure (11);clf;
plot(t, gam_srvf2,'linewidth', 1);% plot the simulated warping function
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
% title('Resample in \Gamma_1 via SRVF fpca');
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width = 12;
opts.height = 10;
opts.fontType = 'Times';