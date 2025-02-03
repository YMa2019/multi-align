%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
N = 101;

seed = round(rand*1000,0);
rng(824); %824

t = linspace(0,1,N);
f1 = @(x) 6*(0.8).^(20*x).*cos(10*pi.*x-pi/4);
f2 = @(x) 6*(0.8).^(20*x).*sin(10*pi.*x);
time_gap = 1/(N-1);
a = [-0.5, 2];
f1 = f1(t)';


% % random generate true gamma2 to align f to get f2
% tan2 = zeros(1, N);
% 
% S = [];
% for j =1: dn
%     S(j) = normrnd(0, sigma/j);
%     tan2 = tan2 + S(j)*b1(j,:);
% end
% tan2 = tan2-log(trapz(t,exp(tan2)));
% 
% temp = exp(tan2)./trapz(t, exp(tan2),2);
% gamma2 = cumsum(temp.^2,2)/sum(temp.^2,2);    

% % set fixed gamma2
gamma2 = (exp(a(2)*t)-1)/(exp(a(2))-1);
gamma_true = interp1(gamma2, t, t);
f2 = 1*interp1(t,f2(t),gamma2);
f2 = f2';
figure(1); clf;
plot(t, f1, t, f2);
q1 = sign(gradient(f1)/time_gap).*sqrt(abs(gradient(f1)/time_gap));
q2 = sign(gradient(f2)/time_gap).*sqrt(abs(gradient(f2)/time_gap));

% temp = q2;
% q2 = q1;
% q1 = temp;
%define the covariance for the gamma function
mu = zeros(1, N);

% Define the value of pho
pho = 0.999;  % You can change this to your desired value
% Initialize an empty matrix of size (n+1) x (n+1)

f_cov = ones(1, N);
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
% 
Cr = 10*Cr;


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

try chol(Cr)
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end

inv_cr = inv(Cr);
figure(1002); clf;
imagesc(inv_cr);

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
sigma1_int = 2;

%3: update g and sigma1
J = 10000;
phi_set = [];
phi_set(1,:) = phi_int;

%set the parameters for the pCN-mixture
betals = [0.5, linspace(0.001, 0.1,9)];
probabilities = repmat(0.1, 1, 10);


for j = 1: J
    %propose new phi
    kesi = mvnrnd(mu, Cr, 1);
    kesi = kesi-trapz(t,kesi);
    %propose new phi
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
    gamma_new(j,:) = cumsum(exp(phi_set(j,:)))./sum(exp(phi_set(j,:)));
    gamma_new(j,:) = (gamma_new(j,:)-min(gamma_new(j,:)))/(max(gamma_new(j,:))-min(gamma_new(j,:)));
end

% burning
fgamma = gamma_new(200:100:J+1,:);
fphi_set = phi_set(200:100:J+1,:);

% posterior mean
sample_mean = mean(phi_set,1);
gamma_mean = cumsum(exp(sample_mean))./sum(exp(sample_mean));
gamma_mean = (gamma_mean-min(gamma_mean))/(max(gamma_mean)-min(gamma_mean));


% plot warping
figure(1); clf;
plot(t,fgamma,'color',[0.7,0.7,0.7],'LineWidth', 1);
hold on;
% plot(t,gamma_new);
plot(t,gamma_true ,'k','LineWidth',2);
% plot(t,gamma_t,'k','LineWidth',2);
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

% plot phi
figure(4); clf;
plot(t,fphi_set,'LineWidth', 1);
hold on;
plot(t,mean(fphi_set),'r','LineWidth', 3);
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

%plot aligned function
f2_gamma_mean = interp1(t,f2,gamma_mean);
for i = 1: length(fgamma)-2
    f2_gamma_t(i,:) = interp1(t,f2,fgamma(i,:));
end 

max_f2_gamma = max(f2_gamma_t);
min_f2_gamma = min(f2_gamma_t);
mean_f2_gamma = mean(f2_gamma_t);
figure(2);clf;
plot(t,f1,'b.-','LineWidth', 1.5);
hold on;
patch([t, fliplr(t)], [min_f2_gamma, fliplr(max_f2_gamma)],[0.7 0.7 0.7], 'EdgeColor',[0.7 0.7 0.7]);
plot(t,f1,'b.-','LineWidth', 1.5);
plot(t,f2,'g.-','LineWidth', 1.5);
plot(t, f2_gamma_mean, 'r','LineWidth', 1.5)
% plot(t, mean_f2_gamma, 'r','LineWidth', 1.5)
legend({'','','$f_1$','$f_2$'},'Interpreter','latex','Box','off', 'Fontsize', 30)
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


% gamma_inv = invertGamma(gamma_mean);
% figure(1);hold on;
% plot(t,gamma_inv','c','LineWidth',2);


%plot aligned function
% f1_gamma_t = interp1(t,f1,gamma_inv);
figure(21);clf;
plot(t,f1,'b.','LineWidth', 1.5);
hold on;
plot(t,f2,'g.','LineWidth', 1.5);
% plot(t, f1_gamma_t, 'r','LineWidth', 1.5)