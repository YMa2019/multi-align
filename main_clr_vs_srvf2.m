% Generate the time warping functions without prior information using
% Algorithm 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 1: Create Fourier basis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
lsize = 16; % Label fontsize
nsize = 18; % Axis fontsize

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Step 2: Generate the simple warpings
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(456);
d = 100;
t = linspace(0,1,d);
time_diff = mean(diff(t));
n = 200;
% a = exprnd(2,[N,1]);
% a = laprnd(N,1,2,4);
% a = rand(n/2,1)'*(-2)-3;
% a = [a, rand(n/2,1)'*2 + 3];
x_gen = zeros(n,d);
a = normrnd(0,4,[n,1]);
for i =1:n
    gam_gen(i,:) = (exp(a(i)*t)-1)/(exp(a(i))-1);
end

figure (2);clf;
h = plot(t, gam_gen,'linewidth', 1);% plot the simulated warping function
for i=1:numel(h)
    c = get(h(i), 'Color');
    set(h(i), 'Color', [c 0.8]);
end
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
for i = 1:n
x_gen(i,:) = log(gradient(gam_gen(i,:)));
x_gen(i,:) = x_gen(i,:) - trapz(t, x_gen(i,:));
end

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
h = plot(t, gam_clr,'linewidth', 1);% plot the simulated warping function
for i=1:numel(h)
    c = get(h(i), 'Color');
    set(h(i), 'Color', [c 0.8]);
end
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
% title('Resample in \Gamma_1 via CLR');
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
plot(t, gam_srvf,'color',[0.8,0.8,0.8],'linewidth', 1);% plot the simulated warping function
hold on;
% plot(t, gam_srvf([12 168 130 172 175],:),'k','linewidth', 1.5);% plot the simulated warping function
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

cnt1 = 1;
cnt2 = 1;
for i = 1:n
    if all(vec_exp(i,:)>0)
        gam_srvf1(cnt1,:) = cumsum(vec_exp(i,:).*abs(vec_exp(i,:)),2)./sum(vec_exp(i,:).*abs(vec_exp(i,:)),2);
        gam_srvf1(cnt1,:) = (gam_srvf1(cnt1,:)-min(gam_srvf1(cnt1,:)))/(max(gam_srvf1(cnt1,:))-min(gam_srvf1(cnt1,:))); 
        cnt1 = cnt1 + 1;
    else
        gam_srvf2(cnt2,:) = cumsum(vec_exp(i,:).*abs(vec_exp(i,:)),2)./sum(vec_exp(i,:).*abs(vec_exp(i,:)),2);
        gam_srvf2(cnt2,:) = (gam_srvf2(cnt2,:)-min(gam_srvf2(cnt2,:)))/(max(gam_srvf2(cnt2,:))-min(gam_srvf2(cnt2,:)));
        cnt2 = cnt2 + 1;
    end
end
figure (10);clf;
h = plot(t, gam_srvf1,'linewidth', 1);% plot the simulated warping function
for i=1:numel(h)
    c = get(h(i), 'Color');
    set(h(i), 'Color', [c 0.8]);
end
hold on;
plot(t, gam_srvf2,'color',[1 0 0 0.8],'linewidth', 2.5);% plot the simulated warping function
plot(t, gam_srvf1([15 43 44 65 85 110 133 147 158],:),'k','linewidth', 2.5);% plot the simulated warping function
% plot(t, gam_srvf2(10,:),'r','linewidth', 1.5);% plot the simulated warping function
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
% title('Resampled warping via SRVF');
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