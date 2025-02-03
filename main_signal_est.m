%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all;
nsize = 18;
rng("default");
N = 101;
time_gap = 1/(N-1);
t = linspace(0, 1, N);
for j =1:N 
    g(j) = 0.99*exp(-(t(j)-0.22).^2*12) + 1.01*exp(-(t(j)-0.78).^2*12);
end
figure(1);clf;
plot(t, g,'linewidth', 1.2);
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

M_SET = [5, 10:10:60];
M_SET = 60;
U = sqrt(3)*2*(t-0.5);
record = [];
cnt = 1;
for M = M_SET
    for i = 1:M
        coeff = normrnd(0, 0.2);
        c = normrnd(1, 0.1);
        error = normrnd(0, 0.1);
        h = coeff*U;
        temp = cumtrapz(exp(h))./trapz(exp(h));
        gamma(i,:) = (temp-min(temp))/(max(temp)-min(temp));
        f(i,:) = c*interp1(t,g, gamma(i,:))+error;
        q(i,:) = sign(gradient(f(i,:))/time_gap).*sqrt(abs(gradient(f(i,:))/time_gap));
    end
% figure(2);clf;
% hold on;
% plot(t, gamma,'linewidth', 1.2);
% plot(t, t, 'k','linewidth', 2)
% axis equal;
% ylim([0,1]);
% xlim([0,1]);
% xticks([0 0.2 0.4 0.6 0.8 1]);
% set(gca, 'Fontsize', nsize,'linewidth', 1.5)
% set(gcf,'paperpositionmode','auto');
% set(gcf,'windowstyle','normal');
% set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
% set(gca,'fontweight','normal')
% opts.Colors     = get(groot,'defaultAxesColorOrder');
% opts.saveFolder = 'img/';
% opts.width      = 12;
% opts.height     = 10;
% opts.fontType   = 'Times';
% 
figure(3);clf;
plot(t, f,'linewidth', 1.2);
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
% 
% figure(4);clf;
% plot(t, q);

% initial a qg =  q1
    qg = q(1,:);
    fg = f(1,:);
    for i = 1:M
        temp = DynamicProgrammingQ(q(i,:), qg, 0, 0);
        gamma_et(i,:) = temp';
        h_temp = gradient(gamma_et(i,:), t);
        h_et(i,:) = log(h_temp)-trapz(t, log(h_temp));
    end
    h_bar = mean(h_et);
    temp_bar = cumtrapz(exp(h_bar))./trapz(exp(h_bar));
    gamma_bar = (temp_bar-min(temp_bar))/(max(temp_bar)-min(temp_bar));
    inv_gamma_bar = interp1(gamma_bar, t, t)';
    
    for i = 1:M
        cen_gamma_et(i,:) = interp1(t, gamma_et(i,:), inv_gamma_bar);
        f_(i,:) = interp1(t, f(i,:), cen_gamma_et(i,:));
    end
    est_g = mean(f_);
    record(cnt) = norm(g-est_g);
    cnt = cnt + 1;
end


figure(5);clf;
plot(t, est_g,'linewidth', 1.2);
hold on;
plot(t, g,'linewidth', 1.2);
legend({'{$g$}','$\hat{g}_n$'},'Interpreter','latex','Box','off','FontSize',25)
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

for i = 1: M_SET
    f_al(i,:) = interp1(t,f(i,:), cen_gamma_et(i,:));
end
figure(6);clf;
plot(t, f_al,'linewidth', 1.2);
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

figure(7);clf;
plot(t, cen_gamma_et,'linewidth', 1.2);
hold on;
plot(t, mean(cen_gamma_et),'k','linewidth', 2);
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
% figure(6);clf;
% plot(M_SET(1:7), record(1:7),'linewidth', 1.2);
% hold on;
% scatter(M_SET(1:7), record(1:7),50,'b.');
% % xlim([5 5000])
% xticks([5, 10, 20, 30, 40, 50, 60]);
% pbaspect([1 1 1]);
% set(gca, 'Fontsize', nsize,'linewidth', 1.5)
% set(gcf,'paperpositionmode','auto');
% set(gcf,'windowstyle','normal');
% set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
% set(gca,'fontweight','normal')
% opts.Colors     = get(groot,'defaultAxesColorOrder');
% opts.saveFolder = 'img/';
% opts.width      = 12;
% opts.height     = 10;
% opts.fontType   = 'Times';