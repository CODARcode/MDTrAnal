%% Set-up
% step size for timestamp
step = 100;
load data/nt-ice_md.mat
% sampling rate 
sampling_rates = [0.001 0.005 0.01 0.05 0.1 0.5];
[x, y, z] = size(trace);
rates = zeros(size(sampling_rates));
tss = cell(size(sampling_rates));
for i=1:size(sampling_rates,2)
    [target, tss{i}] = md_compress(trace, sampling_rates(i));
    cssg = eval_singular_gap(target);
    ussg = eval_singular_gap(trace(int32(1/sampling_rates(i)):int32(1/sampling_rates(i)):x,:,:));
    rates(i) = cssg / ussg;
end 

%% Draw result plot
% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create loglog
loglog(sampling_rates,rates,'Marker','o','LineStyle','-','Color',[1 0 0]);

% Create xlabel
xlabel({'Sampling Rate (log)'});

% Create ylabel
ylabel({'Sum of Singular Value Ratio (log)'});

box(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontSize',12,'XMinorTick','on','XScale','log','YMinorTick','on',...
    'YScale','log');


