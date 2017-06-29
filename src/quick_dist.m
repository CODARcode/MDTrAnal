%% Set-up
% step size for timestamp
step = 100;
load data/nt-ice_md-normal_new.mat
% reformat data
[x, y, z] = size(trace);
t2 = reshape(trace, [x, y*z]);
%% SVD Analysis
[c,s,l] = svd(t2);

% Visualize the results in 2D
figure, plot(c(:,1), c(:,2))
hold on
for i =1:step:x
    text(c(i,1),c(i,2), int2str(i))
end

% Visualize the results in 3D
figure, plot3(c(:,1), c(:,2), c(:,3))
hold on
for i =1:step:x
    text(c(i,1),c(i,2),c(i,3), int2str(i))
end

