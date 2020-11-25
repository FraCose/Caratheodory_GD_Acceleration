% Create the the Dataset A of https://github.com/IssamLaradji/BlockCoordinateDescent
% It is used in 
% Cosentino, Oberhauser, Abate - "Acceleration of Descent-based Optimization Algorithms via Carathéodory's Theorem" 
rng(1)
bias = 1; scaling = 10; sparsity = 10; solutionSparsity = 0.1;
n = 1000000;
p = 500;
X = randn(n,p)+bias;
X = X*diag(scaling*randn(p,1));
X = X.*(rand(n,p)<sparsity*log(n)/n);
w = randn(p,1).*(rand(p,1)<solutionSparsity);
y = X*w + randn(n,1);
save('exp4.mat','X','y','-v7.3');