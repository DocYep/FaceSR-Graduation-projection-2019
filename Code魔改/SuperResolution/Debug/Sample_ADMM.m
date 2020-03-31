%% Generate problem data

randn('seed', 0);
rand('seed',0);

m = 1500;       % number of examples
n = 5000;       % number of features
p = 100/n;      % sparsity density

x0 = sprandn(n,1,p);                                               %生成n*1的均匀随机分布，分布密度为p的矩阵
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n);               % normalize columns     产生带状稀疏矩阵，将原始矩阵中的元素放置在，nXn的方阵中，的对角线为0的位置上
b = A*x0 + sqrt(0.001)*randn(m,1);

lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;

%% Solve problem

[x history] = ADMM_lasso(A, b, lambda, 1.0, 1.0);


%% Reporting

K = length(history.objval);

h = figure;
plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);       %画两条曲线，objval是啥
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...           %画两条曲线，r和s都是啥，eps的pri和dual是啥
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');
