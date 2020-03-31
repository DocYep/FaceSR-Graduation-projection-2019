%% Generate problem data

randn('seed', 0);
rand('seed',0);

m = 1500;       % number of examples
n = 5000;       % number of features
p = 100/n;      % sparsity density

x0 = sprandn(n,1,p);                                               %����n*1�ľ�������ֲ����ֲ��ܶ�Ϊp�ľ���
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n);               % normalize columns     ������״ϡ����󣬽�ԭʼ�����е�Ԫ�ط����ڣ�nXn�ķ����У��ĶԽ���Ϊ0��λ����
b = A*x0 + sqrt(0.001)*randn(m,1);

lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;

%% Solve problem

[x history] = ADMM_lasso(A, b, lambda, 1.0, 1.0);


%% Reporting

K = length(history.objval);

h = figure;
plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);       %���������ߣ�objval��ɶ
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...           %���������ߣ�r��s����ɶ��eps��pri��dual��ɶ
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');
