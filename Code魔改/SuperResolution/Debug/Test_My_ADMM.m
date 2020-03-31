%% Generate problem data
clc;clear;

randn('seed', 0);
rand('seed',0);

img1=imread('E:\����\Code\Image\Face_ORL\HR_Dictionary\1.pgm');
imgSize=size(img1);
img2=imread('E:\����\Code\Image\Face_ORL\HR_Dictionary\2.pgm');
img3=imread('E:\����\Code\Image\Face_ORL\HR_Dictionary\3.pgm');
img4=imread('E:\����\Code\Image\Face_ORL\HR_Dictionary\4.pgm');
img1=im2double(img1(:));
img2=im2double(img2(:));
img3=im2double(img3(:));
img4=im2double(img4(:));
A=[img1 img2 img3 img4];
b=img1;
 

n=4;
p=0.01;
m=length(b);
% n = 5000;       % number of features
% p = 100/n;      % sparsity density
% m = 1500;       % number of examples
 

x0 = sprandn(n,1,p);                                               %����n*1�ľ�������ֲ����ֲ��ܶ�Ϊp�ľ���
% A=[1 0 0;0 1 0;0 0 1];
% b=[1;1;1];
% A = randn(m,n);
% A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n);               % normalize columns     ������״ϡ����󣬽�ԭʼ�����е�Ԫ�ط����ڣ�nXn�ķ����У��ĶԽ���Ϊ0��λ����
% b = A*x0 + sqrt(0.001)*randn(m,1);

lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;

%% Solve problem

[x history] = ADMM_lasso(A, b, lambda, 1.0, 1.0);


newImg=x'.*A;           %��ԭͼƬ����ϡ�������ֵ�
newImg=sum(newImg')';       %��Ҫ������һ��
newImg=reshape(newImg,imgSize);     %����ԭ��ͼƬ�����������״
imshow(newImg);     %OJBK

% %% Reporting
% 
% K = length(history.objval);
% 
% h = figure;
% plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);       %���������ߣ�objval��ɶ
% ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');
% 
% g = figure;
% subplot(2,1,1);
% semilogy(1:K, max(1e-8, history.r_norm), 'k', ...           %���������ߣ�r��s����ɶ��eps��pri��dual��ɶ
%     1:K, history.eps_pri, 'k--',  'LineWidth', 2);
% ylabel('||r||_2');
% 
% subplot(2,1,2);
% semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
%     1:K, history.eps_dual, 'k--', 'LineWidth', 2);
% ylabel('||s||_2'); xlabel('iter (k)');
