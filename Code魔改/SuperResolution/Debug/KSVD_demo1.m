% KSVD running file
% in this file a synthetic test of the K-SVD algorithm is performed. First,
% a random dictionary with normalized columns is being generated, and then
% a set of data signals, each as a linear combination of 3 dictionary
% element is created, with noise level of 20SNR. this set is given as input
% to the K-SVD algorithm.

% a different mode for activating the K-SVD algorithm is until a fixed
% error is reached in the Sparse coding stage, instead until a fixed number of coefficients is found
% (it was used by us for the
% denoising experiments). in order to switch between those two modes just
% change the param.errorFlag (0 - for fixed number of coefficients, 1 -
% until a certain error is reached).

clc,clear;
img1=im2double(imread('E:\毕设\TestHome\1.pgm'));
img2=im2double(imread('E:\毕设\TestHome\2.pgm'));
img3=im2double(imread('E:\毕设\TestHome\3.pgm'));
img4=im2double(imread('E:\毕设\TestHome\4.pgm'));
img5=im2double(imread('E:\毕设\TestHome\5.pgm'));
subplot(2,5,1),imshow(img1);
subplot(2,5,2),imshow(img2);
subplot(2,5,3),imshow(img3);
subplot(2,5,4),imshow(img4);
subplot(2,5,5),imshow(img5);
img=SingleImageDownsample(img4);
subplot(2,5,6),imshow(img);fSize=size(img);%Size
img1=img1(:);img2=img2(:);img3=img3(:);img4=img4(:);img5=img5(:);img=img(:);

imgDictionary=[img1 img2 img3 img4 img5];

%
param.L = 3;   % 稀疏度，非百分比
param.K = 5; % number of dictionary elements  原子个数即字典列数
param.numIteration = 50; % number of iteration to execute the K-SVD algorithm. 迭代次数

param.errorFlag = 0; %  do not fix the number of coefficients.
param.preserveDCAtom = 0;       %字典没有固定的初始列

param.InitializationMethod =  'GivenMatrix';%初始化方法是我们给的字典
param.initialDictionary=imgDictionary;

param.displayProgress = 1;%展示结果
disp('Starting to  train the dictionary');%开始训练字典

[Dictionary,output]  = KSVD(imgDictionary,param);%由一组测试信号数据和参数利用KSVD方法来构造矩阵
x=output.CoefMatrix;

% disp(['The KSVD algorithm retrived ',num2str(output.ratio(end)),' atoms from the original dictionary']);

coefficient=full(output.CoefMatrix)
Out_Img=Dictionary*coefficient;

Residual=imgDictionary-Out_Img;    %Calculate the residual between the data and ans, which is determined by OMP and Dictionary that was gained by KSVD.
SumOfResidual=sum(Residual);            %Gain the sum of the residual. 这个可能不靠谱，应该要用RMSE来看 
 RMSE = sqrt( sum( Residual.^2) / 50 )          %误差
 
Out_Img=reshape(Out_Img,fSize);
Out_Img=im2uint8(Out_Img);
subplot(2,5,8),imshow(Out_Img); 

out_dic=Dictionary(:,4)*coefficient(1,1);
out_dic=reshape(out_dic,fSize);
subplot(2,5,9),imshow(out_dic); 

 
 %%  Origin
param.L = 3;   % number of elements in each linear combination.
param.K = 50; % number of dictionary elements  原子个数即字典列数
param.numIteration = 50; % number of iteration to execute the K-SVD algorithm. 迭代次数

param.errorFlag = 0; % decompose signals until a certain error is reached. do not use fix number of coefficients.
%param.errorGoal = sigma;
param.preserveDCAtom = 0;

%%%%%%% creating the data to train on %%%%%%%%
N = 1500; % number of signals to generate 要产生的信号的个数
n = 20;   % dimension of each data 信号的维数
SNRdB = 20; % level of noise to be added 将信噪比为20的噪声添加到信号中
[param.TrueDictionary, D, x] = gererateSyntheticDictionaryAndData(N, param.L, n, param.K, SNRdB);%调用函数产生合成字典和数据，输出原始的字典和数据，及系数
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% initial dictionary: Dictionary elements %%%%%%%%
param.InitializationMethod =  'DataElements';%初始化方法是数据元素

param.displayProgress = 1;%展示结果
disp('Starting to  train the dictionary');%开始训练字典

[Dictionary,output]  = KSVD(D,param);%由一组测试信号数据和参数利用KSVD方法来构造矩阵

disp(['The KSVD algorithm retrived ',num2str(output.ratio(end)),' atoms from the original dictionary']);

[Dictionary,output]  = MOD(D,param);%由一组测试信号数据和参数利用MOD方法来构造矩阵

disp(['The MOD algorithm retrived ',num2str(output.ratio(end)),' atoms from the original dictionary']);


Residual=D-Dictionary*x;    %Calculate the residual between the data and ans, which is determined by OMP and Dictionary that was gained by KSVD.
SumOfResidual=sum(Residual);            %Gain the sum of the residual. 这个可能不靠谱，应该要用RMSE来看 
 RMSE = sqrt( sum( Residual.^2) / 50 ); 
