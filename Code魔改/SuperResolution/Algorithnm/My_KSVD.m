function  [fHR_A fLR_A]=My_KSVD(fData,fHRRows,fIteration)

    global gAtomNum;%计划训练好的原子数目
    
    param.L = floor(0.2*gAtomNum);   % 稀疏度，非百分比
    param.K = gAtomNum; % number of dictionary elements  原子个数即字典列数
    param.numIteration = fIteration; % number of iteration to execute the K-SVD algorithm. 迭代次数

    param.errorFlag = 0; %  do not fix the number of coefficients.
    param.preserveDCAtom = 0;       %字典没有固定的初始列

    param.InitializationMethod =  'DataElements';%初始化方法是我们给的字典
%     param.initialDictionary=fInputDictionary;

    param.displayProgress = 0;%展示结果
    disp('Starting to  train the dictionary');%开始训练字典

    [fOutputDictionary output]  = KSVD(fData,param);%由一组测试信号数据和参数利用KSVD方法来构造矩阵
%     x=output.CoefMatrix;
    [row column]=size(fOutputDictionary);

    fHR_A=fOutputDictionary(1:fHRRows,:);
    fLR_A=fOutputDictionary(fHRRows+1:end,:);
    % disp(['The KSVD algorithm retrived ',num2str(output.ratio(end)),' atoms from the original dictionary']);

%     coefficient=full(output.CoefMatrix)
%     
%     Out_Img=fOutputDictionary*coefficient;
% 
%     Residual=imgDictionary-Out_Img;    %Calculate the residual between the data and ans, which is determined by OMP and Dictionary that was gained by KSVD.
%     SumOfResidual=sum(Residual);            %Gain the sum of the residual. 这个可能不靠谱，应该要用RMSE来看 
%      RMSE = sqrt( sum( Residual.^2) / 50 )          %误差
% 
%     Out_Img=reshape(Out_Img,fSize);
%     Out_Img=im2uint8(Out_Img);
%     subplot(2,5,8),imshow(Out_Img); 
% 
%     out_dic=Dictionary(:,4)*coefficient(1,1);
%     out_dic=reshape(out_dic,fSize);
%     subplot(2,5,9),imshow(out_dic); 


end 