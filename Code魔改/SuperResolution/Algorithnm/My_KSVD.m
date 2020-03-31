function  [fHR_A fLR_A]=My_KSVD(fData,fHRRows,fIteration)

    global gAtomNum;%�ƻ�ѵ���õ�ԭ����Ŀ
    
    param.L = floor(0.2*gAtomNum);   % ϡ��ȣ��ǰٷֱ�
    param.K = gAtomNum; % number of dictionary elements  ԭ�Ӹ������ֵ�����
    param.numIteration = fIteration; % number of iteration to execute the K-SVD algorithm. ��������

    param.errorFlag = 0; %  do not fix the number of coefficients.
    param.preserveDCAtom = 0;       %�ֵ�û�й̶��ĳ�ʼ��

    param.InitializationMethod =  'DataElements';%��ʼ�����������Ǹ����ֵ�
%     param.initialDictionary=fInputDictionary;

    param.displayProgress = 0;%չʾ���
    disp('Starting to  train the dictionary');%��ʼѵ���ֵ�

    [fOutputDictionary output]  = KSVD(fData,param);%��һ������ź����ݺͲ�������KSVD�������������
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
%     SumOfResidual=sum(Residual);            %Gain the sum of the residual. ������ܲ����ף�Ӧ��Ҫ��RMSE���� 
%      RMSE = sqrt( sum( Residual.^2) / 50 )          %���
% 
%     Out_Img=reshape(Out_Img,fSize);
%     Out_Img=im2uint8(Out_Img);
%     subplot(2,5,8),imshow(Out_Img); 
% 
%     out_dic=Dictionary(:,4)*coefficient(1,1);
%     out_dic=reshape(out_dic,fSize);
%     subplot(2,5,9),imshow(out_dic); 


end 