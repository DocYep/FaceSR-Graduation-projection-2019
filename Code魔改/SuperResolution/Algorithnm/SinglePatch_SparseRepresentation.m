%This function is written as an algorithnm for looking for an satifying
%answer about Sparse Representation each patch.

%Attention!    DOUBLE format of the image , not UINT8.

function SinglePatch_SparseRepresentation(fID)  %fID is a integer, fIteration is the num of iteration
    
    global gIter;       %����ָ��
    global gOutput_Path;         %���HRͼ��λ��
    global gSRArray;                %����patch ϡ���ʾϵ������
    global gAtomNum;%�ƻ�ѵ���õ�ԭ����Ŀ
    
    %�������
    global gRowPatches;                    %  �гɼ���patch��ʵ�ʵľ���������ȫ��Ŀ����з���
    global gColumnPatches;             %  �гɼ���patch��ʵ�ʵľ���������ȫ��Ŀ����з���

    global gInputPatch;         %����LRͼ��鼯
    global gInputFeaPatch;  %����LR����ͼ��鼯
    global gLRDictionaryPatch;         %LR�ֵ�鼯
    global gHRDictionaryPatch;         %HR�ֵ�鼯
    
    %������ǿԼ��
    global gLRPatchSize;    %LR������ش�С
    global gHRPatchSize;    %HR������ش�С
    global gLROverlap;            %LR����������ظ���
    global gHROverlap;            %HR����������ظ���
    global gLRResidual;              %LR���С��LR Overlap֮��
    global gHRResidual;              %HR���С��HR Overlap֮��
    global gSRPatch;    %�ع�����SRͼ���
    global gLastAns;                %�ϴε�������Ľ��
    
    global Dh;
    global Dl;
    global gAtomNum;%�ƻ�ѵ���õ�ԭ����Ŀ
    
    disp( ['Now the ',num2str(fID),' is SparseRepresentation!'] );
    %%  ����ͼ��patch
    fInputPatch=im2double( gInputFeaPatch{fID} );
%     gHRPatchSize=sqrt(size( Dh,1 ));      %��HR���С��ԭ�����Ѳ��븴ԭ
    fBicubic=imresize(im2double( gInputPatch{fID} ), [gHRPatchSize gHRPatchSize], 'bicubic');%�Ŵ��LRͼ
    fInputPatch=fInputPatch(:);

    %% ����Լ��  
    %δ��Լ��֮ǰ��:Way: NO
    A=Dl;  
    b=fInputPatch(:);
    
     if gIter~=1  %�״ε�����Լ��        
         fImg=im2double( gLastAns );       %�ϴε��������
         fRow=floor( (fID-1)/gColumnPatches );
         fColumn=mod( (fID-1),gColumnPatches );
         
         fTop=1+fRow*gHRResidual;       %����Լ���������ϴ������HRͼ��
         fLeft=1+fColumn*gHRResidual;
         fImg=fImg( fTop:fTop+gHRPatchSize-1,fLeft:fLeft+gHRPatchSize-1 );
         fImg=fImg(:);
         
         fGama=1;       %����Լ������
         A=[ A;fGama*Dh ];
         b=[ b;fGama*fImg ];
     end
    
    %% ����Լ�����ݡ�������Լ��:Delete��
    %ԭ����˼·�ǣ�ȥ�������HR�ֵ��ϵ������������������С
    if fID~=1  %�׿鲻��Լ��
        if mod((fID-1),gColumnPatches)      %����ж����������ǵ�һ��
            fOverlapLeftHR_SR=im2double( gSRPatch{fID-1} );       
            fOverlapLeftHR_SR=fOverlapLeftHR_SR(:,end-gHROverlap+1:end);  %�ع���ͼƬ��HR����
        end
        if fID>gColumnPatches      %�ϱ��ж����������ǵ�һ��
            fOverlapTopHR_SR=im2double( gSRPatch{fID-gColumnPatches} );       
            fOverlapTopHR_SR=fOverlapTopHR_SR(end-gHROverlap+1:end,:);  %�ع���ͼƬ��HR����
        end
        %�����1���ϲ���2
        fHR_A1=[];               %��������Ӧλ��patch��HR�ֵ�
        fHR_A2=[];               %�����ϲ��Ӧλ��patch��HR�ֵ�
        for fFaceID=1:gAtomNum
            fImg=im2double( Dh(:,fFaceID) );%ͼƬ����Ҫת����ʵ�������԰ٶ�,�����Ϳ��Խ���������������ˣ���Ҫ���N*1��
            fImg=reshape(fImg,[gHRPatchSize gHRPatchSize]);%��Ϊ��ѵ���õ�����Ҫת�ؾ���
            
            if mod((fID-1),gColumnPatches)      %���
                fImg1=fImg(:,1:gHROverlap);
                fImg1=fImg1(:);   %�˴�ͼƬ�Ż�Ϊ����
                fHR_A1(:,fFaceID)=fImg1;
            end
            if fID>gColumnPatches                   %�ϲ�
                fImg2=fImg(1:gHROverlap,:);
                fImg2=fImg2(:);   %�˴�ͼƬ�Ż�Ϊ����
                fHR_A2(:,fFaceID)=fImg2;
            end
        end
        
        if mod((fID-1),gColumnPatches)      %���
            fBeta1=1;      %fBeta��ƽ��Ȩ��
            A=[ A;fBeta1*fHR_A1;]; 
            b=[ b;fBeta1*fOverlapLeftHR_SR(:) ]; 
        end
        if fID>gColumnPatches                   %�ϲ�
            fBeta2=1;
            A=[ A;fBeta2*fHR_A2;]; 
            b=[ b;fBeta2*fOverlapTopHR_SR(:) ]; 
        end
    end
    
%     %����Լ��
%     A=Dl;  
%     b=fInputPatch(:);
    
    %% Normalize��A��Ҫ��һ��
    %��A���й�һ�����������ɣ���Ϊ����Լ���Ժ�Ĺ�һ����֮���A���й�һ�����������Ƕ�LR���ع���ʱ�����ϵ����֪���ܲ���ֱ����HR��һ������
    %ѡ��ָ��ķ�ʽ��1�ǰ���񻯵ı�����0�ǰ�ģ��ֵ
    type=0;%0�İ�ģ�Ȳ����ã���Ϊ�����fBicubic����LR������ǵ���
    
    if type
        B=A; 
        A = normalize(A,'norm',2);
        fZoom=A(1,:)./B(1,:);%����LR�ֵ���ǰ�������С�仯�ı���,����./
    else
        fLRNorm=sqrt(sum(fBicubic.^2));
        A = normalize(A,'norm',2);
    end

    %% ϡ���ʾ�㷨������ѡһ
    %ѡ������~~~
    choose=4;
    lambda=0.15;%Ϲȡ��
    switch(choose)
        case 1
            % 1��OMP�㷨
            x=OMP(A,b,floor(lambda*gAtomNum));        %�������ȫ����double
            x=full(x); 
            x=x(:,1);
            
        case 2
            % 2��MyADMM�㷨
            [x history] = ADMM_lasso(A,b, lambda, 1.0, 1.0); %û�й�񻯣����Կ���
    
        case 3
            % 3��ѧ��ADMM L1�㷨��PSNR28.56
            [x,obj,err,iter] = l1(A,b);%��Ĭ�ϵ�obt�������ҾͲ�����
           
        case 4
            %4��Yang���㷨���������ķ�������Ҫ��ϣ���0�ĸ�ԭ��ʽ
            %���ò�����Ч��Ҫ�ÚG,���������
            %PSNR 27.17 type=1/ type=0 28.72
            D = A'*A;
            b = -A'*b;
            x= L1QP_FeatureSign_yang(lambda,D,b);
    end
    %��Ҫ������
    if type
        gSRArray=x.*fZoom';%ʹ��ϵ��ƽ��,����.*
    else
        gSRArray=x;
    end

%%  SinglePatch_SR_Recovery     ͼ���ع�
    fOutputImg=gSRArray'.*Dh;           %��ԭͼ
    fOutputImg=sum(fOutputImg')';       %��Ҫ������һ��
    fOutputImg=reshape(fOutputImg,[gHRPatchSize gHRPatchSize]);     %����ԭ��ͼƬ�����������״ ���޸Ĺ�δȷ�ϣ�����������������������������������
    if ~type
        fHRNorm=sqrt(sum(fOutputImg.^2));
        fZoom=1.0*fLRNorm./fHRNorm;
        fOutputImg=fOutputImg.*fZoom;%ԭ����1.2��������������
    end
    
    fOutputImg=im2uint8(fOutputImg);        %ת������ԭ����uint8��ʽ
    gSRPatch{fID}=fOutputImg;

end