%This function is written as an algorithnm for looking for an satifying
%answer about Sparse Representation each patch.

%Attention!    DOUBLE format of the image , not UINT8.

function SinglePatch_SparseRepresentation(fID)  %fID is a integer, fIteration is the num of iteration
    
    global gIter;       %迭代指针
    global gOutput_Path;         %输出HR图像位置
    global gSRArray;                %单次patch 稀疏表示系数数组
    global gAtomNum;%计划训练好的原子数目
    
    %参数结果
    global gRowPatches;                    %  切成几行patch，实际的具体数，补全后的块数行方向
    global gColumnPatches;             %  切成几列patch，实际的具体数，补全后的块数列方向

    global gInputPatch;         %输入LR图像块集
    global gInputFeaPatch;  %输入LR特征图像块集
    global gLRDictionaryPatch;         %LR字典块集
    global gHRDictionaryPatch;         %HR字典块集
    
    %用以增强约束
    global gLRPatchSize;    %LR块的像素大小
    global gHRPatchSize;    %HR块的像素大小
    global gLROverlap;            %LR块的冗余像素个数
    global gHROverlap;            %HR块的冗余像素个数
    global gLRResidual;              %LR块大小与LR Overlap之差
    global gHRResidual;              %HR块大小与HR Overlap之差
    global gSRPatch;    %重构出的SR图像块
    global gLastAns;                %上次迭代输出的结果
    
    global Dh;
    global Dl;
    global gAtomNum;%计划训练好的原子数目
    
    disp( ['Now the ',num2str(fID),' is SparseRepresentation!'] );
    %%  输入图像patch
    fInputPatch=im2double( gInputFeaPatch{fID} );
%     gHRPatchSize=sqrt(size( Dh,1 ));      %按HR块大小复原，按已补齐复原
    fBicubic=imresize(im2double( gInputPatch{fID} ), [gHRPatchSize gHRPatchSize], 'bicubic');%放大的LR图
    fInputPatch=fInputPatch(:);

    %% 迭代约束  
    %未加约束之前的:Way: NO
    A=Dl;  
    b=fInputPatch(:);
    
     if gIter~=1  %首次迭代无约束        
         fImg=im2double( gLastAns );       %上次迭代的输出
         fRow=floor( (fID-1)/gColumnPatches );
         fColumn=mod( (fID-1),gColumnPatches );
         
         fTop=1+fRow*gHRResidual;       %迭代约束是依据上次输出的HR图像
         fLeft=1+fColumn*gHRResidual;
         fImg=fImg( fTop:fTop+gHRPatchSize-1,fLeft:fLeft+gHRPatchSize-1 );
         fImg=fImg(:);
         
         fGama=1;       %迭代约束因子
         A=[ A;fGama*Dh ];
         b=[ b;fGama*fImg ];
     end
    
    %% 增加约束内容——领域约束:Delete法
    %原来的思路是，去掉区域的HR字典乘系数，与冗余区域差距最小
    if fID~=1  %首块不加约束
        if mod((fID-1),gColumnPatches)      %左边判断条件——非第一列
            fOverlapLeftHR_SR=im2double( gSRPatch{fID-1} );       
            fOverlapLeftHR_SR=fOverlapLeftHR_SR(:,end-gHROverlap+1:end);  %重构的图片是HR规格的
        end
        if fID>gColumnPatches      %上边判断条件——非第一行
            fOverlapTopHR_SR=im2double( gSRPatch{fID-gColumnPatches} );       
            fOverlapTopHR_SR=fOverlapTopHR_SR(end-gHROverlap+1:end,:);  %重构的图片是HR规格的
        end
        %左侧是1，上侧是2
        fHR_A1=[];               %这是左侧对应位置patch的HR字典
        fHR_A2=[];               %这是上侧对应位置patch的HR字典
        for fFaceID=1:gAtomNum
            fImg=im2double( Dh(:,fFaceID) );%图片必须要转换成实数，来自百度,后来就可以健健康康不怕溢出了，不要变成N*1列
            fImg=reshape(fImg,[gHRPatchSize gHRPatchSize]);%因为是训练好的所以要转回矩阵
            
            if mod((fID-1),gColumnPatches)      %左边
                fImg1=fImg(:,1:gHROverlap);
                fImg1=fImg1(:);   %此处图片才化为单列
                fHR_A1(:,fFaceID)=fImg1;
            end
            if fID>gColumnPatches                   %上侧
                fImg2=fImg(1:gHROverlap,:);
                fImg2=fImg2(:);   %此处图片才化为单列
                fHR_A2(:,fFaceID)=fImg2;
            end
        end
        
        if mod((fID-1),gColumnPatches)      %左边
            fBeta1=1;      %fBeta是平衡权重
            A=[ A;fBeta1*fHR_A1;]; 
            b=[ b;fBeta1*fOverlapLeftHR_SR(:) ]; 
        end
        if fID>gColumnPatches                   %上侧
            fBeta2=1;
            A=[ A;fBeta2*fHR_A2;]; 
            b=[ b;fBeta2*fOverlapTopHR_SR(:) ]; 
        end
    end
    
%     %不加约束
%     A=Dl;  
%     b=fInputPatch(:);
    
    %% Normalize，A需要归一化
    %对A进行归一化操作，存疑，因为加了约束以后的归一化是之后的A进行归一化而不仅仅是对LR，重构的时候这个系数不知道能不能直接用HR归一化的上
    %选择恢复的方式，1是按规格化的比例，0是按模比值
    type=0;%0的按模比不能用，因为上面的fBicubic不是LR本身而是导数
    
    if type
        B=A; 
        A = normalize(A,'norm',2);
        fZoom=A(1,:)./B(1,:);%计算LR字典规格化前后各列缩小变化的倍数,勿忘./
    else
        fLRNorm=sqrt(sum(fBicubic.^2));
        A = normalize(A,'norm',2);
    end

    %% 稀疏表示算法——三选一
    %选方法吧~~~
    choose=4;
    lambda=0.15;%瞎取的
    switch(choose)
        case 1
            % 1、OMP算法
            x=OMP(A,b,floor(lambda*gAtomNum));        %输入输出全部是double
            x=full(x); 
            x=x(:,1);
            
        case 2
            % 2、MyADMM算法
            [x history] = ADMM_lasso(A,b, lambda, 1.0, 1.0); %没有规格化，试试看先
    
        case 3
            % 3、学长ADMM L1算法，PSNR28.56
            [x,obj,err,iter] = l1(A,b);%有默认的obt，所以我就不加了
           
        case 4
            %4、Yang的算法，但是他的方法必须要结合：法0的复原方式
            %不得不服，效果要好欸,还贼他妈快
            %PSNR 27.17 type=1/ type=0 28.72
            D = A'*A;
            b = -A'*b;
            x= L1QP_FeatureSign_yang(lambda,D,b);
    end
    %都要做的事
    if type
        gSRArray=x.*fZoom';%使得系数平衡,勿忘.*
    else
        gSRArray=x;
    end

%%  SinglePatch_SR_Recovery     图像重构
    fOutputImg=gSRArray'.*Dh;           %复原图
    fOutputImg=sum(fOutputImg')';       %还要叠加在一起
    fOutputImg=reshape(fOutputImg,[gHRPatchSize gHRPatchSize]);     %将复原的图片变回正常的形状 ，修改过未确认！！！！！！！！！！！【】【】【】】
    if ~type
        fHRNorm=sqrt(sum(fOutputImg.^2));
        fZoom=1.0*fLRNorm./fHRNorm;
        fOutputImg=fOutputImg.*fZoom;%原来是1.2，矩阵点除别忘了
    end
    
    fOutputImg=im2uint8(fOutputImg);        %转换回来原来的uint8格式
    gSRPatch{fID}=fOutputImg;

end