clc;

%% 画向量图实验
 

%% 规格化速度测评——还是第一个快啊。。。
load('MyDictionary35-1.mat'); %我自己训练的
A=Dl;
tic;
A = normalize(A,'norm',2);
toc;

tic;
A = A./repmat(sqrt(sum(A.^2)), size(A, 1), 1);
toc;

%% 图像GOG提取测试
clear,clc;

f = 1; % 1 -- GOG_RGB, 2 -- GOG_Lab, 3 -- GOG_HSV, 4 -- GOG_nRnG
param = set_default_parameter(f); 

I = im2double(imread('E:\毕设\me.png')); % load image 
feature_vec = GOG(I, param); 
 % Note that output of the feature vector is not normalized.


%% 图像二范数作为距离度量器官 测试
path='E:\毕设\Code魔改\Image\Face_ORL\Origin\Testing10-1-5-2';
patchSize=10;
list=ls(path);
len=length(list);
imgList=[];
feaList=[];
num=0;
for id=3:len
    %读入加分块
    img=im2double(imread([path '\' list(id,:)]));
    imgSize=size(img);
    for i=1:patchSize:imgSize(1)-patchSize
        for j=1:patchSize:imgSize(2)-patchSize
            patch=img(i:i+patchSize-1,j:j+patchSize-1);
            num=num+1;
            patch=patch(:);
            imgList(:,num)=patch;
            feaList(:,num)=SingleImageFeatureExtracting(patch);%特征集
        end
    end
end
img=im2double(imread('E:\毕设\Code魔改\Image\Face_ORL\Input&Output\LR.pgm'));
patchSize=10;
topBegin=23;%X坐标，也就是水平分量
leftBegin=49;%Y坐标，也就是垂直分量

% imshow(img,'InitialMagnification','fit');
% impixelinfo;%显示像素位置
% rect = [leftBegin topBegin patchSize patchSize];
% rectangle('Position',rect,'LineWidth',4,'EdgeColor','r');
%%
patch=img(leftBegin:leftBegin-1+patchSize,topBegin:topBegin-1+patchSize);imshow(patch,'InitialMagnification','fit');
patch=patch(:);
%直接像素二范数
[a b1]=sort(sum((patch-imgList).^2));
%特征提取二范数
[a b2]=sort(sum((SingleImageFeatureExtracting(patch)-feaList).^2));

subplot(2,10,1),imshow(reshape(patch,[patchSize patchSize])),title('Origin');     %将复原的图片变回正常的形状
subplot(2,10,7),imshow(reshape(patch,[patchSize patchSize])),title('Origin');     %将复原的图片变回正常的形状
%显示环节
for i=1:10
    patch=reshape(imgList(:,b1(i)),[patchSize patchSize]);
    subplot(2,10,i+1),imshow(reshape(patch,[patchSize patchSize]));     %将复原的图片变回正常的形状——像素二范数
    patch=reshape(imgList(:,b2(i)),[patchSize patchSize]);
    subplot(2,10,i+1+10),imshow(reshape(patch,[patchSize patchSize]));     %将复原的图片变回正常的形状——特征二范数
end

%% 算法精度误差原因探究——稀疏表示算法还是字典问题
    clc;clear;warning off;
    global gGap;
    gGap=2;
    
    HR=imread('E:\毕设\Code魔改\Image\Face_ORL\Input&Output\LR.pgm');
    LR=SingleImageDownsample(HR);
    patchSize=10;
    leftBegin=49;%Y坐标，也就是垂直分量
    topBegin=23;%X坐标，也就是水平分量
    lambda=0.2;%瞎取的
    
    %画出原图，标记块位置
    subplot(1,2,1),imshow(HR,'InitialMagnification','fit');
    rect = [topBegin leftBegin patchSize patchSize];
    rectangle('Position',rect,'LineWidth',4,'EdgeColor','r');
    HRpatch=HR(leftBegin:leftBegin-1+patchSize,topBegin:topBegin-1+patchSize);
    patchSize=patchSize/2;
    leftBegin=leftBegin/2;
    topBegin=topBegin/2;
    subplot(1,2,2),imshow(LR,'InitialMagnification','fit');
    rect = [topBegin leftBegin patchSize patchSize];
    rectangle('Position',rect,'LineWidth',2,'EdgeColor','r');
    figure;
    LRpatch=LR(leftBegin:leftBegin-1+patchSize,topBegin:topBegin-1+patchSize);
    
    % 字典
    load('MyDictionary10-1.mat'); %我自己训练的，好一点
%     load('MyDictionary35-1.mat'); %我自己训练的，不能忍
    fImgSize=sqrt(size( Dh,1 ));      %按HR块大小复原，按已补齐复原
    fImgSize=[fImgSize fImgSize];
    gAtomNum=1024;%计划训练好的原子数目
    A=Dl;  
    b=im2double( LRpatch(:) );
    
    fBicubic=imresize(LRpatch, fImgSize, 'bicubic');%放大的LR图
    fBicubic=im2double(fBicubic);
    
    %选择恢复的方式，1是按规格化的比例，0是按模比值
    type=0;
    
    if type
        B=A; 
        A = normalize(A,'norm',2);
        fZoom=A(1,:)./B(1,:);%计算LR字典规格化前后各列缩小变化的倍数,勿忘./
    else
        fLRNorm=sqrt(sum(fBicubic.^2));
        A = normalize(A,'norm',2);
    end
    
    %SR——OMP,            0.004s
    x=OMP(A,b,floor(0.15*gAtomNum));        %输入输出全部是double
    x=full(x); 
    x=x(:,1);
    if type
        x=x.*fZoom';%使得系数平衡,勿忘.*        
    end
    fOMPOut=x'.*Dh;           %复原图
    fOMPOut=sum(fOMPOut')';       %还要叠加在一起
    fOMPOut=reshape(fOMPOut,fImgSize);     %将复原的图片变回正常的形状
    if ~type
        fHRNorm=sqrt(sum(fOMPOut.^2));
        fOMPOut=fOMPOut.*(fLRNorm*1.2/fHRNorm);
    end

    %SR——my-ADMM,   0.013s
    %不规格化的话平滑很多但是比较暗
    [x history] = ADMM_lasso(A,b, lambda, 1.0, 1.0); %没有规格化，试试看先
    if type
        x=x.*fZoom';%使得系数平衡,勿忘.*
    end
    fMyADMMOut=x'.*Dh;           %复原图
    fMyADMMOut=sum(fMyADMMOut')';       %还要叠加在一起
    fMyADMMOut=reshape(fMyADMMOut,fImgSize);     %将复原的图片变回正常的形状
    if ~type
        fHRNorm=sqrt(sum(fMyADMMOut.^2));
        fMyADMMOut=fMyADMMOut.*(fLRNorm*1.2/fHRNorm);
    end

    %SR——学长-ADMM,   0.187
    [x,obj,err,iter] = l1(A,b);%有默认的obt，所以我就不加了
    if type
        x=x.*fZoom';%使得系数平衡,勿忘.*
    end
    fL1ADMMOut=x'.*Dh;           %复原图
    fL1ADMMOut=sum(fL1ADMMOut')';       %还要叠加在一起
    fL1ADMMOut=reshape(fL1ADMMOut,fImgSize);     %将复原的图片变回正常的形状
    if ~type
        fHRNorm=sqrt(sum(fL1ADMMOut.^2));
        fL1ADMMOut=fL1ADMMOut.*(fLRNorm*1.2/fHRNorm);
    end
    
    %Yang
    A = Dl'*Dl;
    b = -Dl'*b;
    x= L1QP_FeatureSign_yang(lambda,A,b);
    if type
        x=x.*fZoom';%使得系数平衡,勿忘.*
    end
    fYangOut=x'.*Dh;           %复原图
    fYangOut=sum(fYangOut')';       %还要叠加在一起
    fYangOut=reshape(fYangOut,fImgSize);     %将复原的图片变回正常的形状
    if ~type
        fHRNorm=sqrt(sum(fYangOut.^2));
        fYangOut=fYangOut.*(fLRNorm*1.2/fHRNorm);%(fHRNorm*1.2/fLRNorm);
    end
    
    %Output
    subplot(2,4,1),imshow(HRpatch),title('HR');
    subplot(2,4,2),imshow(LRpatch),title('LR');
    subplot(2,4,3),imshow(fBicubic),title('bicubic');
    
    subplot(2,4,5),imshow(fOMPOut),title('OMP-SR');
    subplot(2,4,6),imshow(fMyADMMOut),title('MyADMM-SR');
    subplot(2,4,7),imshow(fL1ADMMOut),title('L1ADMMSR');
    subplot(2,4,8),imshow(fYangOut),title('Yang');
%     figure;

%% tic toc 实验
    tic;
    for i=1:100000
        disp(6);
    end
    toc;
    
%% 读取本地字典库
    load('D_1024_0.15_5.mat');

%% paper图像
    img0=imread('E:\毕设\Code\Image\Face_ORL\Origin\Paper\t1.bmp');
    size(img0)
    subplot(1,3,1),imshow(img0);
    img0=img0(:,:,1);
    img1=imresize(img0,0.5,'bicubic');                                     %将输入图像插值缩放至字典大小
    subplot(1,3,2),imshow(img1);
    
    
    img=img0(:,:,1);
[row,column]=size(img);%将图像隔行隔列抽取元素，得到缩小的图像f
m=floor(row/2);n=floor(column/2);
f=zeros(m,n);
for i=1:m
    for j=1:n
        f(i,j)=img(2*i,2*j);
    end
end
subplot(1,3,3),imshow(uint8(f));
title('缩小的图像');%显示缩小的图像

        
%% 字典学习测试
        %验证集的相关参数
    global gTestingNum;
    global gTestingHRImage;   %验证集HR图像内容
    global gTestingLRImage;   %验证集LR图像内容
    global gTestingHRPatch;            %测试集块路径
    global gTestingLRPatch;            %测试集块路径
    
    % 取出对应位置的HR测试集块
    fList=ls( [gTestingHRPatch_Path,'\',num2str(fID)] );
    fLen=length(fList);
    gTestingHR=[];                           %这是对应位置patch的TestingHR
    for i=3:fLen
        fImg=imread( [gTestingHRPatch_Path,'\',num2str(fID),'\',fList(i,:)] );
        fImg=im2double(fImg);    %图片必须要转换成实数，来自百度,后来就可以健健康康不怕溢出了，不要变成N*1列
        fImg=fImg(:);   %此处图片才化为单列
        gTestingHR=[gTestingHR fImg];     %即为TestingHR
    end
    % 取出对应位置的LR测试集块
    fList=ls( [gTestingLRPatch_Path,'\',num2str(fID)] );
    fLen=length(fList);
    gTestingLR=[];                           %这是对应位置patch的TestingLR
    for i=3:fLen
        fImg=imread( [gTestingLRPatch_Path,'\',num2str(fID),'\',fList(i,:)] );
        fImg=im2double(fImg);    %图片必须要转换成实数，来自百度,后来就可以健健康康不怕溢出了，不要变成N*1列
        fImg=fImg(:);   %此处图片才化为单列
        gTestingLR=[gTestingLR fImg];     %即为TestingLR
    end
    fTestingData=[gTestingHR;gTestingLR]; %HR上LR下
    fDictionary=[fHR_A1;fLR_A];%HR上LR下
    [fHR_A1 fLR_A]=My_KSVD(fTestingData,fDictionary); %得到更新以后的字典
    %HR_A进行归一化操作
%     fHR_A = normalize(fHR_A,'norm',2);      %OMP在稀疏表示时系数是按照归一化求的,那么复原重构的时候字典也要归一化！！！




%%  Debug
    clc;
    fColumn=31;
    fResidual=4;
    fPatchSize=10;
    
    fColumnRes=mod(fResidual-(fColumn-fPatchSize),fResidual);     %列不足
    mod(fColumnRes,fResidual)
    
    fNew_Column=fColumn+fColumnRes
    fColumnPatches=(fNew_Column-fPatchSize)/fResidual+1         %一行的块数,切成几列patch，实际的具体数

%% 分块提特征会怎么样
    figure();
    img=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\LR.pgm');
%     img=img(1:end/2,1:end/2);
    [row column]=size(img);

    filter=[-1,0,1];       %一阶导数    行向量则列方向特征明显，列向量则行方向特征明显；LR特征均无
%     filter=[1,0,-2,0,1];       %二阶导数    同上。LR一点点，几乎没有
%     filter=[0,-1,0;-1,4,-1;0,-1,0];       %二阶微分，方向无变化，内容一致，无法规避椒盐
    subplot(3,3,1),imshow(img),title('HR图像');
    new_img=imfilter(img,filter);     subplot(3,3,2),imshow(new_img),title('HR一阶导数-行');filter=filter';
    new_img=imfilter(img,filter);     subplot(3,3,3),imshow(new_img),title('HR一阶导数-列');filter=filter';
    
    %下采样
    img1=img(1:3:row,:);%隔两行采样
    img1=img1(:,1:3:column);%隔两列采样
    subplot(3,3,4),imshow(img1),title('LR原图');
    new_img=imfilter(img1,filter);     subplot(3,3,5),imshow(new_img),title('LR一阶导数-行');filter=filter';
    new_img=imfilter(img1,filter);     subplot(3,3,6),imshow(new_img),title('LR一阶导数-列');filter=filter';
    
    img1=imnoise(img1,'salt & pepper',0.02);                        %加入椒盐躁声，LR字典是不需要椒盐噪声的
    subplot(3,3,7),imshow(img1),title('LR图像');
    new_img=imfilter(img1,filter);     subplot(3,3,8),imshow(new_img),title('LR一阶导数-行');filter=filter';
    new_img=imfilter(img1,filter);     subplot(3,3,9),imshow(new_img),title('LR一阶导数-列');filter=filter';

%% 图像数据库的移动
clc;clear;
    %自己把原来的删掉，然后修改一下下面的路径
    RootPath='E:\毕设\人脸数据库【原】\ORL_Face';
    Training_Path='E:\毕设\Code魔改\Image\Face_ORL\Origin\Training10-1';
    Testing_Path='E:\毕设\Code魔改\Image\Face_ORL\Origin\Testing10-1-5-2';
    
    TariningNum=0; %训练集个数
    TestingNum=0;%测试集个数
    
    root_list=ls(RootPath);
    len1=length(root_list);
    for i=3:2+10        %前35个
        FilePath=[ RootPath,'\',strtrim(root_list(i,:)) ];
        file_list=ls(FilePath);
        len2=length(file_list);
        for j=3:2+1 %训练集1个
            TariningNum=TariningNum+1;
            copyfile( [FilePath,'\',strtrim(file_list(j,:))],[Training_Path,'\',num2str(TariningNum),'.pgm'] );
        end
        
        begin_index=j;
        for j=begin_index:begin_index-1+1 %测试集1个
            TestingNum=TestingNum+1;
            copyfile( [FilePath,'\',strtrim(file_list(j,:))],[Testing_Path,'\',num2str(TestingNum),'.pgm'] );
        end
    end
            
    for i=3+35:len1        %后5个
        FilePath=[ RootPath,'\',strtrim(root_list(i,:)) ];
        file_list=ls(FilePath);
        len2=length(file_list);
        for j=3:2+2 %5个
             TestingNum=TestingNum+1;
             copyfile( [FilePath,'\',strtrim(file_list(j,:))],[Testing_Path,'\',num2str(TestingNum),'.pgm'] );
        end
    end
    
    length(ls(Training_Path))-2
    length(ls(Testing_Path))-2
    


%% 测试图像的查看
    img1=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\Test\2-10.pgm');
    img2=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\Test\me.pgm');
    img3=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\Test\otherman.pgm');
    img4=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\Test\test.pgm');
    subplot(1,4,1),imshow(img1);
    subplot(1,4,2),imshow(img2);
    subplot(1,4,3),imshow(img3);
    subplot(1,4,4),imshow(img4);

%% 下采样代码
    img=imread([ 'E:\毕设\人脸数据库【原】\ORL_Face\s1','\1.pgm' ]);
    subplot(1,3,1),imshow(img);
    [row,column]=size(img)
    img=img(1:2:row,:);%隔两行采样
    img=img(:,1:2:column);%隔两列采样
    [row,column]=size(img)
    subplot(1,3,2),imshow(img);

%% 数据库查看
    list=ls('E:\毕设\人脸数据库【原】\FaceDB\#FaceDB#\FaceDB_orl\001');
    len=size(list);
    for i=3:8
        subplot(1,5,i-2);
        img=imread([ 'E:\毕设\人脸数据库【原】\FaceDB\#FaceDB#\FaceDB_orl\008','\',list(i,:) ]);
        imshow(img);
    end


%%   输出图像平滑一下
    img=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\LR.pgm');
    subplot(1,4,1),imshow(img);
    img=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\HROuput\LR\LR.pgm');
    subplot(1,4,2),imshow(img);
    
    fH=fspecial('average',3);               %均值滤波器
    img=imfilter(img,fH);      %模糊
    subplot(1,4,3),imshow(img);

%%  自己的照片转换为输入图像
    img=imread('E:\otherman.png');
    % 设置输出文件名  
    % 最后目录中的imageName文件即为转化后的pgm文件  
    img=imresize(img,[112 92]);
    imwrite(img,'E:\毕设\LR.pgm');
    
    img=imread('E:\毕设\LR.pgm');
    imshow(img);

    %%  输入图像大小
%     img=imread('E:\毕设\2.pgm');
%     size(img)

    
%% 测试论文不同的特征提取算法，不同旋转的效果：一阶导数、二阶导数、二阶微分

%下采样
%     img=img(1:3:row,:);%隔两行采样
%     img=img(:,1:3:column);%隔两列采样

    filter=[-1,0,1];       %一阶导数    行向量则列方向特征明显，列向量则行方向特征明显；LR特征均无
%     filter=[1,0,-2,0,1];       %二阶导数    同上。LR一点点，几乎没有
%     filter=[0,-1,0;-1,4,-1;0,-1,0];       %二阶微分，方向无变化，内容一致，无法规避椒盐
    img=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\LR.pgm');
    subplot(3,3,1),imshow(img),title('HR图像');
    new_img=imfilter(img,filter);     subplot(3,3,2),imshow(new_img),title('HR一阶导数-行');filter=filter';
    new_img=imfilter(img,filter);     subplot(3,3,3),imshow(new_img),title('HR一阶导数-列');filter=filter';
    
    %下采样
    [row,column]=size(img);
    img1=img(1:3:row,:);%隔两行采样
    img1=img1(:,1:3:column);%隔两列采样
    subplot(3,3,4),imshow(img1),title('下采样获得LR图像');
    new_img=imfilter(img1,filter);     subplot(3,3,5),imshow(new_img),title('LR一阶导数-行');filter=filter';
    new_img=imfilter(img1,filter);     subplot(3,3,6),imshow(new_img),title('LR一阶导数-列');filter=filter';
    
    fH=fspecial('average',3);               
    img1=imfilter(img,fH);      %均值模糊
    img1 = imnoise(img1, 'gaussian', 0, 10^2/255^2);%加入高斯白躁声
    img1=imnoise(img1,'salt & pepper',0.02); %加入椒盐躁声
    subplot(3,3,7),imshow(img1),title('滤波加噪声获得模糊图像');
    new_img=imfilter(img1,filter);     subplot(3,3,8),imshow(new_img),title('LR一阶导数-行');filter=filter';
    new_img=imfilter(img1,filter);     subplot(3,3,9),imshow(new_img),title('LR一阶导数-列');filter=filter';
    


%%  测试论文不同的特征提取算法的效果：一阶导数、二阶导数、二阶微分
    filter1=[-1,0,1];       %一阶导数，LR图像提不出来，椒盐噪声倒是提的很好，笑死
    filter2=[1,0,-2,0,1];       %二阶导数，同上
    filter3=[0,-1,0;-1,8,-1;0,-1,0];       %二阶微分

    img=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\LR.pgm');
    subplot(3,4,1),imshow(img),title('HR原图像');
    new_img=imfilter(img,filter1);    subplot(3,4,2),imshow(new_img),title('HR一阶导数');
    new_img=imfilter(img,filter2);    subplot(3,4,3),imshow(new_img),title('HR二阶导数');
    new_img=imfilter(img,filter3);    subplot(3,4,4),imshow(new_img),title('HR二阶微分');
    
%     new_img=edge(img,'canny');    subplot(3,4,9),imshow(new_img),title('HRCanny算子边缘检测');
        
    %下采样
    [row,column]=size(img);
    img1=img(1:3:row,:);%隔两行采样
    img1=img1(:,1:3:column);%隔两列采样
    subplot(3,4,5),imshow(img1),title('下采样LR图像');
    new_img=imfilter(img1,filter1);    subplot(3,4,6),imshow(new_img),title('下采样LR图像一阶导数');
    new_img=imfilter(img1,filter2);    subplot(3,4,7),imshow(new_img),title('下采样LR图像二阶导数');
    new_img=imfilter(img1,filter3);    subplot(3,4,8),imshow(new_img),title('下采样LR图像二阶微分');
    
    fH=fspecial('average',3);               
    img1=imfilter(img,fH);      %均值模糊
    img1 = imnoise(img1, 'gaussian', 0, 10^2/255^2);%加入高斯白躁声
    img1=imnoise(img1,'salt & pepper',0.02); %加入椒盐躁声
    subplot(3,4,9),imshow(img1),title('滤波&噪声获得模糊图像');
    new_img=imfilter(img1,filter1);    subplot(3,4,10),imshow(new_img),title('滤波&噪声获得模糊图像一阶导数');
    new_img=imfilter(img1,filter2);    subplot(3,4,11),imshow(new_img),title('滤波&噪声获得模糊图像二阶导数');
    new_img=imfilter(img1,filter3);    subplot(3,4,12),imshow(new_img),title('滤波&噪声获得模糊图像二阶微分');
%     new_img=edge(img,'canny');    subplot(3,4,10),imshow(new_img),title('LRCanny算子边缘检测');

%     img=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\HROuput\LR\LR.pgm');
%     subplot(1,4,4),imshow(img);

%%  测试OMP归一化修改情况
    fID=5;
    SinglePatch_SparseRepresentation(fID);
    SinglePatch_SR_Recovery(fID);


%%  输出处理后的LR图像
    img=imread('E:\毕设\Code\Image\Face_ORL\LR_Dictionary\5.pgm');imshow(img);


%%  查看输出的HR块
     img1=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\HROuput\LR\1.pgm');
     img2=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\HROuput\LR\2.pgm');
    subplot(1,3,1),imshow(img1);
    subplot(1,3,2),imshow(img2);


%%  局部测试——测试全局拼接
    TotalImage_SR_Recovery(fRowRes,fColumnRes);

%%  DEBUG——图像溢出
     img1=imread('E:\毕设\2.pgm');img1=int16(img1);
     img2=imread('E:\毕设\2.pgm');img2=int16(img2);
     img3=(img1+img2)/2;
     
     img1=uint8(img1);
     img2=uint8(img2);
     img3=uint8(img3);
    subplot(1,3,1),imshow(img1);
    subplot(1,3,2),imshow(img2);
    subplot(1,3,3),imshow(img3);

%%  DEBUG——块的全局拼接算法2
    img=imread('E:\毕设\2.pgm');
    overlap=10;
    [n m]=size(img);
    x=(m+overlap)/2;
    img1=img(:,1:x);
    img2=img(:,end-x+1:end);
    
    size0=size(img);size0=size0(2)
    size1=size(img1);size1=size1(2);
    size2=size(img2);size2=size2(2);
    
    subplot(1,4,1),imshow(img1);
    subplot(1,4,2),imshow(img2);
    img3=[ img1(:,1:end-overlap), (img1(:,end-overlap+1:end)+img2(:,1:overlap))/2 ,img2(:,1+overlap:end) ];
    subplot(1,4,3),imshow(img);
    subplot(1,4,4),imshow(img3);
    

%%   DEBUG——块的全局拼接算法1
    img1=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\对比\1.pgm');
    img2=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\对比\2.pgm');
    subplot(1,3,1),imshow(img1);
    subplot(1,3,2),imshow(img2);
    img3=[ img1(:,1:end-7), (img1(:,end-6:end)+img2(:,1:7))/2 ,img2(:,8:end) ];
    subplot(1,3,3),imshow(img3);
    
%     TotalImage_SR_Recovery(fRowRes,fColumnRes);

%%  滤波器效果测试
    img=imread('E:\毕设\2.pgm');
    subplot(1,4,1),imshow(img);
    fH=fspecial('average',6);               %均值滤波器
    % fH = fspecial('gaussian',40,0.5);    %高斯滤波器，好像没为什么用
    fImgDownsample=imfilter(img,fH);      %模糊
    fImgDownsample=imfilter(fImgDownsample,fH);      %模糊
    
    subplot(1,4,2),imshow(fImgDownsample);
    
    fImgDownsample = imnoise(fImgDownsample, 'gaussian', 0, 10^2/255^2);%加入高斯白躁声
    subplot(1,4,3),imshow(fImgDownsample);
        fImgDownsample=imnoise(fImgDownsample,'salt & pepper',0.02); %加入椒盐躁声
    subplot(1,4,4),imshow(fImgDownsample);



%%  特征提取效果测试2
img1=imread('E:\毕设\2.pgm');
subplot(1,3,1),imshow(img1);
img2=SingleImageDownsample(img1);
subplot(1,3,2),imshow(img2);

filter=[0,-1,0;-1,8,-1;0,-1,0];       %二阶微分最理想
img3=imfilter(img2,filter);
subplot(1,3,3),imshow(img3);



%%  特征提取效果测试1
%二阶微分特征提取,效果很一般
i=1;
 img=imread('E:\毕设\face1.jfif');
 figure(i),imshow(img);
 filter=[0,-1,0;-1,4,-1;0,-1,0];
 img=imfilter(img,filter);
 figure(i),imshow(img),i=i+1;
  filter=[0,-1,0;-1,8,-1;0,-1,0];       %最理想
 img=imfilter(img,filter);
 figure(i),imshow(img),i=i+1;
  filter=-filter;
 img=imfilter(img,filter);
 figure(i),imshow(img),i=i+1;
 




%%
% matrix = randi(50, 10, 5)
% mc = mat2cell(matrix, [3 5 2], [3 2])
% 
% 



%%
% M=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\LR.pgm') %读取MATLAB中的名为cameraman的图像
% subplot(3,3,1)
% imshow(M) %显示原始图像
% title('original')
% 
% P1=imnoise(M,'gaussian',0.02) %加入高斯躁声
% subplot(3,3,2)
% imshow(P1) %加入高斯躁声后显示图像
% title('gaussian noise');
% 
% P2=imnoise(M,'salt & pepper',0.02) %加入椒盐躁声
% subplot(3,3,3)
% imshow(P2) %%加入椒盐躁声后显示图像
% title('salt & pepper noise');
% 
% g=medfilt2(P1) %对高斯躁声中值滤波
% subplot(3,3,5)
% imshow(g)
% title('medfilter gaussian')
% 
% h=medfilt2(P2) %对椒盐躁声中值滤波
% subplot(3,3,6)
% imshow(h)
% title('medfilter salt & pepper noise')
% 
% l=[1 1 1 %对高斯躁声算术均值滤波
% 1 1 1
% 1 1 1];
% l=l/9;
% k=conv2(P1,l)
% subplot(3,3,8)
% imshow(k,[])
% title('arithmeticfilter gaussian')
% 
% %对椒盐躁声算术均值滤波
% d=conv2(P2,l)
% subplot(3,3,9)
% imshow(d,[])
% title('arithmeticfilter salt & pepper noise') 




%%

% 
% 
% N=80;
% fRow=N/3;
% fColumn=N/3;
% overlap=3;
% 
%     for i=1:3
%         for j=1:3 
%             fLeft=max( 1,(j-1)*fColumn-overlap/2 );                   %左侧像与上侧素不可小于1，故用max
%             fRight=min( j*fColumn+overlap/2,   N);                  %右侧与下侧像素不可大于最大值，故用min
%             fTop=max( 1,(i-1)*fRow-overlap/2 );
%             fBottom=min( i*fRow+overlap/2, N );
%             
%             row=fBottom-fTop+1;
%             column=fRight-fLeft+1;
%             fprintf("%d %d\n",row,column);
%         end
%     end









%%

% 
% img1=imread("E:\毕设\FaceData\BioID-FaceDatabase-V1.2\BioID_0020.pgm");
% img2=imread("E:\毕设\FaceData\BioID-FaceDatabase-V1.2\BioID_0050.pgm");
% 
% %第一张图像的显示
% subplot(1,2,1);
% imshow(img1);
% title("First!");%title一定放在后面
% 
% 
% %第二张图像的显示
% subplot(1,2,2);
% imshow(img2);
% title("Second!");%title一定放在后面
% 
