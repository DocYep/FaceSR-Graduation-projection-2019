clc;

%% ������ͼʵ��
 

%% ����ٶȲ����������ǵ�һ���찡������
load('MyDictionary35-1.mat'); %���Լ�ѵ����
A=Dl;
tic;
A = normalize(A,'norm',2);
toc;

tic;
A = A./repmat(sqrt(sum(A.^2)), size(A, 1), 1);
toc;

%% ͼ��GOG��ȡ����
clear,clc;

f = 1; % 1 -- GOG_RGB, 2 -- GOG_Lab, 3 -- GOG_HSV, 4 -- GOG_nRnG
param = set_default_parameter(f); 

I = im2double(imread('E:\����\me.png')); % load image 
feature_vec = GOG(I, param); 
 % Note that output of the feature vector is not normalized.


%% ͼ���������Ϊ����������� ����
path='E:\����\Codeħ��\Image\Face_ORL\Origin\Testing10-1-5-2';
patchSize=10;
list=ls(path);
len=length(list);
imgList=[];
feaList=[];
num=0;
for id=3:len
    %����ӷֿ�
    img=im2double(imread([path '\' list(id,:)]));
    imgSize=size(img);
    for i=1:patchSize:imgSize(1)-patchSize
        for j=1:patchSize:imgSize(2)-patchSize
            patch=img(i:i+patchSize-1,j:j+patchSize-1);
            num=num+1;
            patch=patch(:);
            imgList(:,num)=patch;
            feaList(:,num)=SingleImageFeatureExtracting(patch);%������
        end
    end
end
img=im2double(imread('E:\����\Codeħ��\Image\Face_ORL\Input&Output\LR.pgm'));
patchSize=10;
topBegin=23;%X���꣬Ҳ����ˮƽ����
leftBegin=49;%Y���꣬Ҳ���Ǵ�ֱ����

% imshow(img,'InitialMagnification','fit');
% impixelinfo;%��ʾ����λ��
% rect = [leftBegin topBegin patchSize patchSize];
% rectangle('Position',rect,'LineWidth',4,'EdgeColor','r');
%%
patch=img(leftBegin:leftBegin-1+patchSize,topBegin:topBegin-1+patchSize);imshow(patch,'InitialMagnification','fit');
patch=patch(:);
%ֱ�����ض�����
[a b1]=sort(sum((patch-imgList).^2));
%������ȡ������
[a b2]=sort(sum((SingleImageFeatureExtracting(patch)-feaList).^2));

subplot(2,10,1),imshow(reshape(patch,[patchSize patchSize])),title('Origin');     %����ԭ��ͼƬ�����������״
subplot(2,10,7),imshow(reshape(patch,[patchSize patchSize])),title('Origin');     %����ԭ��ͼƬ�����������״
%��ʾ����
for i=1:10
    patch=reshape(imgList(:,b1(i)),[patchSize patchSize]);
    subplot(2,10,i+1),imshow(reshape(patch,[patchSize patchSize]));     %����ԭ��ͼƬ�����������״�������ض�����
    patch=reshape(imgList(:,b2(i)),[patchSize patchSize]);
    subplot(2,10,i+1+10),imshow(reshape(patch,[patchSize patchSize]));     %����ԭ��ͼƬ�����������״��������������
end

%% �㷨�������ԭ��̽������ϡ���ʾ�㷨�����ֵ�����
    clc;clear;warning off;
    global gGap;
    gGap=2;
    
    HR=imread('E:\����\Codeħ��\Image\Face_ORL\Input&Output\LR.pgm');
    LR=SingleImageDownsample(HR);
    patchSize=10;
    leftBegin=49;%Y���꣬Ҳ���Ǵ�ֱ����
    topBegin=23;%X���꣬Ҳ����ˮƽ����
    lambda=0.2;%Ϲȡ��
    
    %����ԭͼ����ǿ�λ��
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
    
    % �ֵ�
    load('MyDictionary10-1.mat'); %���Լ�ѵ���ģ���һ��
%     load('MyDictionary35-1.mat'); %���Լ�ѵ���ģ�������
    fImgSize=sqrt(size( Dh,1 ));      %��HR���С��ԭ�����Ѳ��븴ԭ
    fImgSize=[fImgSize fImgSize];
    gAtomNum=1024;%�ƻ�ѵ���õ�ԭ����Ŀ
    A=Dl;  
    b=im2double( LRpatch(:) );
    
    fBicubic=imresize(LRpatch, fImgSize, 'bicubic');%�Ŵ��LRͼ
    fBicubic=im2double(fBicubic);
    
    %ѡ��ָ��ķ�ʽ��1�ǰ���񻯵ı�����0�ǰ�ģ��ֵ
    type=0;
    
    if type
        B=A; 
        A = normalize(A,'norm',2);
        fZoom=A(1,:)./B(1,:);%����LR�ֵ���ǰ�������С�仯�ı���,����./
    else
        fLRNorm=sqrt(sum(fBicubic.^2));
        A = normalize(A,'norm',2);
    end
    
    %SR����OMP,            0.004s
    x=OMP(A,b,floor(0.15*gAtomNum));        %�������ȫ����double
    x=full(x); 
    x=x(:,1);
    if type
        x=x.*fZoom';%ʹ��ϵ��ƽ��,����.*        
    end
    fOMPOut=x'.*Dh;           %��ԭͼ
    fOMPOut=sum(fOMPOut')';       %��Ҫ������һ��
    fOMPOut=reshape(fOMPOut,fImgSize);     %����ԭ��ͼƬ�����������״
    if ~type
        fHRNorm=sqrt(sum(fOMPOut.^2));
        fOMPOut=fOMPOut.*(fLRNorm*1.2/fHRNorm);
    end

    %SR����my-ADMM,   0.013s
    %����񻯵Ļ�ƽ���ܶ൫�ǱȽϰ�
    [x history] = ADMM_lasso(A,b, lambda, 1.0, 1.0); %û�й�񻯣����Կ���
    if type
        x=x.*fZoom';%ʹ��ϵ��ƽ��,����.*
    end
    fMyADMMOut=x'.*Dh;           %��ԭͼ
    fMyADMMOut=sum(fMyADMMOut')';       %��Ҫ������һ��
    fMyADMMOut=reshape(fMyADMMOut,fImgSize);     %����ԭ��ͼƬ�����������״
    if ~type
        fHRNorm=sqrt(sum(fMyADMMOut.^2));
        fMyADMMOut=fMyADMMOut.*(fLRNorm*1.2/fHRNorm);
    end

    %SR����ѧ��-ADMM,   0.187
    [x,obj,err,iter] = l1(A,b);%��Ĭ�ϵ�obt�������ҾͲ�����
    if type
        x=x.*fZoom';%ʹ��ϵ��ƽ��,����.*
    end
    fL1ADMMOut=x'.*Dh;           %��ԭͼ
    fL1ADMMOut=sum(fL1ADMMOut')';       %��Ҫ������һ��
    fL1ADMMOut=reshape(fL1ADMMOut,fImgSize);     %����ԭ��ͼƬ�����������״
    if ~type
        fHRNorm=sqrt(sum(fL1ADMMOut.^2));
        fL1ADMMOut=fL1ADMMOut.*(fLRNorm*1.2/fHRNorm);
    end
    
    %Yang
    A = Dl'*Dl;
    b = -Dl'*b;
    x= L1QP_FeatureSign_yang(lambda,A,b);
    if type
        x=x.*fZoom';%ʹ��ϵ��ƽ��,����.*
    end
    fYangOut=x'.*Dh;           %��ԭͼ
    fYangOut=sum(fYangOut')';       %��Ҫ������һ��
    fYangOut=reshape(fYangOut,fImgSize);     %����ԭ��ͼƬ�����������״
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

%% tic toc ʵ��
    tic;
    for i=1:100000
        disp(6);
    end
    toc;
    
%% ��ȡ�����ֵ��
    load('D_1024_0.15_5.mat');

%% paperͼ��
    img0=imread('E:\����\Code\Image\Face_ORL\Origin\Paper\t1.bmp');
    size(img0)
    subplot(1,3,1),imshow(img0);
    img0=img0(:,:,1);
    img1=imresize(img0,0.5,'bicubic');                                     %������ͼ���ֵ�������ֵ��С
    subplot(1,3,2),imshow(img1);
    
    
    img=img0(:,:,1);
[row,column]=size(img);%��ͼ����и��г�ȡԪ�أ��õ���С��ͼ��f
m=floor(row/2);n=floor(column/2);
f=zeros(m,n);
for i=1:m
    for j=1:n
        f(i,j)=img(2*i,2*j);
    end
end
subplot(1,3,3),imshow(uint8(f));
title('��С��ͼ��');%��ʾ��С��ͼ��

        
%% �ֵ�ѧϰ����
        %��֤������ز���
    global gTestingNum;
    global gTestingHRImage;   %��֤��HRͼ������
    global gTestingLRImage;   %��֤��LRͼ������
    global gTestingHRPatch;            %���Լ���·��
    global gTestingLRPatch;            %���Լ���·��
    
    % ȡ����Ӧλ�õ�HR���Լ���
    fList=ls( [gTestingHRPatch_Path,'\',num2str(fID)] );
    fLen=length(fList);
    gTestingHR=[];                           %���Ƕ�Ӧλ��patch��TestingHR
    for i=3:fLen
        fImg=imread( [gTestingHRPatch_Path,'\',num2str(fID),'\',fList(i,:)] );
        fImg=im2double(fImg);    %ͼƬ����Ҫת����ʵ�������԰ٶ�,�����Ϳ��Խ���������������ˣ���Ҫ���N*1��
        fImg=fImg(:);   %�˴�ͼƬ�Ż�Ϊ����
        gTestingHR=[gTestingHR fImg];     %��ΪTestingHR
    end
    % ȡ����Ӧλ�õ�LR���Լ���
    fList=ls( [gTestingLRPatch_Path,'\',num2str(fID)] );
    fLen=length(fList);
    gTestingLR=[];                           %���Ƕ�Ӧλ��patch��TestingLR
    for i=3:fLen
        fImg=imread( [gTestingLRPatch_Path,'\',num2str(fID),'\',fList(i,:)] );
        fImg=im2double(fImg);    %ͼƬ����Ҫת����ʵ�������԰ٶ�,�����Ϳ��Խ���������������ˣ���Ҫ���N*1��
        fImg=fImg(:);   %�˴�ͼƬ�Ż�Ϊ����
        gTestingLR=[gTestingLR fImg];     %��ΪTestingLR
    end
    fTestingData=[gTestingHR;gTestingLR]; %HR��LR��
    fDictionary=[fHR_A1;fLR_A];%HR��LR��
    [fHR_A1 fLR_A]=My_KSVD(fTestingData,fDictionary); %�õ������Ժ���ֵ�
    %HR_A���й�һ������
%     fHR_A = normalize(fHR_A,'norm',2);      %OMP��ϡ���ʾʱϵ���ǰ��չ�һ�����,��ô��ԭ�ع���ʱ���ֵ�ҲҪ��һ��������




%%  Debug
    clc;
    fColumn=31;
    fResidual=4;
    fPatchSize=10;
    
    fColumnRes=mod(fResidual-(fColumn-fPatchSize),fResidual);     %�в���
    mod(fColumnRes,fResidual)
    
    fNew_Column=fColumn+fColumnRes
    fColumnPatches=(fNew_Column-fPatchSize)/fResidual+1         %һ�еĿ���,�гɼ���patch��ʵ�ʵľ�����

%% �ֿ�����������ô��
    figure();
    img=imread('E:\����\Code\Image\Face_ORL\Input&Output\LR.pgm');
%     img=img(1:end/2,1:end/2);
    [row column]=size(img);

    filter=[-1,0,1];       %һ�׵���    ���������з����������ԣ����������з����������ԣ�LR��������
%     filter=[1,0,-2,0,1];       %���׵���    ͬ�ϡ�LRһ��㣬����û��
%     filter=[0,-1,0;-1,4,-1;0,-1,0];       %����΢�֣������ޱ仯������һ�£��޷���ܽ���
    subplot(3,3,1),imshow(img),title('HRͼ��');
    new_img=imfilter(img,filter);     subplot(3,3,2),imshow(new_img),title('HRһ�׵���-��');filter=filter';
    new_img=imfilter(img,filter);     subplot(3,3,3),imshow(new_img),title('HRһ�׵���-��');filter=filter';
    
    %�²���
    img1=img(1:3:row,:);%�����в���
    img1=img1(:,1:3:column);%�����в���
    subplot(3,3,4),imshow(img1),title('LRԭͼ');
    new_img=imfilter(img1,filter);     subplot(3,3,5),imshow(new_img),title('LRһ�׵���-��');filter=filter';
    new_img=imfilter(img1,filter);     subplot(3,3,6),imshow(new_img),title('LRһ�׵���-��');filter=filter';
    
    img1=imnoise(img1,'salt & pepper',0.02);                        %���뽷��������LR�ֵ��ǲ���Ҫ����������
    subplot(3,3,7),imshow(img1),title('LRͼ��');
    new_img=imfilter(img1,filter);     subplot(3,3,8),imshow(new_img),title('LRһ�׵���-��');filter=filter';
    new_img=imfilter(img1,filter);     subplot(3,3,9),imshow(new_img),title('LRһ�׵���-��');filter=filter';

%% ͼ�����ݿ���ƶ�
clc;clear;
    %�Լ���ԭ����ɾ����Ȼ���޸�һ�������·��
    RootPath='E:\����\�������ݿ⡾ԭ��\ORL_Face';
    Training_Path='E:\����\Codeħ��\Image\Face_ORL\Origin\Training10-1';
    Testing_Path='E:\����\Codeħ��\Image\Face_ORL\Origin\Testing10-1-5-2';
    
    TariningNum=0; %ѵ��������
    TestingNum=0;%���Լ�����
    
    root_list=ls(RootPath);
    len1=length(root_list);
    for i=3:2+10        %ǰ35��
        FilePath=[ RootPath,'\',strtrim(root_list(i,:)) ];
        file_list=ls(FilePath);
        len2=length(file_list);
        for j=3:2+1 %ѵ����1��
            TariningNum=TariningNum+1;
            copyfile( [FilePath,'\',strtrim(file_list(j,:))],[Training_Path,'\',num2str(TariningNum),'.pgm'] );
        end
        
        begin_index=j;
        for j=begin_index:begin_index-1+1 %���Լ�1��
            TestingNum=TestingNum+1;
            copyfile( [FilePath,'\',strtrim(file_list(j,:))],[Testing_Path,'\',num2str(TestingNum),'.pgm'] );
        end
    end
            
    for i=3+35:len1        %��5��
        FilePath=[ RootPath,'\',strtrim(root_list(i,:)) ];
        file_list=ls(FilePath);
        len2=length(file_list);
        for j=3:2+2 %5��
             TestingNum=TestingNum+1;
             copyfile( [FilePath,'\',strtrim(file_list(j,:))],[Testing_Path,'\',num2str(TestingNum),'.pgm'] );
        end
    end
    
    length(ls(Training_Path))-2
    length(ls(Testing_Path))-2
    


%% ����ͼ��Ĳ鿴
    img1=imread('E:\����\Code\Image\Face_ORL\Input&Output\Test\2-10.pgm');
    img2=imread('E:\����\Code\Image\Face_ORL\Input&Output\Test\me.pgm');
    img3=imread('E:\����\Code\Image\Face_ORL\Input&Output\Test\otherman.pgm');
    img4=imread('E:\����\Code\Image\Face_ORL\Input&Output\Test\test.pgm');
    subplot(1,4,1),imshow(img1);
    subplot(1,4,2),imshow(img2);
    subplot(1,4,3),imshow(img3);
    subplot(1,4,4),imshow(img4);

%% �²�������
    img=imread([ 'E:\����\�������ݿ⡾ԭ��\ORL_Face\s1','\1.pgm' ]);
    subplot(1,3,1),imshow(img);
    [row,column]=size(img)
    img=img(1:2:row,:);%�����в���
    img=img(:,1:2:column);%�����в���
    [row,column]=size(img)
    subplot(1,3,2),imshow(img);

%% ���ݿ�鿴
    list=ls('E:\����\�������ݿ⡾ԭ��\FaceDB\#FaceDB#\FaceDB_orl\001');
    len=size(list);
    for i=3:8
        subplot(1,5,i-2);
        img=imread([ 'E:\����\�������ݿ⡾ԭ��\FaceDB\#FaceDB#\FaceDB_orl\008','\',list(i,:) ]);
        imshow(img);
    end


%%   ���ͼ��ƽ��һ��
    img=imread('E:\����\Code\Image\Face_ORL\Input&Output\LR.pgm');
    subplot(1,4,1),imshow(img);
    img=imread('E:\����\Code\Image\Face_ORL\Input&Output\HROuput\LR\LR.pgm');
    subplot(1,4,2),imshow(img);
    
    fH=fspecial('average',3);               %��ֵ�˲���
    img=imfilter(img,fH);      %ģ��
    subplot(1,4,3),imshow(img);

%%  �Լ�����Ƭת��Ϊ����ͼ��
    img=imread('E:\otherman.png');
    % ��������ļ���  
    % ���Ŀ¼�е�imageName�ļ���Ϊת�����pgm�ļ�  
    img=imresize(img,[112 92]);
    imwrite(img,'E:\����\LR.pgm');
    
    img=imread('E:\����\LR.pgm');
    imshow(img);

    %%  ����ͼ���С
%     img=imread('E:\����\2.pgm');
%     size(img)

    
%% �������Ĳ�ͬ��������ȡ�㷨����ͬ��ת��Ч����һ�׵��������׵���������΢��

%�²���
%     img=img(1:3:row,:);%�����в���
%     img=img(:,1:3:column);%�����в���

    filter=[-1,0,1];       %һ�׵���    ���������з����������ԣ����������з����������ԣ�LR��������
%     filter=[1,0,-2,0,1];       %���׵���    ͬ�ϡ�LRһ��㣬����û��
%     filter=[0,-1,0;-1,4,-1;0,-1,0];       %����΢�֣������ޱ仯������һ�£��޷���ܽ���
    img=imread('E:\����\Code\Image\Face_ORL\Input&Output\LR.pgm');
    subplot(3,3,1),imshow(img),title('HRͼ��');
    new_img=imfilter(img,filter);     subplot(3,3,2),imshow(new_img),title('HRһ�׵���-��');filter=filter';
    new_img=imfilter(img,filter);     subplot(3,3,3),imshow(new_img),title('HRһ�׵���-��');filter=filter';
    
    %�²���
    [row,column]=size(img);
    img1=img(1:3:row,:);%�����в���
    img1=img1(:,1:3:column);%�����в���
    subplot(3,3,4),imshow(img1),title('�²������LRͼ��');
    new_img=imfilter(img1,filter);     subplot(3,3,5),imshow(new_img),title('LRһ�׵���-��');filter=filter';
    new_img=imfilter(img1,filter);     subplot(3,3,6),imshow(new_img),title('LRһ�׵���-��');filter=filter';
    
    fH=fspecial('average',3);               
    img1=imfilter(img,fH);      %��ֵģ��
    img1 = imnoise(img1, 'gaussian', 0, 10^2/255^2);%�����˹������
    img1=imnoise(img1,'salt & pepper',0.02); %���뽷������
    subplot(3,3,7),imshow(img1),title('�˲����������ģ��ͼ��');
    new_img=imfilter(img1,filter);     subplot(3,3,8),imshow(new_img),title('LRһ�׵���-��');filter=filter';
    new_img=imfilter(img1,filter);     subplot(3,3,9),imshow(new_img),title('LRһ�׵���-��');filter=filter';
    


%%  �������Ĳ�ͬ��������ȡ�㷨��Ч����һ�׵��������׵���������΢��
    filter1=[-1,0,1];       %һ�׵�����LRͼ���᲻��������������������ĺܺã�Ц��
    filter2=[1,0,-2,0,1];       %���׵�����ͬ��
    filter3=[0,-1,0;-1,8,-1;0,-1,0];       %����΢��

    img=imread('E:\����\Code\Image\Face_ORL\Input&Output\LR.pgm');
    subplot(3,4,1),imshow(img),title('HRԭͼ��');
    new_img=imfilter(img,filter1);    subplot(3,4,2),imshow(new_img),title('HRһ�׵���');
    new_img=imfilter(img,filter2);    subplot(3,4,3),imshow(new_img),title('HR���׵���');
    new_img=imfilter(img,filter3);    subplot(3,4,4),imshow(new_img),title('HR����΢��');
    
%     new_img=edge(img,'canny');    subplot(3,4,9),imshow(new_img),title('HRCanny���ӱ�Ե���');
        
    %�²���
    [row,column]=size(img);
    img1=img(1:3:row,:);%�����в���
    img1=img1(:,1:3:column);%�����в���
    subplot(3,4,5),imshow(img1),title('�²���LRͼ��');
    new_img=imfilter(img1,filter1);    subplot(3,4,6),imshow(new_img),title('�²���LRͼ��һ�׵���');
    new_img=imfilter(img1,filter2);    subplot(3,4,7),imshow(new_img),title('�²���LRͼ����׵���');
    new_img=imfilter(img1,filter3);    subplot(3,4,8),imshow(new_img),title('�²���LRͼ�����΢��');
    
    fH=fspecial('average',3);               
    img1=imfilter(img,fH);      %��ֵģ��
    img1 = imnoise(img1, 'gaussian', 0, 10^2/255^2);%�����˹������
    img1=imnoise(img1,'salt & pepper',0.02); %���뽷������
    subplot(3,4,9),imshow(img1),title('�˲�&�������ģ��ͼ��');
    new_img=imfilter(img1,filter1);    subplot(3,4,10),imshow(new_img),title('�˲�&�������ģ��ͼ��һ�׵���');
    new_img=imfilter(img1,filter2);    subplot(3,4,11),imshow(new_img),title('�˲�&�������ģ��ͼ����׵���');
    new_img=imfilter(img1,filter3);    subplot(3,4,12),imshow(new_img),title('�˲�&�������ģ��ͼ�����΢��');
%     new_img=edge(img,'canny');    subplot(3,4,10),imshow(new_img),title('LRCanny���ӱ�Ե���');

%     img=imread('E:\����\Code\Image\Face_ORL\Input&Output\HROuput\LR\LR.pgm');
%     subplot(1,4,4),imshow(img);

%%  ����OMP��һ���޸����
    fID=5;
    SinglePatch_SparseRepresentation(fID);
    SinglePatch_SR_Recovery(fID);


%%  ���������LRͼ��
    img=imread('E:\����\Code\Image\Face_ORL\LR_Dictionary\5.pgm');imshow(img);


%%  �鿴�����HR��
     img1=imread('E:\����\Code\Image\Face_ORL\Input&Output\HROuput\LR\1.pgm');
     img2=imread('E:\����\Code\Image\Face_ORL\Input&Output\HROuput\LR\2.pgm');
    subplot(1,3,1),imshow(img1);
    subplot(1,3,2),imshow(img2);


%%  �ֲ����ԡ�������ȫ��ƴ��
    TotalImage_SR_Recovery(fRowRes,fColumnRes);

%%  DEBUG����ͼ�����
     img1=imread('E:\����\2.pgm');img1=int16(img1);
     img2=imread('E:\����\2.pgm');img2=int16(img2);
     img3=(img1+img2)/2;
     
     img1=uint8(img1);
     img2=uint8(img2);
     img3=uint8(img3);
    subplot(1,3,1),imshow(img1);
    subplot(1,3,2),imshow(img2);
    subplot(1,3,3),imshow(img3);

%%  DEBUG�������ȫ��ƴ���㷨2
    img=imread('E:\����\2.pgm');
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
    

%%   DEBUG�������ȫ��ƴ���㷨1
    img1=imread('E:\����\Code\Image\Face_ORL\Input&Output\�Ա�\1.pgm');
    img2=imread('E:\����\Code\Image\Face_ORL\Input&Output\�Ա�\2.pgm');
    subplot(1,3,1),imshow(img1);
    subplot(1,3,2),imshow(img2);
    img3=[ img1(:,1:end-7), (img1(:,end-6:end)+img2(:,1:7))/2 ,img2(:,8:end) ];
    subplot(1,3,3),imshow(img3);
    
%     TotalImage_SR_Recovery(fRowRes,fColumnRes);

%%  �˲���Ч������
    img=imread('E:\����\2.pgm');
    subplot(1,4,1),imshow(img);
    fH=fspecial('average',6);               %��ֵ�˲���
    % fH = fspecial('gaussian',40,0.5);    %��˹�˲���������ûΪʲô��
    fImgDownsample=imfilter(img,fH);      %ģ��
    fImgDownsample=imfilter(fImgDownsample,fH);      %ģ��
    
    subplot(1,4,2),imshow(fImgDownsample);
    
    fImgDownsample = imnoise(fImgDownsample, 'gaussian', 0, 10^2/255^2);%�����˹������
    subplot(1,4,3),imshow(fImgDownsample);
        fImgDownsample=imnoise(fImgDownsample,'salt & pepper',0.02); %���뽷������
    subplot(1,4,4),imshow(fImgDownsample);



%%  ������ȡЧ������2
img1=imread('E:\����\2.pgm');
subplot(1,3,1),imshow(img1);
img2=SingleImageDownsample(img1);
subplot(1,3,2),imshow(img2);

filter=[0,-1,0;-1,8,-1;0,-1,0];       %����΢��������
img3=imfilter(img2,filter);
subplot(1,3,3),imshow(img3);



%%  ������ȡЧ������1
%����΢��������ȡ,Ч����һ��
i=1;
 img=imread('E:\����\face1.jfif');
 figure(i),imshow(img);
 filter=[0,-1,0;-1,4,-1;0,-1,0];
 img=imfilter(img,filter);
 figure(i),imshow(img),i=i+1;
  filter=[0,-1,0;-1,8,-1;0,-1,0];       %������
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
% M=imread('E:\����\Code\Image\Face_ORL\Input&Output\LR.pgm') %��ȡMATLAB�е���Ϊcameraman��ͼ��
% subplot(3,3,1)
% imshow(M) %��ʾԭʼͼ��
% title('original')
% 
% P1=imnoise(M,'gaussian',0.02) %�����˹����
% subplot(3,3,2)
% imshow(P1) %�����˹��������ʾͼ��
% title('gaussian noise');
% 
% P2=imnoise(M,'salt & pepper',0.02) %���뽷������
% subplot(3,3,3)
% imshow(P2) %%���뽷����������ʾͼ��
% title('salt & pepper noise');
% 
% g=medfilt2(P1) %�Ը�˹������ֵ�˲�
% subplot(3,3,5)
% imshow(g)
% title('medfilter gaussian')
% 
% h=medfilt2(P2) %�Խ���������ֵ�˲�
% subplot(3,3,6)
% imshow(h)
% title('medfilter salt & pepper noise')
% 
% l=[1 1 1 %�Ը�˹����������ֵ�˲�
% 1 1 1
% 1 1 1];
% l=l/9;
% k=conv2(P1,l)
% subplot(3,3,8)
% imshow(k,[])
% title('arithmeticfilter gaussian')
% 
% %�Խ�������������ֵ�˲�
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
%             fLeft=max( 1,(j-1)*fColumn-overlap/2 );                   %��������ϲ��ز���С��1������max
%             fRight=min( j*fColumn+overlap/2,   N);                  %�Ҳ����²����ز��ɴ������ֵ������min
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
% img1=imread("E:\����\FaceData\BioID-FaceDatabase-V1.2\BioID_0020.pgm");
% img2=imread("E:\����\FaceData\BioID-FaceDatabase-V1.2\BioID_0050.pgm");
% 
% %��һ��ͼ�����ʾ
% subplot(1,2,1);
% imshow(img1);
% title("First!");%titleһ�����ں���
% 
% 
% %�ڶ���ͼ�����ʾ
% subplot(1,2,2);
% imshow(img2);
% title("Second!");%titleһ�����ں���
% 
