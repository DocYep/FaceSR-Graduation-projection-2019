%This script is written for LRDictionary_PreProcessing, which preprocesses
%LRDictionary ,such as HR2LR by blurring, images' feature extracting, and
%LR or HR patches generating.

%1. The script is done for image-processing. That like LR image processing
%from Origin-LR images. And Origin-LR images are actually HR images.
%2. Then, according to 'PachingProducing.m', we divide every image into
%patches in this script.


%% 0. 自己把数据库手动复制到核心是gHR_Path下（当然如果有字典训练，当我没说）

global gFormat;
global gLR_Path;            %低分辨率字典路径
global gLRPatch_Path;
global gHR_Path;            %高分辨率字典路径
global gHRPatch_Path;
global gLRFeaturePatch_Path;
global gTrainingNum;           %人脸数

%% 1. Delete last Result。 核心是gHR_Path，之后的LR、LR块、HR块、特征都是派生出来的。所以要清理4个
    rmdir(gLR_Path,'s');
    mkdir( [gLR_Path] );
    
    rmdir(gLRPatch_Path,'s');
    mkdir( [gLRPatch_Path] );
    
    rmdir(gHRPatch_Path,'s');
    mkdir( [gHRPatch_Path] );
    
    rmdir(gLRFeaturePatch_Path,'s');
    mkdir( [gLRFeaturePatch_Path] );

     disp('Have Clearing work done!');

%% 2. LR dictinary's transforming,由HR位置分块到HR碎片，再复制到LR位置，再分块到LR碎片

for i=1:gTrainingNum 
    fFileName=[num2str(i),gFormat];
    fSourcePath=[gHR_Path,'\',fFileName];
    fImg=imread(fSourcePath);        %Read Image
    %     imshow(fImg);                       %show
    SingleImagePatchGenerating(fImg,i,gHRPatch_Path);     %Generating HR Patches
    fImg=SingleImageDownsample(fImg);       %Downsample the HR image to LR image.
    %     figure,imshow(fImg);
    fDistPath=[gLR_Path,'\',num2str(i),gFormat];
    imwrite(fImg,fDistPath);
    
    SingleImagePatchGenerating(fImg,i,gLRPatch_Path);     %Generating LR Patches
    
    disp([ 'Have LR ',num2str(i),' Patches Generating done!' ]);
end

%     disp('Have LR Patches Generating done!');
%% 3.  images' feature extracting，由LR位置，特征提取到块位置，再原地分块
%无特征提取算法，故使用原LR文件
gLRFeaturePatch_Path=gLRPatch_Path;
% 
% for i=1:gTrainingNum 
%     fFileName=[num2str(i),gFormat];
%     fSourcePath=[gLR_Path,'\',fFileName];
%     fImg=imread(fSourcePath);        %Read Image
%     %     imshow(fImg);                       %show
%     fImg=SingleImageFeatureExtracting(fImg);      %特征
%     %     figure,imshow(fImg);
%     fDistPath=[gLRFeaturePatch_Path,'\',num2str(i),gFormat];
%     imwrite(fImg,fDistPath);
%     
%     %  images' Patches Generating
%     SingleImagePatchGenerating(fImg,i,gLRFeaturePatch_Path);     %Generating feature Patches in this path!
% end

