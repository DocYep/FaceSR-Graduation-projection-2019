%This script is written for LRDictionary_PreProcessing, which preprocesses
%LRDictionary ,such as HR2LR by blurring, images' feature extracting, and
%LR or HR patches generating.

%1. The script is done for image-processing. That like LR image processing
%from Origin-LR images. And Origin-LR images are actually HR images.
%2. Then, according to 'PachingProducing.m', we divide every image into
%patches in this script.


%% 0. �Լ������ݿ��ֶ����Ƶ�������gHR_Path�£���Ȼ������ֵ�ѵ��������û˵��

global gFormat;
global gLR_Path;            %�ͷֱ����ֵ�·��
global gLRPatch_Path;
global gHR_Path;            %�߷ֱ����ֵ�·��
global gHRPatch_Path;
global gLRFeaturePatch_Path;
global gTrainingNum;           %������

%% 1. Delete last Result�� ������gHR_Path��֮���LR��LR�顢HR�顢�����������������ġ�����Ҫ����4��
    rmdir(gLR_Path,'s');
    mkdir( [gLR_Path] );
    
    rmdir(gLRPatch_Path,'s');
    mkdir( [gLRPatch_Path] );
    
    rmdir(gHRPatch_Path,'s');
    mkdir( [gHRPatch_Path] );
    
    rmdir(gLRFeaturePatch_Path,'s');
    mkdir( [gLRFeaturePatch_Path] );

     disp('Have Clearing work done!');

%% 2. LR dictinary's transforming,��HRλ�÷ֿ鵽HR��Ƭ���ٸ��Ƶ�LRλ�ã��ٷֿ鵽LR��Ƭ

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
%% 3.  images' feature extracting����LRλ�ã�������ȡ����λ�ã���ԭ�طֿ�
%��������ȡ�㷨����ʹ��ԭLR�ļ�
gLRFeaturePatch_Path=gLRPatch_Path;
% 
% for i=1:gTrainingNum 
%     fFileName=[num2str(i),gFormat];
%     fSourcePath=[gLR_Path,'\',fFileName];
%     fImg=imread(fSourcePath);        %Read Image
%     %     imshow(fImg);                       %show
%     fImg=SingleImageFeatureExtracting(fImg);      %����
%     %     figure,imshow(fImg);
%     fDistPath=[gLRFeaturePatch_Path,'\',num2str(i),gFormat];
%     imwrite(fImg,fDistPath);
%     
%     %  images' Patches Generating
%     SingleImagePatchGenerating(fImg,i,gLRFeaturePatch_Path);     %Generating feature Patches in this path!
% end

