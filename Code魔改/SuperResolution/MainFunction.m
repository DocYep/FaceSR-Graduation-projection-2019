clc;
clear;
warning off;

% %% 读取本地字典库
global Dh;
global Dl;
global gAtomNum;%计划训练好的原子数目
% load('feature-HROrigin-Dic35-1_10.mat'); %我自己训练的，意思是基于35张HR图像，块大小10
load('feature-HROrigin-Dic10-1_5.mat'); %我自己训练的，基于10张HR图像，块大小5
% load('MyDictionary10-1_5.mat'); %我自己训练的
% load('MyDictionary35-1_5.mat'); %我自己训练的

%% 程序内容
global gRootFilePath;           %程序根目录位置
gRootFilePath='E:\毕设\Code魔改\';     

global gFormat;                     %图像格式
global gInputLR;                    %输入LR图像
global gInput_Path;            %输入LR图像位置
global gInputFileName;      %输入LR图像文件名
global gOutput_Path;         %输出HR图像位置
global gHR_Path;            %高分辨率字典路径
global gTainingFolder;
global gTestingFolder;

global gRowPatches;                    %  切成几行patch，实际的具体数，补全后的块数行方向
global gColumnPatches;             %  切成几列patch，实际的具体数，补全后的块数列方向
global gLRPatchSize;              %LR块的像素大小
global gHRPatchSize;              %HR块的像素大小
global gLROverlap;            %LR块的冗余像素个数
global gHROverlap;            %HR块的冗余像素个数
global gLRResidual;              %LR块大小与LR Overlap之差
global gHRResidual;              %HR块大小与HR Overlap之差

%分块前
global gTrainingNum;             %训练集人脸数
global gHRDictionary;            %高分辨率字典
global gLRDictionary;            %低分辨率字典
global gHRRow;              %HR插值补齐后的大小,输出HR图像要按这个复原!!!
global gHRColumn;
global gLRRow;              %LR插值后的大小
global gLRColumn;

%分块完毕
%字典库分块结果
global gInputPatch;         %输入LR图像块集
global gInputFeaPatch;  %输入LR特征图像块集
global gLRDictionaryPatch;         %LR字典块集
global gHRDictionaryPatch;         %HR字典块集

%验证集的相关参数
global gTestingNum;         %测试集人脸数
global gTestingHRImage;   %验证集HR图像内容
global gTestingLRImage;   %验证集LR图像内容
global gTestingHRPatch;            %测试集块
global gTestingLRPatch;            %测试集块

global gSRPatch;    %重构出的SR图像块！！！
global gLastAns;                %上次迭代输出的结果
global gSRArray;                %单次patch 稀疏表示系数数组
global gGap;                    %下采样提取的间隔，默认3，影响HR、LR块大小
global gOverlapRatio;       %冗余区域比例，默认2/3

gFormat='.pgm';
gGap=2;%特征提取的间隔，默认3
gOverlapRatio=2/3;%冗余区域比例，默认2/3
gInputFileName='';                          %输入LR图像文件名
gInput_Path=[ gRootFilePath '\Image\Face_ORL\Input&Output\LRInput' ];                    %输入LR图像位置
gOutput_Path=[  gRootFilePath '\Image\Face_ORL\Input&Output\HROuput' ];               %输出HR图像位置
gHR_Path=[ gRootFilePath '\Image\Face_ORL\HR_Dictionary' ];             %HR字典训练集图像位置
gTainingFolder='Training35-1';%----------训练字典要修改的地方------------------------------------------------------------------------------------
gTestingFolder='Testing35-1-5-2';

gAtomNum=1024;  %训练好的字典数目为多少
gLastAns=[];
gSRArray=zeros(int32(gAtomNum));
%% 输入图像
rmdir(gInput_Path,'s'); 
mkdir( [gInput_Path] );
copyfile([ gRootFilePath '\Image\Face_ORL\Input&Output\LR.pgm'] ,gInput_Path);
gInputFileName='LR';
%% 本地数据库的迁移——训练人脸数目35,  测试人脸数目45
TrainingData_LocalCopy;
% TestingImageReading;
%% 迭代信息设置
global gIter;       %迭代指针
fNumOfIteration=2;      %迭代次数
% fLRPatchSizeArray=[ 3 ];       %测试1，单次重构各迭代LR块大小
fLRPatchSizeArray=[ 5 5 5 5 ];       %测试2，多次重构，同块大小
% fLRPatchSizeArray=[ 10 5 ];       %测试3，究极； 但是字典要分别启用

%% 正文迭代
for gIter=1:fNumOfIteration     %多层迭代
    % 输入测试图像、HR字典、LR字典的分块操作
    gLRPatchSize=fLRPatchSizeArray( gIter );              %LR块的像素大小
%     if gIter==1  %如果是测试3则开启，但是最上面的载入代码别忘了注释
%         load('feature-HROrigin-Dic35-1_10.mat'); %我自己训练的，块10
%     else
%         load('feature-HROrigin-Dic10-1_5.mat'); %我自己训练的，块5
%     end
    
    gHRPatchSize=gLRPatchSize*gGap;     %与下采样倍数相同
    
    DictionaryDownsampleing;          %2、计算下采样的LR图片信息与下采样补齐，生成待训练图片对
    DictionaryPatchGenerating;      %4、输入图像与字典的分块补齐与分块以及验证集预处理
%     Dictionary_Training;          %图片对分块并进行字典学习--【后续不需要重复运行】
    disp( ['now the gAtomNum is ' num2str(gAtomNum)]);  %输出当前数据库大小

    %% 5、开始稀疏表示算法
    tic;
    for i=1:gRowPatches
        for j=1:gColumnPatches
                fId=(i-1)*gColumnPatches+j;
                SinglePatch_SparseRepresentation(fId);                                      %各块求稀疏表示
        end
    end
    toc;
    gInputLR=gInputLR(1:gLRRow,1:gLRColumn);                      %行列0补齐收回，将输入图像逆变换成原大小
    gLastAns=TotalImage_SR_Recovery();                 %6、碎片合并，更新为上一次结果
end
disp('Done!');
OutputTesting;    
%实验结果比对
% Test_ImageRead;             %Just for test;