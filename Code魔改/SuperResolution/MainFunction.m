clc;
clear;
warning off;

% %% ��ȡ�����ֵ��
global Dh;
global Dl;
global gAtomNum;%�ƻ�ѵ���õ�ԭ����Ŀ
% load('feature-HROrigin-Dic35-1_10.mat'); %���Լ�ѵ���ģ���˼�ǻ���35��HRͼ�񣬿��С10
load('feature-HROrigin-Dic10-1_5.mat'); %���Լ�ѵ���ģ�����10��HRͼ�񣬿��С5
% load('MyDictionary10-1_5.mat'); %���Լ�ѵ����
% load('MyDictionary35-1_5.mat'); %���Լ�ѵ����

%% ��������
global gRootFilePath;           %�����Ŀ¼λ��
gRootFilePath='E:\����\Codeħ��\';     

global gFormat;                     %ͼ���ʽ
global gInputLR;                    %����LRͼ��
global gInput_Path;            %����LRͼ��λ��
global gInputFileName;      %����LRͼ���ļ���
global gOutput_Path;         %���HRͼ��λ��
global gHR_Path;            %�߷ֱ����ֵ�·��
global gTainingFolder;
global gTestingFolder;

global gRowPatches;                    %  �гɼ���patch��ʵ�ʵľ���������ȫ��Ŀ����з���
global gColumnPatches;             %  �гɼ���patch��ʵ�ʵľ���������ȫ��Ŀ����з���
global gLRPatchSize;              %LR������ش�С
global gHRPatchSize;              %HR������ش�С
global gLROverlap;            %LR����������ظ���
global gHROverlap;            %HR����������ظ���
global gLRResidual;              %LR���С��LR Overlap֮��
global gHRResidual;              %HR���С��HR Overlap֮��

%�ֿ�ǰ
global gTrainingNum;             %ѵ����������
global gHRDictionary;            %�߷ֱ����ֵ�
global gLRDictionary;            %�ͷֱ����ֵ�
global gHRRow;              %HR��ֵ�����Ĵ�С,���HRͼ��Ҫ�������ԭ!!!
global gHRColumn;
global gLRRow;              %LR��ֵ��Ĵ�С
global gLRColumn;

%�ֿ����
%�ֵ��ֿ���
global gInputPatch;         %����LRͼ��鼯
global gInputFeaPatch;  %����LR����ͼ��鼯
global gLRDictionaryPatch;         %LR�ֵ�鼯
global gHRDictionaryPatch;         %HR�ֵ�鼯

%��֤������ز���
global gTestingNum;         %���Լ�������
global gTestingHRImage;   %��֤��HRͼ������
global gTestingLRImage;   %��֤��LRͼ������
global gTestingHRPatch;            %���Լ���
global gTestingLRPatch;            %���Լ���

global gSRPatch;    %�ع�����SRͼ��飡����
global gLastAns;                %�ϴε�������Ľ��
global gSRArray;                %����patch ϡ���ʾϵ������
global gGap;                    %�²�����ȡ�ļ����Ĭ��3��Ӱ��HR��LR���С
global gOverlapRatio;       %�������������Ĭ��2/3

gFormat='.pgm';
gGap=2;%������ȡ�ļ����Ĭ��3
gOverlapRatio=2/3;%�������������Ĭ��2/3
gInputFileName='';                          %����LRͼ���ļ���
gInput_Path=[ gRootFilePath '\Image\Face_ORL\Input&Output\LRInput' ];                    %����LRͼ��λ��
gOutput_Path=[  gRootFilePath '\Image\Face_ORL\Input&Output\HROuput' ];               %���HRͼ��λ��
gHR_Path=[ gRootFilePath '\Image\Face_ORL\HR_Dictionary' ];             %HR�ֵ�ѵ����ͼ��λ��
gTainingFolder='Training35-1';%----------ѵ���ֵ�Ҫ�޸ĵĵط�------------------------------------------------------------------------------------
gTestingFolder='Testing35-1-5-2';

gAtomNum=1024;  %ѵ���õ��ֵ���ĿΪ����
gLastAns=[];
gSRArray=zeros(int32(gAtomNum));
%% ����ͼ��
rmdir(gInput_Path,'s'); 
mkdir( [gInput_Path] );
copyfile([ gRootFilePath '\Image\Face_ORL\Input&Output\LR.pgm'] ,gInput_Path);
gInputFileName='LR';
%% �������ݿ��Ǩ�ơ���ѵ��������Ŀ35,  ����������Ŀ45
TrainingData_LocalCopy;
% TestingImageReading;
%% ������Ϣ����
global gIter;       %����ָ��
fNumOfIteration=2;      %��������
% fLRPatchSizeArray=[ 3 ];       %����1�������ع�������LR���С
fLRPatchSizeArray=[ 5 5 5 5 ];       %����2������ع���ͬ���С
% fLRPatchSizeArray=[ 10 5 ];       %����3�������� �����ֵ�Ҫ�ֱ�����

%% ���ĵ���
for gIter=1:fNumOfIteration     %������
    % �������ͼ��HR�ֵ䡢LR�ֵ�ķֿ����
    gLRPatchSize=fLRPatchSizeArray( gIter );              %LR������ش�С
%     if gIter==1  %����ǲ���3�����������������������������ע��
%         load('feature-HROrigin-Dic35-1_10.mat'); %���Լ�ѵ���ģ���10
%     else
%         load('feature-HROrigin-Dic10-1_5.mat'); %���Լ�ѵ���ģ���5
%     end
    
    gHRPatchSize=gLRPatchSize*gGap;     %���²���������ͬ
    
    DictionaryDownsampleing;          %2�������²�����LRͼƬ��Ϣ���²������룬���ɴ�ѵ��ͼƬ��
    DictionaryPatchGenerating;      %4������ͼ�����ֵ�ķֿ鲹����ֿ��Լ���֤��Ԥ����
%     Dictionary_Training;          %ͼƬ�Էֿ鲢�����ֵ�ѧϰ--����������Ҫ�ظ����С�
    disp( ['now the gAtomNum is ' num2str(gAtomNum)]);  %�����ǰ���ݿ��С

    %% 5����ʼϡ���ʾ�㷨
    tic;
    for i=1:gRowPatches
        for j=1:gColumnPatches
                fId=(i-1)*gColumnPatches+j;
                SinglePatch_SparseRepresentation(fId);                                      %������ϡ���ʾ
        end
    end
    toc;
    gInputLR=gInputLR(1:gLRRow,1:gLRColumn);                      %����0�����ջأ�������ͼ����任��ԭ��С
    gLastAns=TotalImage_SR_Recovery();                 %6����Ƭ�ϲ�������Ϊ��һ�ν��
end
disp('Done!');
OutputTesting;    
%ʵ�����ȶ�
% Test_ImageRead;             %Just for test;