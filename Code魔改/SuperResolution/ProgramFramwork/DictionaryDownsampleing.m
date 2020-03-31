%DictionaryDownsampleing
%���������ͼ��HR��LR�ֵ�Ķ�ȡ��LR��������LRͼ���С

%% ����ͼ����HR��LRͼ��ļ���
global gFormat;
global gInputLR;                    %����LRͼ��
global gInputFileName;
global gInput_Path;

global gHRRow;              %HR��ֵ�����Ĵ�С
global gHRColumn;
global gLRRow;              %LR��ֵ��Ĵ�С
global gLRColumn;

gInputLR=imread( [gInput_Path,'\',gInputFileName,gFormat] );        %��ȡ����ͼ��
[ fRow fColumn ]=size(gInputLR);
[ gHRRow gHRColumn ]=NewMatrixForDownsample(fRow,fColumn,gGap);%���������Ĵ�С�����HRͼ��Ҫ�������ԭ!!!

gInputLR(gHRRow,gHRColumn)=0;       %�²�������
% figure(1),subplot(1,3,1),imshow(gInputLR),title('ԭ����HRͼ��');
gInputLR=SingleImageDownsample(gInputLR);                            %����ͼ���ֵ
% subplot(1,3,2),imshow(gInputLR),title('�²�������LRͼ��');
[ gLRRow gLRColumn ]=size(gInputLR);        %��ֵ��LR�Ĵ�С
bicubicImg = imresize(gInputLR, [fRow, fColumn], 'bicubic');
% subplot(1,3,3),imshow(bicubicImg),title('˫���β�ֵ�Ŵ�LRͼ��');
disp([ 'Have Input Image Downsample Processing done!' ]);

%% HR�ֵ��ȡ
global gHR_Path;            %�߷ֱ����ֵ�·��
global gHRDictionary;            %�߷ֱ����ֵ�
global gLRDictionary;            %�ͷֱ����ֵ�
global gTrainingNum;           %������

gHRDictionary={};
gLRDictionary={};
for i=1:gTrainingNum 
    fFileName=[num2str(i),gFormat];
    fSourcePath=[gHR_Path,'\',fFileName];
    fImg=imread(fSourcePath);        %Read Image
    fImg( gHRRow,gHRColumn )=0;     %HR��ֵ����
    
    gHRDictionary{i}=fImg;
    fImg=SingleImageDownsample(fImg);                            %HRͼ���ֵ
    gLRDictionary{i}=fImg;
    disp([ 'Have ',num2str(i),' Dictionary Downsample Processing done!' ]);
end