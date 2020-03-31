%This script is written for Testing Image Reading

global gTestingNum;
global gRootFilePath;           %�����Ŀ¼λ��
global gTestingHRImage;   %��֤��HRͼ������
global gTestingLRImage;   %��֤��LRͼ������
global gTestingFolder;

Source_RootPath=[ gRootFilePath '\Image\Face_ORL\Origin\' gTestingFolder ];        %���Լ�ͼ��Ŀ¼
list=ls(Source_RootPath);
len=length(list);
gTestingNum=len-2;
gTestingHRImage={};
gTestingLRImage={};
for i=3:len
        fImg=imread([ Source_RootPath,'\', list(i,:) ]);
        gTestingHRImage{i-2}=fImg;
        fImg=SingleImageDownsample(fImg);                            %HRͼ���ֵ
        gTestingLRImage{i-2}=fImg;
end
disp('Testing Images Reading Over!');
disp( ['now the gTrainingNum is ' num2str(gTrainingNum)]);  %�����ǰ���ݿ��С