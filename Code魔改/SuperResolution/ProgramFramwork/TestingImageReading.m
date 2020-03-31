%This script is written for Testing Image Reading

global gTestingNum;
global gRootFilePath;           %程序根目录位置
global gTestingHRImage;   %验证集HR图像内容
global gTestingLRImage;   %验证集LR图像内容
global gTestingFolder;

Source_RootPath=[ gRootFilePath '\Image\Face_ORL\Origin\' gTestingFolder ];        %测试集图像目录
list=ls(Source_RootPath);
len=length(list);
gTestingNum=len-2;
gTestingHRImage={};
gTestingLRImage={};
for i=3:len
        fImg=imread([ Source_RootPath,'\', list(i,:) ]);
        gTestingHRImage{i-2}=fImg;
        fImg=SingleImageDownsample(fImg);                            %HR图像插值
        gTestingLRImage{i-2}=fImg;
end
disp('Testing Images Reading Over!');
disp( ['now the gTrainingNum is ' num2str(gTrainingNum)]);  %输出当前数据库大小