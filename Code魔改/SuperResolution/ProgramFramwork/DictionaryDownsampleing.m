%DictionaryDownsampleing
%对输入测试图像、HR与LR字典的读取，LR化并返回LR图像大小

%% 输入图像处理、HR与LR图像的计算
global gFormat;
global gInputLR;                    %输入LR图像
global gInputFileName;
global gInput_Path;

global gHRRow;              %HR插值补齐后的大小
global gHRColumn;
global gLRRow;              %LR插值后的大小
global gLRColumn;

gInputLR=imread( [gInput_Path,'\',gInputFileName,gFormat] );        %读取输入图像
[ fRow fColumn ]=size(gInputLR);
[ gHRRow gHRColumn ]=NewMatrixForDownsample(fRow,fColumn,gGap);%补采样齐后的大小，输出HR图像要按这个复原!!!

gInputLR(gHRRow,gHRColumn)=0;       %下采样补齐
% figure(1),subplot(1,3,1),imshow(gInputLR),title('原输入HR图像');
gInputLR=SingleImageDownsample(gInputLR);                            %输入图像插值
% subplot(1,3,2),imshow(gInputLR),title('下采样输入LR图像');
[ gLRRow gLRColumn ]=size(gInputLR);        %插值后LR的大小
bicubicImg = imresize(gInputLR, [fRow, fColumn], 'bicubic');
% subplot(1,3,3),imshow(bicubicImg),title('双三次插值放大LR图像');
disp([ 'Have Input Image Downsample Processing done!' ]);

%% HR字典读取
global gHR_Path;            %高分辨率字典路径
global gHRDictionary;            %高分辨率字典
global gLRDictionary;            %低分辨率字典
global gTrainingNum;           %人脸数

gHRDictionary={};
gLRDictionary={};
for i=1:gTrainingNum 
    fFileName=[num2str(i),gFormat];
    fSourcePath=[gHR_Path,'\',fFileName];
    fImg=imread(fSourcePath);        %Read Image
    fImg( gHRRow,gHRColumn )=0;     %HR插值补齐
    
    gHRDictionary{i}=fImg;
    fImg=SingleImageDownsample(fImg);                            %HR图像插值
    gLRDictionary{i}=fImg;
    disp([ 'Have ',num2str(i),' Dictionary Downsample Processing done!' ]);
end