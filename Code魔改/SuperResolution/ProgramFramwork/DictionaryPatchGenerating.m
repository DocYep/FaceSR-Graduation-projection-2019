%This function is written for Input, and LR、HR Dictionary Generating Patches.
    
%基本库
global gGap;                    %下采样提取的间隔，默认2，影响HR、LR块大小
global gTrainingNum;             %训练集人脸数
global gInputLR;                %输入LR图像，要分块对齐
global gLastAns;                %上次迭代输出的结果，不分块只对齐
global gLRDictionary;         %低分辨率字典
global gHRDictionary;        %高分辨率字典

%参数中间使用
global gLRRow;              %LR插值后的大小
global gLRColumn;
global gHRRow;              %HR插值补齐后的大小,输出HR图像要按这个复原!!!
global gHRColumn;
global gLRPatchSize;    %LR块的像素大小
global gHRPatchSize;    %HR块的像素大小
global gOverlapRatio;       %冗余区域比例，默认2/3

%参数结果
global gLROverlap;            %LR块的冗余像素个数
global gHROverlap;            %HR块的冗余像素个数
global gLRResidual;              %LR块大小与LR Overlap之差
global gHRResidual;              %HR块大小与HR Overlap之差
global gRowPatches;                    %  切成几行patch，实际的具体数，补全后的块数行方向
global gColumnPatches;             %  切成几列patch，实际的具体数，补全后的块数列方向
%字典库分块结果
global gInputPatch;         %输入LR图像块集
global gInputFeaPatch;  %输入LR特征图像块集
global gLRDictionaryPatch;         %LR字典块集
global gHRDictionaryPatch;         %HR字典块集

%验证集的相关参数
global gTestingNum;
global gTestingHRImage;   %验证集HR图像内容
global gTestingLRImage;   %验证集LR图像内容
global gTestingHRPatch;            %测试集块路径
global gTestingLRPatch;            %测试集块路径

%计算LR分块的相关参数
disp( ['now the gLRSizePatch is ' num2str(gLRPatchSize)]);  %输出当前LR块大小
disp( ['now the gHRSizePatch is ' num2str(gHRPatchSize)]);  %输出当前LR块大小
gLROverlap=floor(gLRPatchSize*gOverlapRatio);     %LR重叠区域默认以2/3的块大小
gHROverlap=gLROverlap*gGap;     %HR不能直接算对应的2/3，会有浮点数的floor导致的误差，使得后面gColumnPatches大小不一致
gLRResidual=gLRPatchSize-gLROverlap;      %LR块大小与overlap之差
gHRResidual=gLRResidual*gGap;   %理由同上
[ fLRRow,fLRColumn,gRowPatches,gColumnPatches ]=NewMatrixForPatchGenerating( gLRRow,gLRColumn,gLRPatchSize,gLRResidual ); %获得块补全以后的LR图像块大小
[ fHRRow,fHRColumn,gRowPatches,gColumnPatches ]=NewMatrixForPatchGenerating( gHRRow,gHRColumn,gHRPatchSize,gHRResidual ); %获得块补全以后的HR图像块大小

%接下来开始补齐和分块
gInputLR( fLRRow,fLRColumn )=0;%输入图像是LR，按LR大小对齐
gLastAns( fHRRow,fHRColumn )=0;%上次输出结果是HR，按HR大小对齐
disp('The parameters are calculated done!');

%输入图像特征提取效果展示
% figure(2);
% fInputFeature=SingleImageFeatureExtracting(gInputLR);
% for i=1:4
%     subplot(1,4,i),imshow(fInputFeature((i-1)*gLRRow+1:i*gLRRow,:));
% end
% clear fInputFeature;

%输入图像分块，再提特征

gInputPatch={};
gInputFeaPatch={};
fId=0;
for i=1:gRowPatches
    for j=1:gColumnPatches
        fId=fId+1;
        fLeft=1+(j-1)*gLRResidual;
        fRight=fLeft+gLRPatchSize-1;          %算上了左右边界的，所以要减去1才能表示下标
        fTop=1+(i-1)*gLRResidual;
        fBottom=fTop+gLRPatchSize-1;      %算上了左右边界的，所以要减去1才能表示下标
        fPatch=gInputLR(fTop:fBottom,fLeft:fRight);         %The patch. 参数是先行再列
        gInputPatch{fId}=fPatch;
        fPatch=SingleImageFeatureExtracting(fPatch);
        gInputFeaPatch{fId}=fPatch;
    end
end
disp('The InputImage ''s patches are generated done!');

%HR字典分块
fId=0;
gHRDictionaryPatch={};
for fFaceID=1:gTrainingNum
    gHRDictionary{fFaceID}( fHRRow,fHRColumn )=0;%HR元素按分块对齐
        for i=1:gRowPatches
            for j=1:gColumnPatches
                fId=fId+1;
                fLeft=1+(j-1)*gHRResidual;
                fRight=fLeft+gHRPatchSize-1;          %算上了左右边界的，所以要减去1才能表示下标
                fTop=1+(i-1)*gHRResidual;
                fBottom=fTop+gHRPatchSize-1;      %算上了左右边界的，所以要减去1才能表示下标
                fPatch=gHRDictionary{fFaceID}(fTop:fBottom,fLeft:fRight);         %The patch. 参数是先行再列
                gHRDictionaryPatch{fId}=fPatch;
            end
        end
end
disp('The HRDictionary ''s patches are generated done!');

%LR字典分块，再提特征
fId=0;
gLRDictionaryPatch={};
for fFaceID=1:gTrainingNum
    gLRDictionary{fFaceID}( fLRRow,fLRColumn )=0;%HR元素按分块对齐
        for i=1:gRowPatches
            for j=1:gColumnPatches
                fId=fId+1;
                fLeft=1+(j-1)*gLRResidual;
                fRight=fLeft+gLRPatchSize-1;          %算上了左右边界的，所以要减去1才能表示下标
                fTop=1+(i-1)*gLRResidual;
                fBottom=fTop+gLRPatchSize-1;      %算上了左右边界的，所以要减去1才能表示下标
                fPatch=gLRDictionary{fFaceID}(fTop:fBottom,fLeft:fRight);         %The patch. 参数是先行再列
                fPatch=SingleImageFeatureExtracting(fPatch);
                gLRDictionaryPatch{fId}=fPatch;
            end
        end
end
disp('The LRDictionary ''s patches are generated done!');

% %% 验证集的分块情况
% gTestingHRPatch;            %测试集块
% gTestingLRPatch;            %测试集块
% 
% % HR分块
% fId=0;
% gTestingHRPatch={};
% for fFaceID=1:gTestingNum
%     gTestingHRImage{fFaceID}( fHRRow,fHRColumn )=0;%HR元素按分块对齐
%         for i=1:gRowPatches
%             for j=1:gColumnPatches
%                 fId=fId+1;
%                 fLeft=1+(j-1)*gHRResidual;
%                 fRight=fLeft+gHRPatchSize-1;          %算上了左右边界的，所以要减去1才能表示下标
%                 fTop=1+(i-1)*gHRResidual;
%                 fBottom=fTop+gHRPatchSize-1;      %算上了左右边界的，所以要减去1才能表示下标
%                 fPatch=gTestingHRImage{fFaceID}(fTop:fBottom,fLeft:fRight);         %The patch. 参数是先行再列
%                 gTestingHRPatch{fId}=fPatch;
%             end
%         end
% end
% disp('The HRTesting ''s patches are generated done!');
% 
% % LR分块，再提特征
% fId=0;
% gTestingLRPatch={};
% for fFaceID=1:gTestingNum
%     gTestingLRImage{fFaceID}( fLRRow,fLRColumn )=0;%HR元素按分块对齐
%         for i=1:gRowPatches
%             for j=1:gColumnPatches
%                 fId=fId+1;
%                 fLeft=1+(j-1)*gLRResidual;
%                 fRight=fLeft+gLRPatchSize-1;          %算上了左右边界的，所以要减去1才能表示下标
%                 fTop=1+(i-1)*gLRResidual;
%                 fBottom=fTop+gLRPatchSize-1;      %算上了左右边界的，所以要减去1才能表示下标
%                 fPatch=gTestingLRImage{fFaceID}(fTop:fBottom,fLeft:fRight);         %The patch. 参数是先行再列
%                 fPatch=SingleImageFeatureExtracting(fPatch);
%                 gTestingLRPatch{fId}=fPatch;
%             end
%         end
% end
% disp('The LRTesting ''s patches are generated done!');