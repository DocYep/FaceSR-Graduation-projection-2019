%This function is written for Input, and LR��HR Dictionary Generating Patches.
    
%������
global gGap;                    %�²�����ȡ�ļ����Ĭ��2��Ӱ��HR��LR���С
global gTrainingNum;             %ѵ����������
global gInputLR;                %����LRͼ��Ҫ�ֿ����
global gLastAns;                %�ϴε�������Ľ�������ֿ�ֻ����
global gLRDictionary;         %�ͷֱ����ֵ�
global gHRDictionary;        %�߷ֱ����ֵ�

%�����м�ʹ��
global gLRRow;              %LR��ֵ��Ĵ�С
global gLRColumn;
global gHRRow;              %HR��ֵ�����Ĵ�С,���HRͼ��Ҫ�������ԭ!!!
global gHRColumn;
global gLRPatchSize;    %LR������ش�С
global gHRPatchSize;    %HR������ش�С
global gOverlapRatio;       %�������������Ĭ��2/3

%�������
global gLROverlap;            %LR����������ظ���
global gHROverlap;            %HR����������ظ���
global gLRResidual;              %LR���С��LR Overlap֮��
global gHRResidual;              %HR���С��HR Overlap֮��
global gRowPatches;                    %  �гɼ���patch��ʵ�ʵľ���������ȫ��Ŀ����з���
global gColumnPatches;             %  �гɼ���patch��ʵ�ʵľ���������ȫ��Ŀ����з���
%�ֵ��ֿ���
global gInputPatch;         %����LRͼ��鼯
global gInputFeaPatch;  %����LR����ͼ��鼯
global gLRDictionaryPatch;         %LR�ֵ�鼯
global gHRDictionaryPatch;         %HR�ֵ�鼯

%��֤������ز���
global gTestingNum;
global gTestingHRImage;   %��֤��HRͼ������
global gTestingLRImage;   %��֤��LRͼ������
global gTestingHRPatch;            %���Լ���·��
global gTestingLRPatch;            %���Լ���·��

%����LR�ֿ����ز���
disp( ['now the gLRSizePatch is ' num2str(gLRPatchSize)]);  %�����ǰLR���С
disp( ['now the gHRSizePatch is ' num2str(gHRPatchSize)]);  %�����ǰLR���С
gLROverlap=floor(gLRPatchSize*gOverlapRatio);     %LR�ص�����Ĭ����2/3�Ŀ��С
gHROverlap=gLROverlap*gGap;     %HR����ֱ�����Ӧ��2/3�����и�������floor���µ���ʹ�ú���gColumnPatches��С��һ��
gLRResidual=gLRPatchSize-gLROverlap;      %LR���С��overlap֮��
gHRResidual=gLRResidual*gGap;   %����ͬ��
[ fLRRow,fLRColumn,gRowPatches,gColumnPatches ]=NewMatrixForPatchGenerating( gLRRow,gLRColumn,gLRPatchSize,gLRResidual ); %��ÿ鲹ȫ�Ժ��LRͼ����С
[ fHRRow,fHRColumn,gRowPatches,gColumnPatches ]=NewMatrixForPatchGenerating( gHRRow,gHRColumn,gHRPatchSize,gHRResidual ); %��ÿ鲹ȫ�Ժ��HRͼ����С

%��������ʼ����ͷֿ�
gInputLR( fLRRow,fLRColumn )=0;%����ͼ����LR����LR��С����
gLastAns( fHRRow,fHRColumn )=0;%�ϴ���������HR����HR��С����
disp('The parameters are calculated done!');

%����ͼ��������ȡЧ��չʾ
% figure(2);
% fInputFeature=SingleImageFeatureExtracting(gInputLR);
% for i=1:4
%     subplot(1,4,i),imshow(fInputFeature((i-1)*gLRRow+1:i*gLRRow,:));
% end
% clear fInputFeature;

%����ͼ��ֿ飬��������

gInputPatch={};
gInputFeaPatch={};
fId=0;
for i=1:gRowPatches
    for j=1:gColumnPatches
        fId=fId+1;
        fLeft=1+(j-1)*gLRResidual;
        fRight=fLeft+gLRPatchSize-1;          %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
        fTop=1+(i-1)*gLRResidual;
        fBottom=fTop+gLRPatchSize-1;      %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
        fPatch=gInputLR(fTop:fBottom,fLeft:fRight);         %The patch. ��������������
        gInputPatch{fId}=fPatch;
        fPatch=SingleImageFeatureExtracting(fPatch);
        gInputFeaPatch{fId}=fPatch;
    end
end
disp('The InputImage ''s patches are generated done!');

%HR�ֵ�ֿ�
fId=0;
gHRDictionaryPatch={};
for fFaceID=1:gTrainingNum
    gHRDictionary{fFaceID}( fHRRow,fHRColumn )=0;%HRԪ�ذ��ֿ����
        for i=1:gRowPatches
            for j=1:gColumnPatches
                fId=fId+1;
                fLeft=1+(j-1)*gHRResidual;
                fRight=fLeft+gHRPatchSize-1;          %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
                fTop=1+(i-1)*gHRResidual;
                fBottom=fTop+gHRPatchSize-1;      %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
                fPatch=gHRDictionary{fFaceID}(fTop:fBottom,fLeft:fRight);         %The patch. ��������������
                gHRDictionaryPatch{fId}=fPatch;
            end
        end
end
disp('The HRDictionary ''s patches are generated done!');

%LR�ֵ�ֿ飬��������
fId=0;
gLRDictionaryPatch={};
for fFaceID=1:gTrainingNum
    gLRDictionary{fFaceID}( fLRRow,fLRColumn )=0;%HRԪ�ذ��ֿ����
        for i=1:gRowPatches
            for j=1:gColumnPatches
                fId=fId+1;
                fLeft=1+(j-1)*gLRResidual;
                fRight=fLeft+gLRPatchSize-1;          %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
                fTop=1+(i-1)*gLRResidual;
                fBottom=fTop+gLRPatchSize-1;      %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
                fPatch=gLRDictionary{fFaceID}(fTop:fBottom,fLeft:fRight);         %The patch. ��������������
                fPatch=SingleImageFeatureExtracting(fPatch);
                gLRDictionaryPatch{fId}=fPatch;
            end
        end
end
disp('The LRDictionary ''s patches are generated done!');

% %% ��֤���ķֿ����
% gTestingHRPatch;            %���Լ���
% gTestingLRPatch;            %���Լ���
% 
% % HR�ֿ�
% fId=0;
% gTestingHRPatch={};
% for fFaceID=1:gTestingNum
%     gTestingHRImage{fFaceID}( fHRRow,fHRColumn )=0;%HRԪ�ذ��ֿ����
%         for i=1:gRowPatches
%             for j=1:gColumnPatches
%                 fId=fId+1;
%                 fLeft=1+(j-1)*gHRResidual;
%                 fRight=fLeft+gHRPatchSize-1;          %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
%                 fTop=1+(i-1)*gHRResidual;
%                 fBottom=fTop+gHRPatchSize-1;      %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
%                 fPatch=gTestingHRImage{fFaceID}(fTop:fBottom,fLeft:fRight);         %The patch. ��������������
%                 gTestingHRPatch{fId}=fPatch;
%             end
%         end
% end
% disp('The HRTesting ''s patches are generated done!');
% 
% % LR�ֿ飬��������
% fId=0;
% gTestingLRPatch={};
% for fFaceID=1:gTestingNum
%     gTestingLRImage{fFaceID}( fLRRow,fLRColumn )=0;%HRԪ�ذ��ֿ����
%         for i=1:gRowPatches
%             for j=1:gColumnPatches
%                 fId=fId+1;
%                 fLeft=1+(j-1)*gLRResidual;
%                 fRight=fLeft+gLRPatchSize-1;          %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
%                 fTop=1+(i-1)*gLRResidual;
%                 fBottom=fTop+gLRPatchSize-1;      %���������ұ߽�ģ�����Ҫ��ȥ1���ܱ�ʾ�±�
%                 fPatch=gTestingLRImage{fFaceID}(fTop:fBottom,fLeft:fRight);         %The patch. ��������������
%                 fPatch=SingleImageFeatureExtracting(fPatch);
%                 gTestingLRPatch{fId}=fPatch;
%             end
%         end
% end
% disp('The LRTesting ''s patches are generated done!');