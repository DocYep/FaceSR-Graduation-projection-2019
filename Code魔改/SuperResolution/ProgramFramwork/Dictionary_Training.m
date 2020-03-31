%Dictionary_Training

global gTrainingNum;             %ѵ����������
global gAtomNum;    %�ֵ�ԭ����

global Dh;
global Dl;

global gLRDictionaryPatch;         %LR�ֵ�鼯
global gHRDictionaryPatch;         %HR�ֵ�鼯

fTrainingHR=[];
fTrainingLR=[];
fLen=size(gHRDictionaryPatch,2);
disp([ 'The Num of Traning Patches is ', num2str(fLen) ]);
fSize=size( gHRDictionaryPatch{1} );
fHRLength=fSize(1)*fSize(2); %�����ֵ�ѵ�������Ժ�ǰ��������HR�ֵ�ĵ�����
disp('Now gaining the Training Data...');
% tic;
for i=1:fLen
    fImg=im2double( gHRDictionaryPatch{i}(:) );
    fTrainingHR(:,i)=fImg;
    fImg=im2double( gLRDictionaryPatch{i}(:) );
    fTrainingLR(:,i)=fImg;
end
% toc;
fDictionary=[ fTrainingHR;fTrainingLR ];    %��ѵ������
clear fTrainingHR;
clear fTrainingLR;

% %����ͼ���
% fTestingHR=[];
% fTestingLR=[];
% fLen=size(gTestingHRPatch,2);
% for i=1:fLen
%     fImg=im2double( gTestingHRPatch{i}(:) );
%     fTestingHR=[ fTestingHR fImg ];
%     fImg=im2double( gTestingLRPatch{i}(:) );
%     fTestingLR=[ fTestingLR fImg ];
% end
% fTesting=[ fTestingHR;fTestingLR ];    %��ѵ������

disp('Now we are going to Training the Dictionary!');
tic;
[Dh Dl]=My_KSVD( fDictionary,fHRLength,20 ); %�õ������Ժ���ֵ�
toc;
save('feature-HROrigin-Dic35-1_10.mat','Dh','Dl');
clear fDictionary;

%Testing For Work
% Dh=fTrainingHR;
% Dl=fTrainingLR;
% gAtomNum=size(fTrainingHR,2);
% disp([ 'The Num of Atom is ',num2str(gAtomNum) ]);
