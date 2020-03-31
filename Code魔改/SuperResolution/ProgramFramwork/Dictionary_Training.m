%Dictionary_Training

global gTrainingNum;             %训练集人脸数
global gAtomNum;    %字典原子数

global Dh;
global Dl;

global gLRDictionaryPatch;         %LR字典块集
global gHRDictionaryPatch;         %HR字典块集

fTrainingHR=[];
fTrainingLR=[];
fLen=size(gHRDictionaryPatch,2);
disp([ 'The Num of Traning Patches is ', num2str(fLen) ]);
fSize=size( gHRDictionaryPatch{1} );
fHRLength=fSize(1)*fSize(2); %联合字典训练结束以后，前多少行是HR字典的的内容
disp('Now gaining the Training Data...');
% tic;
for i=1:fLen
    fImg=im2double( gHRDictionaryPatch{i}(:) );
    fTrainingHR(:,i)=fImg;
    fImg=im2double( gLRDictionaryPatch{i}(:) );
    fTrainingLR(:,i)=fImg;
end
% toc;
fDictionary=[ fTrainingHR;fTrainingLR ];    %待训练数据
clear fTrainingHR;
clear fTrainingLR;

% %测试图像块
% fTestingHR=[];
% fTestingLR=[];
% fLen=size(gTestingHRPatch,2);
% for i=1:fLen
%     fImg=im2double( gTestingHRPatch{i}(:) );
%     fTestingHR=[ fTestingHR fImg ];
%     fImg=im2double( gTestingLRPatch{i}(:) );
%     fTestingLR=[ fTestingLR fImg ];
% end
% fTesting=[ fTestingHR;fTestingLR ];    %待训练数据

disp('Now we are going to Training the Dictionary!');
tic;
[Dh Dl]=My_KSVD( fDictionary,fHRLength,20 ); %得到更新以后的字典
toc;
save('feature-HROrigin-Dic35-1_10.mat','Dh','Dl');
clear fDictionary;

%Testing For Work
% Dh=fTrainingHR;
% Dl=fTrainingLR;
% gAtomNum=size(fTrainingHR,2);
% disp([ 'The Num of Atom is ',num2str(gAtomNum) ]);
