%This script is written for images PSNR and others testing .

global gFormat;
global gInput_Path;
global gOutput_Path;
global gInputFileName;
global gInputLR;
global gHRRow;              %HR��ֵ�����Ĵ�С,���HRͼ��Ҫ�������ԭ!!!
global gHRColumn;
global gIter;
len=gIter;

img0=imread( [gInput_Path,'\',gInputFileName,gFormat] );
img0( gHRRow,gHRColumn )=0;
subplot(1,len+3,1),imshow(img0),title('ԭʼHRͼ��');
img1=gInputLR;
subplot(1,len+3,2),imshow(img1),title('�²���LRͼ��');

PSNR_Array=[];
for i=1:len
    img2=imread( [gOutput_Path,'\LR',num2str(i),'.pgm'] );
    subplot(1,len+3,i+2),imshow(img2),title(['�ع�',num2str(i),'th  HRͼ��']);
        
     [psnr mse]=PSNR(img0,img2);
    PSNR_Array=[ PSNR_Array,psnr ];
end

disp(PSNR_Array);

fSize=size(img0);
bicubicImg = imresize(img1, fSize, 'bicubic');
subplot(1,len+3,len+3),imshow(bicubicImg),title('˫���β�ֵ');
[psnr mse]=PSNR(img0,bicubicImg);
disp('And the bicubicImg''s PSNR is ');
disp(psnr);

% figure();
% Residual=D-Dictionary*x;    %Calculate the residual between the data and ans, which is determined by OMP and Dictionary that was gained by KSVD.
% SumOfResidual=sum(Residual);            %Gain the sum of the residual. ������ܲ����ף�Ӧ��Ҫ��RMSE���� 
%  RMSE = sqrt( sum( Residual.^2) / 50 ); 
%  
%  
 
