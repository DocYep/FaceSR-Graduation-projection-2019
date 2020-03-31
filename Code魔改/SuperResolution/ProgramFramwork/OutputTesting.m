%This script is written for images PSNR and others testing .

global gFormat;
global gInput_Path;
global gOutput_Path;
global gInputFileName;
global gInputLR;
global gHRRow;              %HR插值补齐后的大小,输出HR图像要按这个复原!!!
global gHRColumn;
global gIter;
len=gIter;

img0=imread( [gInput_Path,'\',gInputFileName,gFormat] );
img0( gHRRow,gHRColumn )=0;
subplot(1,len+3,1),imshow(img0),title('原始HR图像');
img1=gInputLR;
subplot(1,len+3,2),imshow(img1),title('下采样LR图像');

PSNR_Array=[];
for i=1:len
    img2=imread( [gOutput_Path,'\LR',num2str(i),'.pgm'] );
    subplot(1,len+3,i+2),imshow(img2),title(['重构',num2str(i),'th  HR图像']);
        
     [psnr mse]=PSNR(img0,img2);
    PSNR_Array=[ PSNR_Array,psnr ];
end

disp(PSNR_Array);

fSize=size(img0);
bicubicImg = imresize(img1, fSize, 'bicubic');
subplot(1,len+3,len+3),imshow(bicubicImg),title('双三次插值');
[psnr mse]=PSNR(img0,bicubicImg);
disp('And the bicubicImg''s PSNR is ');
disp(psnr);

% figure();
% Residual=D-Dictionary*x;    %Calculate the residual between the data and ans, which is determined by OMP and Dictionary that was gained by KSVD.
% SumOfResidual=sum(Residual);            %Gain the sum of the residual. 这个可能不靠谱，应该要用RMSE来看 
%  RMSE = sqrt( sum( Residual.^2) / 50 ); 
%  
%  
 
