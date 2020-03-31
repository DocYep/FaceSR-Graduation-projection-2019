%This function is written for Face Image Downsampling, such as blurring HR
%image to LR images.
%The sending param is fImg, the variable of img, rather than filePath.

function    fImgDownsample=SingleImageDownsample(fImg)

    global gGap;                    %下采样提取的间隔，默认3
    [row column]=size(fImg);
    fImgDownsample=fImg(1:gGap:row,:);%隔行采样
    fImgDownsample=fImgDownsample(:,1:gGap:column);%隔列采样
    
%     fH=fspecial('average',3);               %均值滤波器
%     fImgDownsample=imfilter(fImg,fH);      %模糊
%     fImgDownsample = imnoise(fImgDownsample, 'gaussian', 0, 10^2/255^2);%加入高斯白躁声
end
