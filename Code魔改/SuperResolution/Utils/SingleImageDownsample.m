%This function is written for Face Image Downsampling, such as blurring HR
%image to LR images.
%The sending param is fImg, the variable of img, rather than filePath.

function    fImgDownsample=SingleImageDownsample(fImg)

    global gGap;                    %�²�����ȡ�ļ����Ĭ��3
    [row column]=size(fImg);
    fImgDownsample=fImg(1:gGap:row,:);%���в���
    fImgDownsample=fImgDownsample(:,1:gGap:column);%���в���
    
%     fH=fspecial('average',3);               %��ֵ�˲���
%     fImgDownsample=imfilter(fImg,fH);      %ģ��
%     fImgDownsample = imnoise(fImgDownsample, 'gaussian', 0, 10^2/255^2);%�����˹������
end
