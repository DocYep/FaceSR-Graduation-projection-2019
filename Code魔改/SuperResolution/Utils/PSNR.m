 function [PSNR, MSE] = PSNR(X, Y)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % �����ֵ�����PSNR
    % ��RGBת��YCbCr��ʽ���м���
    % ���ֱ�Ӽ�����ת�����ֵҪС2dB���ң���Ȼ�Ǹ�����ԣ�
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
     if size(X,3)~=1   %�ж�ͼ��ʱ���ǲ�ɫͼ������ǣ����Ϊ3������Ϊ1
       org=rgb2ycbcr(X);
       test=rgb2ycbcr(Y);
       Y1=org(:,:,1);
       Y2=test(:,:,1);
       Y1=double(Y1);  %����ƽ��ʱ����Ҫת��double���ͣ�����uchar���ͻᶪʧ����
       Y2=double(Y2);
     else              %�Ҷ�ͼ�񣬲���ת��
         Y1=double(X);
         Y2=double(Y);
     end

    if nargin<2    
       D = Y1;
    else
      if any(size(Y1)~=size(Y2))
        error('The input size is not equal to each other!');
      end
     D = Y1 - Y2; 
    end
    MSE = sum(D(:).*D(:)) / numel(Y1); 
    PSNR = 10*log10(255^2 / MSE);
 end