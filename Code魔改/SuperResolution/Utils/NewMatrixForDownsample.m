%This function is written for offerring the new size of the matrix,
%which we are going to downsample.

%�²�����0���¾����С
%row ��
%column ��
%K ģʲô
function [new_Row new_Column]=NewMatrixForDownsample(row,column,K)
    new_Row=row+mod( K-(row-1),K );         %��Ϊ�²����Ǵ�1��ʼ��K��������Ҫ��1��Ȼ�������Ҫ�Ĳ�λ��
    new_Column=column+mod( K-(column-1),K );
end