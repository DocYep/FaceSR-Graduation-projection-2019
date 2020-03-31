%This function is written for offerring the new size of the matrix,
%which we are going to downsample.

%下采样补0的新矩阵大小
%row 行
%column 行
%K 模什么
function [new_Row new_Column]=NewMatrixForDownsample(row,column,K)
    new_Row=row+mod( K-(row-1),K );         %因为下采样是从1开始隔K个，所以要减1，然后加上需要的补位数
    new_Column=column+mod( K-(column-1),K );
end