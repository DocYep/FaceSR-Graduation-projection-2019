%This function is written for offerring the new size of the matrix,
%which we are going to Patch Generating.

%下采样补0的新矩阵大小
%---Input Parameter---
%fRow 原行数
%fColumn 原列数
%fPatchSize 块大小
%fResidual 块大小与overlap之差
%---Output Parameter---
%fNew_Row 补全后的行数
%fNew_Column 补全后的列数
%fRowPatches 补全后的块数行方向
%fColumnPatches 补全后的块数列方向

function [fNew_Row fNew_Column fRowPatches fColumnPatches]=NewMatrixForPatchGenerating( fRow,fColumn,fPatchSize,fResidual )
    
    fColumnRes=mod(fResidual-(fColumn-fPatchSize),fResidual);     %列不足,fResidual要放在里面，不然如果后面减得是0，就会加上fResidual，大了一倍
    fNew_Column=fColumn+fColumnRes; 
    fColumnPatches=(fNew_Column-fPatchSize)/fResidual+1;          %一行的块数,切成几列patch，实际的具体数
    
    fRowRes=mod(fResidual-(fRow-fPatchSize),fResidual);           %行不足
    fNew_Row=fRow+fRowRes;
    fRowPatches=(fNew_Row-fPatchSize)/fResidual+1;                    %一列的块数,切成几行patch，实际的具体数
end