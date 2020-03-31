%This function is written for offerring the new size of the matrix,
%which we are going to Patch Generating.

%�²�����0���¾����С
%---Input Parameter---
%fRow ԭ����
%fColumn ԭ����
%fPatchSize ���С
%fResidual ���С��overlap֮��
%---Output Parameter---
%fNew_Row ��ȫ�������
%fNew_Column ��ȫ�������
%fRowPatches ��ȫ��Ŀ����з���
%fColumnPatches ��ȫ��Ŀ����з���

function [fNew_Row fNew_Column fRowPatches fColumnPatches]=NewMatrixForPatchGenerating( fRow,fColumn,fPatchSize,fResidual )
    
    fColumnRes=mod(fResidual-(fColumn-fPatchSize),fResidual);     %�в���,fResidualҪ�������棬��Ȼ������������0���ͻ����fResidual������һ��
    fNew_Column=fColumn+fColumnRes; 
    fColumnPatches=(fNew_Column-fPatchSize)/fResidual+1;          %һ�еĿ���,�гɼ���patch��ʵ�ʵľ�����
    
    fRowRes=mod(fResidual-(fRow-fPatchSize),fResidual);           %�в���
    fNew_Row=fRow+fRowRes;
    fRowPatches=(fNew_Row-fPatchSize)/fResidual+1;                    %һ�еĿ���,�гɼ���patch��ʵ�ʵľ�����
end