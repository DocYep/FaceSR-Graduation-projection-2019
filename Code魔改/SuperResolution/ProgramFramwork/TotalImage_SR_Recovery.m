%This script is written for scraping all the patches , to gain the whole
%output image.

%Attention!    DOUBLE format of the image , not UINT8.
function fImg=TotalImage_SR_Recovery()        %�����������в����ջ�

    global gInputLR;                    %����LRͼ������ȫ��Լ��
    global gInputFileName;      %����LRͼ���ļ���
    global gOutput_Path;         %���HRͼ��λ��
    global gFormat;     
    
    global gHRRow;              %HR��ֵ�����Ĵ�С,���HRͼ��Ҫ�������ԭ!!!
    global gHRColumn;
    
    global gRowPatches;                    %  �гɼ���patch��ʵ�ʵľ���������ȫ��Ŀ����з���
    global gColumnPatches;             %  �гɼ���patch��ʵ�ʵľ���������ȫ��Ŀ����з���

    global gIter;       %����ָ��
    global gSRPatch;    %�ع�����SRͼ���
    global gHROverlap;            %HR����������ظ���
    
    fImg=[];
    %����ƴ��ƴ��������ƴ��
    for i=1:gRowPatches
        fEach_Row=[];
        for j=1:gColumnPatches      %Merge the column
            fID=(i-1)*gColumnPatches+j;
            fPatch=im2double( gSRPatch{fID} );        %ת��double��ʽ,��ֹ����fImg���
            if j==1     %First column
                fEach_Row=fPatch;
            else
                fEach_Row=[ fEach_Row(:,1:end-gHROverlap),(fEach_Row(:,end-gHROverlap+1:end)+fPatch(:,1:gHROverlap))/2,fPatch(:,gHROverlap+1:end) ];     %�ϲ��ұ�
            end
        end
        %Merge the row
        if i==1     %First column
            fImg=fEach_Row;
        else
%             imshow(fImg);%Debug
            fImg=[ fImg(1:end-gHROverlap,:);(fImg(end-gHROverlap+1:end,:)+fEach_Row(1:gHROverlap,:))/2;fEach_Row(gHROverlap+1:end,:) ];     %�ϲ��ϱ�,��ƴ���÷ֺ�
        end
    end
    
     %����0�����ջ�,�޸Ĺ�δȷ�ϣ�����������������������������������
    fImg=fImg(1:gHRRow,1:gHRColumn);
    
    %ȫ��Լ��
    fImg=backprojection(fImg, im2double(gInputLR), 20);
    
    %д�����
%     figure(),imshow(fImg);
    fImg=im2uint8(fImg);        %ת������ԭ����uint8��ʽ
    imwrite( fImg, [gOutput_Path,'\',gInputFileName,num2str(gIter),gFormat] );
end