%This script is written for scraping all the patches , to gain the whole
%output image.

%Attention!    DOUBLE format of the image , not UINT8.
function fImg=TotalImage_SR_Recovery()        %参数用于行列补齐收回

    global gInputLR;                    %输入LR图像，用于全局约束
    global gInputFileName;      %输入LR图像文件名
    global gOutput_Path;         %输出HR图像位置
    global gFormat;     
    
    global gHRRow;              %HR插值补齐后的大小,输出HR图像要按这个复原!!!
    global gHRColumn;
    
    global gRowPatches;                    %  切成几行patch，实际的具体数，补全后的块数行方向
    global gColumnPatches;             %  切成几列patch，实际的具体数，补全后的块数列方向

    global gIter;       %迭代指针
    global gSRPatch;    %重构出的SR图像块
    global gHROverlap;            %HR块的冗余像素个数
    
    fImg=[];
    %列先拼，拼好了整体拼行
    for i=1:gRowPatches
        fEach_Row=[];
        for j=1:gColumnPatches      %Merge the column
            fID=(i-1)*gColumnPatches+j;
            fPatch=im2double( gSRPatch{fID} );        %转换double格式,防止后续fImg溢出
            if j==1     %First column
                fEach_Row=fPatch;
            else
                fEach_Row=[ fEach_Row(:,1:end-gHROverlap),(fEach_Row(:,end-gHROverlap+1:end)+fPatch(:,1:gHROverlap))/2,fPatch(:,gHROverlap+1:end) ];     %合并右边
            end
        end
        %Merge the row
        if i==1     %First column
            fImg=fEach_Row;
        else
%             imshow(fImg);%Debug
            fImg=[ fImg(1:end-gHROverlap,:);(fImg(end-gHROverlap+1:end,:)+fEach_Row(1:gHROverlap,:))/2;fEach_Row(gHROverlap+1:end,:) ];     %合并上边,行拼接用分号
        end
    end
    
     %行列0补齐收回,修改过未确认！！！！！！！！！！！【】【】【】】
    fImg=fImg(1:gHRRow,1:gHRColumn);
    
    %全局约束
    fImg=backprojection(fImg, im2double(gInputLR), 20);
    
    %写出结果
%     figure(),imshow(fImg);
    fImg=im2uint8(fImg);        %转换回来原来的uint8格式
    imwrite( fImg, [gOutput_Path,'\',gInputFileName,num2str(gIter),gFormat] );
end