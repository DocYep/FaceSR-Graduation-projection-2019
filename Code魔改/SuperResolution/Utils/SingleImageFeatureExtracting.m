%This function is written for single image's feature extracting.
%However, the algorithnm needs to be rewritten!

function fImgFeature=SingleImageFeatureExtracting(fImg)
    
    fImgFeature=[];
%     fImgFeature=[fImgFeature;fImg];
    filter1=[-1,0,1];            %一阶导数    行向量则列方向特征明显，列向量则行方向特征明显
    filter2=[1,0,-2,0,1];       %二阶导数    同上
    PerFeature=imfilter(fImg,filter1);
    fImgFeature=[fImgFeature;PerFeature];
    PerFeature=imfilter(fImg,filter1');
    fImgFeature=[fImgFeature;PerFeature];
    PerFeature=imfilter(fImg,filter2);
    fImgFeature=[fImgFeature;PerFeature];
    PerFeature=imfilter(fImg,filter2');
    fImgFeature=[fImgFeature;PerFeature];

end