%This function is written for single image's feature extracting.
%However, the algorithnm needs to be rewritten!

function fImgFeature=SingleImageFeatureExtracting(fImg)
    
    fImgFeature=[];
%     fImgFeature=[fImgFeature;fImg];
    filter1=[-1,0,1];            %һ�׵���    ���������з����������ԣ����������з�����������
    filter2=[1,0,-2,0,1];       %���׵���    ͬ��
    PerFeature=imfilter(fImg,filter1);
    fImgFeature=[fImgFeature;PerFeature];
    PerFeature=imfilter(fImg,filter1');
    fImgFeature=[fImgFeature;PerFeature];
    PerFeature=imfilter(fImg,filter2);
    fImgFeature=[fImgFeature;PerFeature];
    PerFeature=imfilter(fImg,filter2');
    fImgFeature=[fImgFeature;PerFeature];

end