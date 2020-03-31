%This script is written for Matlab memory testing !

clc;clear;
tic

img0 = imread('E:\±œ…Ë\2.pgm');
img0(114,96)=0;%––¡–≤π¡„
[row column]=size(img0);
row=(row-3)/1+1;
column=(column-3)/1+1;
imshow(img0);

%Testing
All_IMG=[];
for num=1:1024
    disp(num);
    Per_IMG=[];
    for i=1:row
        for j=1:column
            img1=img0(i:i+2,j:j+2);
%             disp((i-1)*column+j)
%             size(img1)
            Per_IMG=[Per_IMG img1];
        end
    end
    All_IMG=[All_IMG;Per_IMG];
end

toc
%3*3 image,each patch will be 9 double variables. Totally there will be 10528 patches.
%so the memory will be occupied about 94752B, which is 92.5KB

% And if the num of images is 1024, the memory will increase to 92.5MB.

