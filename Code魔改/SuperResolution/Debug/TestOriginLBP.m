%图像读取和显示Demo

%%  LBP算法实现

    function TestOriginLBP()
    
    img=imread('E:\毕设\Code\Image\Face_ORL\Input&Output\LR.pgm');
    subplot(1,2,1),imshow(img);
        
        imgSize = size(img);
        if numel(imgSize) > 2
            imgG = rgb2gray(img);
        else
            imgG = img;
        end
        [rows, cols] = size(imgG);
        rows=int16(rows);
        cols=int16(cols);
        imglbp = uint8(zeros(rows-2, cols-2));  

        for i=2:rows-2
            for j=2:cols-2
                center = imgG(i,j);
                lbpCode = 0;
                lbpCode = bitor(lbpCode, (bitshift(compareCenter(i-1, j-1, center, imgG), 7)));
                lbpCode = bitor(lbpCode, (bitshift(compareCenter(i-1,j, center, imgG), 6)));
                lbpCode = bitor(lbpCode, (bitshift(compareCenter(i-1,j+1, center, imgG), 5)));
                lbpCode = bitor(lbpCode, (bitshift(compareCenter(i,j+1, center, imgG), 4)));
                lbpCode = bitor(lbpCode, (bitshift(compareCenter(i+1,j+1, center, imgG), 3)));
                lbpCode = bitor(lbpCode, (bitshift(compareCenter(i+1,j, center, imgG), 2)));
                lbpCode = bitor(lbpCode, (bitshift(compareCenter(i+1,j-1, center, imgG), 1)));
                lbpCode = bitor(lbpCode, (bitshift(compareCenter(i, j-1, center, imgG), 0)));
                imglbp(i-1,j-1) = lbpCode;
            end
        end

            subplot(1,2,2),imshow(imglbp);
    end

    function flag = compareCenter(x, y, center, imgG)
        if imgG(x, y) > center 
            flag = 1;
        else
            flag = 0;
        end
    end
