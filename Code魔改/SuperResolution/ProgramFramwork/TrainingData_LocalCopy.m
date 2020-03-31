%The script is just for moving image files.
%Moving ORL_Face images For Training .

%% 青天已死，黄天当道
global gRootFilePath;           %程序根目录位置
global gHR_Path;
global gTrainingNum;
global gTainingFolder;

% Source_RootPath = 'E:\毕设\Code\Image\Face_ORL\Origin\Training40';                        %源根目录
Source_Path = [ gRootFilePath '\Image\Face_ORL\Origin\' gTainingFolder ];                        %源根目录
rmdir(gHR_Path,'s'); 
mkdir( gHR_Path );
list=ls(Source_Path);
len=length(list);
gTrainingNum=len-2;
copyfile( Source_Path,gHR_Path);
list=ls(Source_Path);
len=length(list);

disp('Training Images Reading Over!');
disp( ['now the gTestingNum is ' num2str(gTestingNum)]);  %输出当前数据库大小