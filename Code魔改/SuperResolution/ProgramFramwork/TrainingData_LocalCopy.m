%The script is just for moving image files.
%Moving ORL_Face images For Training .

%% �������������쵱��
global gRootFilePath;           %�����Ŀ¼λ��
global gHR_Path;
global gTrainingNum;
global gTainingFolder;

% Source_RootPath = 'E:\����\Code\Image\Face_ORL\Origin\Training40';                        %Դ��Ŀ¼
Source_Path = [ gRootFilePath '\Image\Face_ORL\Origin\' gTainingFolder ];                        %Դ��Ŀ¼
rmdir(gHR_Path,'s'); 
mkdir( gHR_Path );
list=ls(Source_Path);
len=length(list);
gTrainingNum=len-2;
copyfile( Source_Path,gHR_Path);
list=ls(Source_Path);
len=length(list);

disp('Training Images Reading Over!');
disp( ['now the gTestingNum is ' num2str(gTestingNum)]);  %�����ǰ���ݿ��С