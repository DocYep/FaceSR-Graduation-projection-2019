function [Dictionary,output] = KSVD(...
    Data,... % an nXN matrix that contins N signals (Y), each of dimension n.
    param)
% =========================================================================
%                          K-SVD algorithm
% =========================================================================
% The K-SVD algorithm finds a dictionary for linear representation of
% signals. Given a set of signals, it searches for the best dictionary that
% can sparsely represent each signal. Detailed discussion on the algorithm
% and possible applications can be found in "The K-SVD: An Algorithm for 
% Designing of Overcomplete Dictionaries for Sparse Representation", written
% by M. Aharon, M. Elad, and A.M. Bruckstein and appeared in the IEEE Trans. 
% On Signal Processing, Vol. 54, no. 11, pp. 4311-4322, November 2006. 
% =========================================================================
% INPUT ARGUMENTS:
% Data                         an nXN matrix that contins N signals (Y), each of dimension n. 
% param                        structure that includes all required
%                                 parameters for the K-SVD execution.
%                                 Required fields are:
%    K, ...                    the number of dictionary elements to train
%    numIteration,...          number of iterations to perform.
%    errorFlag...              if =0, a fix number of coefficients is
%                                 used for representation of each signal. If so, param.L must be
%                                 specified as the number of representing atom. if =1, arbitrary number
%                                 of atoms represent each signal, until a specific representation error
%                                 is reached. If so, param.errorGoal must be specified as the allowed
%                                 error.
%    preserveDCAtom...         if =1 then the first atom in the dictionary
%                                 is set to be constant, and does not ever change. This
%                                 might be useful for working with natural
%                                 images (in this case, only param.K-1
%                                 atoms are trained).
%    (optional, see errorFlag) L,...                 % maximum coefficients to use in OMP coefficient calculations.
%    (optional, see errorFlag) errorGoal, ...        % allowed representation error in representing each signal.
%    InitializationMethod,...  mehtod to initialize the dictionary, can
%                                 be one of the following arguments: 
%                                 * 'DataElements' (initialization by the signals themselves), or: 
%                                 * 'GivenMatrix' (initialization by a given matrix param.initialDictionary).
%    (optional, see InitializationMethod) initialDictionary,...      % if the initialization method 
%                                 is 'GivenMatrix', this is the matrix that will be used.
%    (optional) TrueDictionary, ...        % if specified, in each
%                                 iteration the difference between this dictionary and the trained one
%                                 is measured and displayed.
%    displayProgress, ...      if =1 progress information is displyed. If param.errorFlag==0, 
%                                 the average repersentation error (RMSE) is displayed, while if 
%                                 param.errorFlag==1, the average number of required coefficients for 
%                                 representation of each signal is displayed.
% =========================================================================
% OUTPUT ARGUMENTS:
%  Dictionary                  The extracted dictionary of size nX(param.K).
%  output                      Struct that contains information about the current run. It may include the following fields:
%    CoefMatrix                  The final coefficients matrix (it should hold that Data equals approximately Dictionary*output.CoefMatrix.
%    ratio                       If the true dictionary was defined (in
%                                synthetic experiments), this parameter holds a vector of length
%                                param.numIteration that includes the detection ratios in each
%                                iteration).
%    totalerr                    The total representation error after each
%                                iteration (defined only if
%                                param.displayProgress=1 and
%                                param.errorFlag = 0)
%    numCoef                     A vector of length param.numIteration that
%                                include the average number of coefficients required for representation
%                                of each signal (in each iteration) (defined only if
%                                param.displayProgress=1 and
%                                param.errorFlag = 1)
% =========================================================================

if (~isfield(param,'displayProgress'))    %如果字符串displayProgress不是结构数组param的一个领域的名字
    param.displayProgress = 0;
end
totalerr(1) = 99999; %第一次迭代后的整体表示误差是99999
if (isfield(param,'errorFlag')==0)
    param.errorFlag = 0;
end

if (isfield(param,'TrueDictionary'))  % 如果字符串TrueDictionary是结构数组param的一个领域的名字
    displayErrorWithTrueDictionary = 1;%显示字典误差
    ErrorBetweenDictionaries = zeros(param.numIteration+1,1);%训练的字典与原始字典的误差向量，列向量50*1
    ratio = zeros(param.numIteration+1,1);% 
else
    displayErrorWithTrueDictionary = 0;
	ratio = 0;
end
if (param.preserveDCAtom>0)%字典的第一列元素是常数
    FixedDictionaryElement(1:size(Data,1),1) = 1/sqrt(size(Data,1));%将字典的第一列赋值，固定的字典元素
else
    FixedDictionaryElement = [];
end
% coefficient calculation method is OMP with fixed number of coefficients

%初始化字典
if ( size(Data,2) < param.K)%数据的个数小于待训练的原子的个数：HYP修改内容：因为我的数据必然为1列，所以此段删去，加了0---------------------------
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    Dictionary = Data(:,1:size(Data,2));%将数据矩阵赋予字典
    return;
%     %-------------------------------------------------------------------------------------------------------------------------------------------
elseif (strcmp(param.InitializationMethod,'DataElements'))%当初始化方法是信号本身且信号个数大于字典列数时
    Dictionary(:,1:param.K-param.preserveDCAtom) = Data(:,1:param.K-param.preserveDCAtom);%将信号数据的第一列到待训练原子的个数的那些信号全部赋予给字典的这些列
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))%如果初始化方法是给定的矩阵时
    disp('The Dicionart is given by User!');
    Dictionary(:,1:param.K-param.preserveDCAtom) = param.initialDictionary(:,1:param.K-param.preserveDCAtom);%用初始字典的待训练的这些列赋予字典
end
% reduce the components in Dictionary that are spanned by the fixed
%   字典的归一化
% elements
if (param.preserveDCAtom)%当字典中有一个原子是常数时
    tmpMat = FixedDictionaryElement \ Dictionary;%用字典左除第一列元素
    Dictionary = Dictionary - FixedDictionaryElement*tmpMat;%公式D=D-D1*D1/D
end
%normalize the dictionary.         归一化
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));%1/字典的F-范数再生成对角阵再乘以字典就得到归一化的字典
Dictionary = Dictionary.*repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.字典每一行的元素乘以第一行元素的符号（正数为1负数为0）
totalErr = zeros(1,param.numIteration);%误差向量

% the K-SVD algorithm starts here.KSVD算法开始

for iterNum = 1:param.numIteration%迭代次数变量  50次迭代
    disp(['Now is the Iteration   ',num2str(iterNum),' \ ' , num2str(param.numIteration)]);  %迭代次数声明——我加的，可以去掉
       
    % find the coefficients    一、稀疏编码求稀疏系数
    if (param.errorFlag==0)%每一个信号的系数的个数固定
        %CoefMatrix = mexOMPIterative2(Data, [FixedDictionaryElement,Dictionary],param.L);
        CoefMatrix = OMP([FixedDictionaryElement,Dictionary],Data, param.L);%调用子程序omp.m得到稀疏的系数矩阵
    else 
        %CoefMatrix = mexOMPerrIterative(Data, [FixedDictionaryElement,Dictionary],param.errorGoal);
        CoefMatrix = OMPerr([FixedDictionaryElement,Dictionary],Data, param.errorGoal);%调用函数omperr.m
        param.L = 1;
    end
    
    replacedVectorCounter = 0;
	rPerm = randperm(size(Dictionary,2)); %得到1到50的整数的随机排列，向量
    %二、字典元素更新
    for j = rPerm   %50次迭代，顺序随机
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(Data,...
            [FixedDictionaryElement,Dictionary],j+size(FixedDictionaryElement,2),...
            CoefMatrix ,param.L);%字典第j个原子以及相应系数的更新
        Dictionary(:,j) = betterDictionaryElement;%将更新的j个原子赋给字典的第j列
        if (param.preserveDCAtom) %如果字典的第一个原子是固定的？？？？？？？？？？？？？？？？？？？
            tmpCoef = FixedDictionaryElement\betterDictionaryElement;
            Dictionary(:,j) = betterDictionaryElement - FixedDictionaryElement*tmpCoef;%将第j列元素归一化？
            Dictionary(:,j) = Dictionary(:,j)./sqrt(Dictionary(:,j)'*Dictionary(:,j));
        end
        replacedVectorCounter = replacedVectorCounter+addedNewVector;%？？？？？？？？？？？？？？？
    end

    if (iterNum>1 & param.displayProgress)%迭代次数大于1时输出...
        if (param.errorFlag==0)%非零系数个数固定时
            output.totalerr(iterNum-1) = sqrt(sum(sum((Data-[FixedDictionaryElement,Dictionary]*CoefMatrix).^2))/prod(size(Data)));%||Y-DX||F^2/n^2再求和开根号得到这一级迭代的均方差
            disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalerr(iterNum-1))]);%显示误差
        else
            output.numCoef(iterNum-1) = length(find(CoefMatrix))/size(Data,2);
            disp(['Iteration   ',num2str(iterNum),'   Average number of coefficients: ',num2str(output.numCoef(iterNum-1))]);%显示每个信号的非零系数的平均个数
        end
    end
    if (displayErrorWithTrueDictionary ) 
        [ratio(iterNum+1),ErrorBetweenDictionaries(iterNum+1)] = I_findDistanseBetweenDictionaries(param.TrueDictionary,Dictionary);%得到训练的字典和原始字典的误差
        disp(strcat(['Iteration  ', num2str(iterNum),' ratio of restored elements: ',num2str(ratio(iterNum+1))]));%输出迭代次数及其检测率
        output.ratio = ratio;
    end
    Dictionary = I_clearDictionary(Dictionary,CoefMatrix(size(FixedDictionaryElement,2)+1:end,:),Data);%
    
    if (isfield(param,'waitBarHandle'))
        waitbar(iterNum/param.counterForWaitBar);
    end
end

disp('New Matrix!');
output.CoefMatrix = CoefMatrix;
Dictionary = [FixedDictionaryElement,Dictionary];
% output.Dictionary=Dictionary;       %我HYP自己加的
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findBetterDictionaryElement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Dictionary,j,CoefMatrix,numCoefUsed)%字典第j个原子以及相应系数的更新
if (length(who('numCoefUsed'))==0) %？？？？？？？？？？？？
    numCoefUsed = 1;
end
relevantDataIndices = find(CoefMatrix(j,:)); % the data indices that uses the j'th dictionary element.返回第j行系数的非零元素的位置向量
if (length(relevantDataIndices)<1) %(length(relevantDataIndices)==0)如果第j个系数全零
    ErrorMat = Data-Dictionary*CoefMatrix;%求误差矩阵E=Y-DX
    ErrorNormVec = sum(ErrorMat.^2);%误差的F范数的平方，得到一个行向量1*50
    [d,i] = max(ErrorNormVec);%找到向量中最大分量及其位置i
    betterDictionaryElement = Data(:,i);%ErrorMat(:,i); 将信号第i列赋给betterDictionaryElement
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);%再将其归一化，得到列向量20*1
    betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));%将其乘以第一个元素的符号
    CoefMatrix(j,:) = 0;%将0赋给系数的第j行，更新了系数的第j行
    NewVectorAdded = 1;%更新了一个原子？？？？？？？？？？？？？？？
    return;
end

NewVectorAdded = 0;
tmpCoefMatrix = CoefMatrix(:,relevantDataIndices); %得到第j行非零系数的那些列的系数
tmpCoefMatrix(j,:) = 0;% the coeffitients of the element we now improve are not relevant.把第j行清零
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); % vector of errors that we want to minimize with the new element求误差Ek
% % the better dictionary element and the values of beta are found using svd.
% % This is because we would like to minimize || errors - beta*element ||_F^2. 
% % that is, to approximate the matrix 'errors' with a one-rank matrix. This
% % is done using the largest singular value.
[betterDictionaryElement,singularValue,betaVector] = svds(errors,1);%第四步利用SVD分解Ek，得到更新后的原子
CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';% *signOfFirstElem第四步，得到更新后第k行系数

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findDistanseBetweenDictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ratio,totalDistances] = I_findDistanseBetweenDictionaries(original,new) %同MOD算法
% first, all the column in oiginal starts with positive values.
catchCounter = 0;
totalDistances = 0;
for i = 1:size(new,2)
    new(:,i) = sign(new(1,i))*new(:,i);
end
for i = 1:size(original,2)
    d = sign(original(1,i))*original(:,i);
    distances =sum ( (new-repmat(d,1,size(new,2))).^2);
    [minValue,index] = min(distances);
    errorOfElement = 1-abs(new(:,index)'*d);
    totalDistances = totalDistances+errorOfElement;
    catchCounter = catchCounter+(errorOfElement<0.01);
end
ratio = 100*catchCounter/size(original,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  I_clearDictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Dictionary = I_clearDictionary(Dictionary,CoefMatrix,Data)
T2 = 0.99;
T1 = 3;
K=size(Dictionary,2);
Er=sum((Data-Dictionary*CoefMatrix).^2,1); % remove identical atoms
G=Dictionary'*Dictionary; G = G-diag(diag(G));
for jj=1:1:K,
    if max(G(jj,:))>T2 | length(find(abs(CoefMatrix(jj,:))>1e-7))<=T1 ,
        [val,pos]=max(Er);
        Er(pos(1))=0;
        Dictionary(:,jj)=Data(:,pos(1))/norm(Data(:,pos(1)));
        G=Dictionary'*Dictionary; G = G-diag(diag(G));
    end;
end;

