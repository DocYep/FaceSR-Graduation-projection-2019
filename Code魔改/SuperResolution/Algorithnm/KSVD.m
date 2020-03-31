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

if (~isfield(param,'displayProgress'))    %����ַ���displayProgress���ǽṹ����param��һ�����������
    param.displayProgress = 0;
end
totalerr(1) = 99999; %��һ�ε�����������ʾ�����99999
if (isfield(param,'errorFlag')==0)
    param.errorFlag = 0;
end

if (isfield(param,'TrueDictionary'))  % ����ַ���TrueDictionary�ǽṹ����param��һ�����������
    displayErrorWithTrueDictionary = 1;%��ʾ�ֵ����
    ErrorBetweenDictionaries = zeros(param.numIteration+1,1);%ѵ�����ֵ���ԭʼ�ֵ�����������������50*1
    ratio = zeros(param.numIteration+1,1);% 
else
    displayErrorWithTrueDictionary = 0;
	ratio = 0;
end
if (param.preserveDCAtom>0)%�ֵ�ĵ�һ��Ԫ���ǳ���
    FixedDictionaryElement(1:size(Data,1),1) = 1/sqrt(size(Data,1));%���ֵ�ĵ�һ�и�ֵ���̶����ֵ�Ԫ��
else
    FixedDictionaryElement = [];
end
% coefficient calculation method is OMP with fixed number of coefficients

%��ʼ���ֵ�
if ( size(Data,2) < param.K)%���ݵĸ���С�ڴ�ѵ����ԭ�ӵĸ�����HYP�޸����ݣ���Ϊ�ҵ����ݱ�ȻΪ1�У����Դ˶�ɾȥ������0---------------------------
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    Dictionary = Data(:,1:size(Data,2));%�����ݾ������ֵ�
    return;
%     %-------------------------------------------------------------------------------------------------------------------------------------------
elseif (strcmp(param.InitializationMethod,'DataElements'))%����ʼ���������źű������źŸ��������ֵ�����ʱ
    Dictionary(:,1:param.K-param.preserveDCAtom) = Data(:,1:param.K-param.preserveDCAtom);%���ź����ݵĵ�һ�е���ѵ��ԭ�ӵĸ�������Щ�ź�ȫ��������ֵ����Щ��
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))%�����ʼ�������Ǹ����ľ���ʱ
    disp('The Dicionart is given by User!');
    Dictionary(:,1:param.K-param.preserveDCAtom) = param.initialDictionary(:,1:param.K-param.preserveDCAtom);%�ó�ʼ�ֵ�Ĵ�ѵ������Щ�и����ֵ�
end
% reduce the components in Dictionary that are spanned by the fixed
%   �ֵ�Ĺ�һ��
% elements
if (param.preserveDCAtom)%���ֵ�����һ��ԭ���ǳ���ʱ
    tmpMat = FixedDictionaryElement \ Dictionary;%���ֵ������һ��Ԫ��
    Dictionary = Dictionary - FixedDictionaryElement*tmpMat;%��ʽD=D-D1*D1/D
end
%normalize the dictionary.         ��һ��
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));%1/�ֵ��F-���������ɶԽ����ٳ����ֵ�͵õ���һ�����ֵ�
Dictionary = Dictionary.*repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.�ֵ�ÿһ�е�Ԫ�س��Ե�һ��Ԫ�صķ��ţ�����Ϊ1����Ϊ0��
totalErr = zeros(1,param.numIteration);%�������

% the K-SVD algorithm starts here.KSVD�㷨��ʼ

for iterNum = 1:param.numIteration%������������  50�ε���
    disp(['Now is the Iteration   ',num2str(iterNum),' \ ' , num2str(param.numIteration)]);  %�����������������Ҽӵģ�����ȥ��
       
    % find the coefficients    һ��ϡ�������ϡ��ϵ��
    if (param.errorFlag==0)%ÿһ���źŵ�ϵ���ĸ����̶�
        %CoefMatrix = mexOMPIterative2(Data, [FixedDictionaryElement,Dictionary],param.L);
        CoefMatrix = OMP([FixedDictionaryElement,Dictionary],Data, param.L);%�����ӳ���omp.m�õ�ϡ���ϵ������
    else 
        %CoefMatrix = mexOMPerrIterative(Data, [FixedDictionaryElement,Dictionary],param.errorGoal);
        CoefMatrix = OMPerr([FixedDictionaryElement,Dictionary],Data, param.errorGoal);%���ú���omperr.m
        param.L = 1;
    end
    
    replacedVectorCounter = 0;
	rPerm = randperm(size(Dictionary,2)); %�õ�1��50��������������У�����
    %�����ֵ�Ԫ�ظ���
    for j = rPerm   %50�ε�����˳�����
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(Data,...
            [FixedDictionaryElement,Dictionary],j+size(FixedDictionaryElement,2),...
            CoefMatrix ,param.L);%�ֵ��j��ԭ���Լ���Ӧϵ���ĸ���
        Dictionary(:,j) = betterDictionaryElement;%�����µ�j��ԭ�Ӹ����ֵ�ĵ�j��
        if (param.preserveDCAtom) %����ֵ�ĵ�һ��ԭ���ǹ̶��ģ�������������������������������������
            tmpCoef = FixedDictionaryElement\betterDictionaryElement;
            Dictionary(:,j) = betterDictionaryElement - FixedDictionaryElement*tmpCoef;%����j��Ԫ�ع�һ����
            Dictionary(:,j) = Dictionary(:,j)./sqrt(Dictionary(:,j)'*Dictionary(:,j));
        end
        replacedVectorCounter = replacedVectorCounter+addedNewVector;%������������������������������
    end

    if (iterNum>1 & param.displayProgress)%������������1ʱ���...
        if (param.errorFlag==0)%����ϵ�������̶�ʱ
            output.totalerr(iterNum-1) = sqrt(sum(sum((Data-[FixedDictionaryElement,Dictionary]*CoefMatrix).^2))/prod(size(Data)));%||Y-DX||F^2/n^2����Ϳ����ŵõ���һ�������ľ�����
            disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalerr(iterNum-1))]);%��ʾ���
        else
            output.numCoef(iterNum-1) = length(find(CoefMatrix))/size(Data,2);
            disp(['Iteration   ',num2str(iterNum),'   Average number of coefficients: ',num2str(output.numCoef(iterNum-1))]);%��ʾÿ���źŵķ���ϵ����ƽ������
        end
    end
    if (displayErrorWithTrueDictionary ) 
        [ratio(iterNum+1),ErrorBetweenDictionaries(iterNum+1)] = I_findDistanseBetweenDictionaries(param.TrueDictionary,Dictionary);%�õ�ѵ�����ֵ��ԭʼ�ֵ�����
        disp(strcat(['Iteration  ', num2str(iterNum),' ratio of restored elements: ',num2str(ratio(iterNum+1))]));%�������������������
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
% output.Dictionary=Dictionary;       %��HYP�Լ��ӵ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findBetterDictionaryElement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Dictionary,j,CoefMatrix,numCoefUsed)%�ֵ��j��ԭ���Լ���Ӧϵ���ĸ���
if (length(who('numCoefUsed'))==0) %������������������������
    numCoefUsed = 1;
end
relevantDataIndices = find(CoefMatrix(j,:)); % the data indices that uses the j'th dictionary element.���ص�j��ϵ���ķ���Ԫ�ص�λ������
if (length(relevantDataIndices)<1) %(length(relevantDataIndices)==0)�����j��ϵ��ȫ��
    ErrorMat = Data-Dictionary*CoefMatrix;%��������E=Y-DX
    ErrorNormVec = sum(ErrorMat.^2);%����F������ƽ�����õ�һ��������1*50
    [d,i] = max(ErrorNormVec);%�ҵ�����������������λ��i
    betterDictionaryElement = Data(:,i);%ErrorMat(:,i); ���źŵ�i�и���betterDictionaryElement
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);%�ٽ����һ�����õ�������20*1
    betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));%������Ե�һ��Ԫ�صķ���
    CoefMatrix(j,:) = 0;%��0����ϵ���ĵ�j�У�������ϵ���ĵ�j��
    NewVectorAdded = 1;%������һ��ԭ�ӣ�����������������������������
    return;
end

NewVectorAdded = 0;
tmpCoefMatrix = CoefMatrix(:,relevantDataIndices); %�õ���j�з���ϵ������Щ�е�ϵ��
tmpCoefMatrix(j,:) = 0;% the coeffitients of the element we now improve are not relevant.�ѵ�j������
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); % vector of errors that we want to minimize with the new element�����Ek
% % the better dictionary element and the values of beta are found using svd.
% % This is because we would like to minimize || errors - beta*element ||_F^2. 
% % that is, to approximate the matrix 'errors' with a one-rank matrix. This
% % is done using the largest singular value.
[betterDictionaryElement,singularValue,betaVector] = svds(errors,1);%���Ĳ�����SVD�ֽ�Ek���õ����º��ԭ��
CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';% *signOfFirstElem���Ĳ����õ����º��k��ϵ��

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findDistanseBetweenDictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ratio,totalDistances] = I_findDistanseBetweenDictionaries(original,new) %ͬMOD�㷨
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

