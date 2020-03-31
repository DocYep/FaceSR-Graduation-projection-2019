function [X]=OMP(D,Y,L); %返回稀疏系数，大小是50*1500
%=============================================
% Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: 
%       D - the dictionary (its columns MUST be normalized).归一化的字典
%       Y - the signals to represent     待表达的信号
%       L - the max. number of coefficients for each signal.  每个信号的系数的最大个数
% output arguments: 
%       X - sparse coefficient matrix.稀疏系数矩阵
%=============================================
[n,P]=size(Y);%信号大小20*1500
[n,K]=size(D);%字典大小20*50
for k=1:1:P,%1到1500次循环，间隔是1
    a=[];
    x=Y(:,k);%信号的第k列（即第k个信号）赋给向量x,向量x是第k个信号向量
    residual=x;
    indx=zeros(L,1);%索引是三维列向量
    for j=1:1:L,    %1到3次循环，间隔是1
        proj=D'*residual;%得到50*1，第k个信号的系数向量，D的转置就是D的逆，因为D是正交的
        [maxVal,pos]=max(abs(proj));%找到这个系数当中绝对值最大的数值及位置下标
        pos=pos(1);
        indx(j)=pos; %把最大数值的下标赋给索引的第j个分量
        a=pinv(D(:,indx(1:j)))*x; %D的部分列的伪逆矩阵再乘以x,意思只取算出来的系数中最大的j个分量对应的字典原子的违逆乘以信号得到一个系数
        residual=x-D(:,indx(1:j))*a;%用信号减去上面的系数乘以对应字典的列得到初始信号与稀疏表示的信号的误差
        if sum(residual.^2) < 1e-6  %判断误差是否在限制范围内
            break;%在，结束for循环
        end     %否则继续循环
    end;
    temp=zeros(K,1);%K=50维向量
    temp(indx(1:j))=a;%a是有最大的j列个元素的系数，最后所求的最多有三个分量的第k个稀疏系数50*1，
    X(:,k)=sparse(temp);%生成稀疏矩阵的k列
end;
return;
