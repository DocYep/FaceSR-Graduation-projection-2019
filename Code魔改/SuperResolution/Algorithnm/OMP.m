function [X]=OMP(D,Y,L); %����ϡ��ϵ������С��50*1500
%=============================================
% Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: 
%       D - the dictionary (its columns MUST be normalized).��һ�����ֵ�
%       Y - the signals to represent     �������ź�
%       L - the max. number of coefficients for each signal.  ÿ���źŵ�ϵ����������
% output arguments: 
%       X - sparse coefficient matrix.ϡ��ϵ������
%=============================================
[n,P]=size(Y);%�źŴ�С20*1500
[n,K]=size(D);%�ֵ��С20*50
for k=1:1:P,%1��1500��ѭ���������1
    a=[];
    x=Y(:,k);%�źŵĵ�k�У�����k���źţ���������x,����x�ǵ�k���ź�����
    residual=x;
    indx=zeros(L,1);%��������ά������
    for j=1:1:L,    %1��3��ѭ���������1
        proj=D'*residual;%�õ�50*1����k���źŵ�ϵ��������D��ת�þ���D���棬��ΪD��������
        [maxVal,pos]=max(abs(proj));%�ҵ����ϵ�����о���ֵ������ֵ��λ���±�
        pos=pos(1);
        indx(j)=pos; %�������ֵ���±긳�������ĵ�j������
        a=pinv(D(:,indx(1:j)))*x; %D�Ĳ����е�α������ٳ���x,��˼ֻȡ�������ϵ��������j��������Ӧ���ֵ�ԭ�ӵ�Υ������źŵõ�һ��ϵ��
        residual=x-D(:,indx(1:j))*a;%���źż�ȥ�����ϵ�����Զ�Ӧ�ֵ���еõ���ʼ�ź���ϡ���ʾ���źŵ����
        if sum(residual.^2) < 1e-6  %�ж�����Ƿ������Ʒ�Χ��
            break;%�ڣ�����forѭ��
        end     %�������ѭ��
    end;
    temp=zeros(K,1);%K=50ά����
    temp(indx(1:j))=a;%a��������j�и�Ԫ�ص�ϵ��������������������������ĵ�k��ϡ��ϵ��50*1��
    X(:,k)=sparse(temp);%����ϡ������k��
end;
return;
