function [TrainingError,TestError,Coefficients] = LSR_cdp30(TrainingData,TestData,N,lambda)
x_train = TrainingData(1,:);
y_train = TrainingData(2,:);
A_train = GenA(x_train,N,lambda);
B_train = [y_train.'; zeros(N+1,1)];
alpha = A_train\B_train;
TrainingError = (1/length(TrainingData(1,:)))*(norm(A_train*alpha-B_train)^2 + lambda*(norm(alpha)^2));

x_test = TestData(1,:);
y_test = TestData(2,:);
A_test = GenA(x_test,N,lambda);
B_test = [y_test.'; zeros(N+1,1)];
TestError = (1/length(TestData(1,:)))*(norm(A_test*alpha-B_test)^2 + lambda*(norm(alpha)^2));
Coefficients = alpha;
end
function A = GenA(X,N,lambda)
M = size(X,2);
A = zeros(M,N+1);
for i=0:N
    for j=1:M
        A(j,N-i+1) = X(j)^(i);
    end
end
botA = zeros(N+1,N+1);
for i=1:N+1
    botA(i,i) = sqrt(lambda);
end
A = [A; botA];
end