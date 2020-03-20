close all
warning off
load('DataFeaImg.mat');
Data1 = class{1, 1};
Data2 = class{1, 2};
Class1 = ones(1,120);
Class2 = -1*ones(1,120);

[test_X1, test_Y1, training_X1, training_Y1] = xval_split(6, Data1, Data2, Class1, Class2);
Ac = zeros(1,6);
setPara = struct('t',1,'beta',15,'Tmax',1000000,'tol',0.000001,'W',ones(1,204),'C',0);
for i=1:6
    disp("Cross Validation: Round "+i)
    [test_X1, test_Y1, training_X1, training_Y1] = xval_next(test_X1, test_Y1, training_X1, training_Y1);

    [optSol, optLamda] = solver_interior(training_X1, training_Y1, setPara);
    
    W = optSol(1:204);
    C = optSol(205);
    val_matrix = (test_Y1.*(W*test_X1+C)>=0);
    Ac(i) = sum(val_matrix)/size(val_matrix,2);
    
    figure()
    show_weights(abs(W));
end
disp("Final Test Accuracies")
disp(Ac)
disp("Mean Accuracy: "+mean(Ac))
disp("Standard Deviation of Accuracies: "+std(Ac))
disp("Fold 6 Optimal W Top 5:")
[maxW, index] = maxk(W(:), 5) %#ok<NOPTS>
disp("Fold 6 Optimal C: "+C)

% Function to evenly (half class A, half class B for each fold) split data 
% for N-fold cross validation
function [test_X, test_Y, training_X, training_Y] = xval_split(N, Data1, Data2, Class1, Class2)
siz = size(Data1,2)/N;
test_X = cat(2, Data1(:,1:siz), Data2(:, 1:siz));
test_Y = cat(2, Class1(:,1:siz), Class2(:, 1:siz));
training_X = [];
training_Y = [];
for i=2:N
    training_X = cat(2, training_X, Data1(:, (i-1)*siz+1:i*siz), Data2(:, (i-1)*siz+1:i*siz));
    training_Y = cat(2, training_Y, Class1(:, (i-1)*siz+1:i*siz), Class2(:, (i-1)*siz+1:i*siz));
end
end
% Function to get the data for the next round of cross validation based on
% the previous round
function [test_X, test_Y, training_X, training_Y] = xval_next(test_X, test_Y, training_X, training_Y)
siz = size(test_X, 2);
X = cat(2, training_X, test_X);
Y = cat(2, training_Y, test_Y);
test_X = X(:, 1:siz);
test_Y = Y(:, 1:siz);
training_X = X(:, siz+1:end);
training_Y = Y(:, siz+1:end);
end