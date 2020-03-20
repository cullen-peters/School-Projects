function [optSol, optLambda] = solver_interior(X, Y, setPara)
% Get the optimal solution using interior point algorithm and get the
% optimal lamda using five fold cross-validation from the given lamda set
%
% INPUTS:
%   X(MxN) : trData(i,j) is the i-th feature from the j-th trial
%   Y(1xN): trData(j) is the label of the j-th trial (1 or -1)
%   setPara : Initialized parameters
%            setPara.t      
%            setPara.beta 
%            setPara.Tmax   
%            setPara.tol    
%            setPara.W      
%            setPara.C      
%
% OUTPUTS:
%   optiLamda: Optimal lamda value 
%   optSol: the optimal solution     
%
% @Cullen Peters, cdp30@duke.edu
% 2020-03-20

beta = setPara.beta;
Tmax = setPara.Tmax;
tol = setPara.tol;
W = setPara.W;
C = setPara.C;
Lambdas = [0.01, 1, 100, 10000];
maxAc = 0;


test_X2 = X(:, 1:40);
test_Y2 = Y(:, 1:40);
training_X2 = X(:, 41:end);
training_Y2 = Y(:, 41:end);
for i=1:4
    Lambda = Lambdas(i);
    Ac = zeros(1,5);
    disp("    Lambda: "+Lambda)
    for j=1:5
        [test_X2, test_Y2, training_X2, training_Y2] = xval_next(test_X2, test_Y2, training_X2, training_Y2);
        kexi = max(1-training_Y2.*((W)*training_X2+C), zeros(1,160)) + 0.001;
        Z = [W C kexi];

        t = setPara.t;
        while (t <= Tmax)
            [newZ, ~] = solver_Newton(@function_cost, Z, Lambda, t, training_X2, training_Y2, tol);
            Z = newZ;
            t = beta*t;
        end

        W = Z(1:204);
        C = Z(205);
        val_matrix = (test_Y2.*(W*test_X2+C)>=0);
        Ac(j) = sum(val_matrix)/size(val_matrix,2);
    end
    disp("        This Lambda's Accuracies: ")
    disp(Ac)
    disp("        Mean of Accuracies: "+mean(Ac))
    disp(" ")
    if mean(Ac) > maxAc
        maxAc = mean(Ac);
        optLambda = Lambda;
    end  
end
disp(" Optimal Lambda: "+optLambda)
disp(" ")
% Train model with optimal lambda on all the training data
kexi = max(1-Y.*((W)*X+C), zeros(1,200)) + 0.001;
Z = [W C kexi];

while (t <= Tmax)    
    [optSol, ~] = solver_Newton(@function_cost, Z, optLamda, t, X, Y, tol);
    Z = optSol;
    t = beta*t;    
end
optSol = Z;
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