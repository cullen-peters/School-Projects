function [F, G, H] = function_cost(Z, X, Y, Lambda, t)
% Compute the cost function F(Z)
%
% INPUTS: 
%   Z: Parameter values
%   X: Features
%   Y: Labels
%   Lambda and t: hyper-parameter in the object function
% OUTPUTS
%   F: Function value
%   G: Gradient value
%   H: Hessian value
%
% @Cullen Peters, cdp30@duke.edu
% 2020-03-20
M = size(X,1);
N = size(X,2);

W = Z(1:M)';
C = Z(M+1);
kexi = Z(M+2:end);
inLog = ((W')*X).*Y + C*Y + kexi - 1;

F = sum(kexi) + Lambda*(W.')*W - (1/t)*sum(log(inLog)) - (1/t)*sum(log(kexi));

% Derivative of F with respect to W
dFdW = 2*Lambda*W - (1/t)*sum((X.*Y)./inLog, 2);
% Derivative of F with respect to C
dFdC = -(1/t)*sum((Y./inLog));
% Derivative of F with respect to kexi
dFdE = 1 - (1/t)*(1./inLog) - (1/t)*(1./kexi);
G = [dFdW; dFdC; dFdE'];

% Second Derivative with respect to C
dFdCdC = (1/t)*sum((Y.^2)./(inLog.^2));
% Derivative with respect to kexi and C for all kexi
dFdCdE = (1/t)*Y./(inLog.^2);
% Second Derivative with respect to kexi for all kexi
dFdEdE = (eye(size(kexi,2))/t).*(1./(inLog.^2)+1./(kexi.^2));
% Second Derivative with respect to W for all W
dFdWdW = zeros(M,M);
for i=1:N
    dFdWdW = dFdWdW + (1/t)*((X(:,i)*X(:,i)')*Y(i)^2) / ((W'*X(:,i)*Y(i)+C*Y(i)+kexi(i)-1)^2);
end
dFdWdW = 2*Lambda*eye(M)+dFdWdW;
% Derivative with respect to W and C
dFdWdC = (1/t)*sum((X.*(Y.^2)./(inLog.^2)), 2);
% Derivative with respect to W and kexi
dFdWdE = (1/t)*(X.*Y)./(inLog.^2);

H = zeros(size(G,2),size(G,2));
H(size(W,1)+1, size(W,1)+1) = dFdCdC;
H(size(W,1)+1, (size(W,1)+2):(size(W,1)+1+size(kexi,2))) = dFdCdE;
H((size(W,1)+2):(size(W,1)+1+size(kexi,2)), size(W,1)+1) = dFdCdE.';
H((size(W,1)+2):(size(W,1)+1+size(kexi,2)), (size(W,1)+2):(size(W,1)+1+size(kexi,2))) = dFdEdE;
H(1:size(W,1), 1:size(W,1)) = dFdWdW;
H(size(W,1)+1, 1:size(W,1)) = dFdWdC;
H(1:size(W,1), size(W,1)+1) = dFdWdC.';
H(1:size(W,1), (size(W,1)+2):(size(W,1)+1+size(kexi,2))) = dFdWdE;
H((size(W,1)+2):(size(W,1)+1+size(kexi,2)), 1:size(W,1)) = dFdWdE.';
