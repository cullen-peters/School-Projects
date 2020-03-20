function [optSol, err] = solver_Newton(function_cost,init_Z,Lambda,t,X,Y,tol)
% Solve the optimization problem using Newton method
%
% INPUTS:
%   function_cost: Function handle of F(Z)
%   init_Z: Initial value of Z
%   tol: Tolerance
%
% OUTPUTS:
%   optSol: Optimal soultion
%   err: Error
%
% @Cullen Peters, cdp30@duke.edu
% 2020-03-20

Z = init_Z;
err = 2*tol+0.001;

% Set the error 2*tol to make sure the loop runs at least once
while err/2 > tol
    % Execute the cost function at the current iteration
    % F : function value, G : gradient, H, hessian
    [~, G, H] = feval(function_cost, Z, X, Y, Lambda, t);
    H_inv = (H)^(-1);
    dZ = -(H_inv*G).';
    err = (G')*H_inv*G;
    if  err/2 < tol
        break;
    end
    
    s = 1;
    Zs = Z + s*dZ;
    W = Zs(1:204).';
    C = Zs(205);
    kexi = Zs(206:end);
    check = (W.'*X).*Y + C*Y + kexi - 1;
    while ~((sum(check > 0) == size(kexi,2)) && (sum(kexi>0) == size(kexi,2)))
        s = 0.5*s;
        Zs = Z + s*dZ;
        W = Zs(1:204).';
        C = Zs(205);
        kexi = Zs(206:end);
        check = (W.'*X).*Y + C*Y + kexi - 1;
    end
    Z = Z + s*dZ;
end
optSol = Z;