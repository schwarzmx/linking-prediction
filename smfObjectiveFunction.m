function [ fun ] = smfObjectiveFunction( W, X, lambda, U, k )
%SMFOBJECTIVEFUNCTION Supervised Matrix Factorization's Objective Function

    uW = reshape(W(1 : U*k), U,k);
    lW = reshape(W(U*k + 1 : end), k, k);
    
    diff = X - uW * lW * uW';
    mse = norm(diff, 'fro') ^ 2;
    regularization = (lambda / 2) * norm(uW, 'fro') ^ 2;
    fun = mse + regularization;
end

