addpath(genpath('minFunc_2012'));
%--- LOAD DATASET ---%
% Tr = csv2struct('dataset/train_50.csv');
% Te = csv2struct('dataset/test_50.csv');
% Tr = csv2struct('dataset/train_20.csv');
% Te = csv2struct('dataset/train_20.csv');
Tr = csv2struct('dataset/train_3x3.csv');
Te = csv2struct('dataset/test_3x3.csv');

% construct data matrix
X = sparse(Tr.u, Tr.v, Tr.y);
disp(X);

%--- PARAMETERS ---%
lambda = 1e0;  % regularization
k = 5;          % number of latent features

%-- INITIALIZE WEIGHTS --%
usersU = Tr.u;
usersV = Tr.v;
labels = Tr.y;
U = max(usersU);
V = max(usersV);
userW = 1/k * randn(U, k);
lambdaW = 1/k * randn(k, k);

% initialize objective function
fun = @(weights) smfObjectiveFunction(weights, X, lambda, U, k);
initialW = [userW(:); lambdaW(:)];

%--- OPTIMIZATION OPTIONS ---%
%--- minFunc ---%
options.numDiff = 1;
options.Display = 'iter';
options.MaxFunEvals = 100000;

%--- fminunc ---%
% finite differences gradient
% options = optimset('Display','iter',...
%             'FunValCheck','on',...
%             'Diagnostics','on');
        
%--- LEARNING ---%
% [W, fval] = fminunc(fun, initialW, options);

W = minFunc(fun, initialW, options);

%--- PREDICTION ---%
userW = reshape(W(1 : U*k), U,k);
lambdaW = reshape(W(U*k + 1 : end), k, k);
W = [];
W.userW = userW;
W.lambdaW = lambdaW;

trainErrors = testSMF(W, Tr);
testErrors = testSMF(W, Te);

format = strcat('\n train/test 0-1 error = %4.4f / %4.4f',...
    ', rmse = %4.4f / %4.4f',', mae = %4.4f / %4.4f ');
disp(sprintf(format, trainErrors.zoe, testErrors.zoe, trainErrors.rmse,...
    testErrors.rmse, trainErrors.mae, testErrors.mae))