%--- LOAD DATASET ---%
% Tr = csv2struct('dataset/train_1x1.csv');
Tr = csv2struct('dataset/train_3x3.csv');

% construct data matrix
X = sparse(Tr.u, Tr.v, Tr.y);

%--- PARAMETERS ---%
lambda = 1e-4;  % regularization
k = 4;          % number of latent features

%-- INITIALIZE WEIGHTS --%
U = max(usersU) - min(usersU) + 1;
V = max(usersV) - min(usersV) + 1;
userW = 1/k * randn(U, k);
lambdaW = 1/k * randn(k, k);

% initialize objective function
fun = @(weights) smfObjectiveFunction(weights, X, lambda, U, k);
initialW = [userW(:); lambdaW(:)];

%--- OPTIMIZATION OPTIONS ---%
% finite differences gradient
options = optimset('Display','iter',...
            'FunValCheck','on',...
            'Diagnostics','on');
        
%--- LEARNING ---%
[W, fval] = fminunc(fun, initialW, options);


%--- PREDICTION ---%
userW = reshape(W(1 : U*k), U,k);
lambdaW = reshape(W(U*k + 1 : end), k, k);

prediction = userW * lambdaW * userW';
disp(prediction);