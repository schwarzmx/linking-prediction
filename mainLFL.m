%--- ADD minFunc ---%
addpath(genpath('minFunc_2012'));
%--- LOAD DATASET ---%
Tr = csv2struct('dataset/train_20.csv');
Te = csv2struct('dataset/train_20.csv');
% Tr = csv2struct('dataset/train_3x3.csv');
% Te = csv2struct('dataset/train_3x3.csv');
% number of latent features
k = 5;
% penalty
lambda = 1e-4;

usersU = Tr.u;
usersV = Tr.v;
labels = Tr.y;

% Shuffle training set
n = length(usersU);
I = randperm(n);
Tr.u = Tr.u(I);
Tr.v = Tr.v(I);
Tr.y = Tr.y(I);

% Figure out number of users and number of possible labels
U = max(usersU) - min(usersU) + 1;
V = max(usersV) - min(usersV) + 1;
% Y = max(labels) - min(labels) + 1;
Y = 2;

%-- INITIALIZE WEIGHTS --%
userW = 1/k * randn(k, Y, U);
% lambdaW(:,:,i,j) is a k by k matrix for each edge ij
lambdaW = 1/k * randn(k, k, U, U);
fun = @(theWeights) lflObjectiveFunction(theWeights, k, Y, U, lambda,...
    usersU, usersV, labels);
initialW = [userW(:); lambdaW(:)];

%--- OPTIMIZATION OPTIONS ---%
%--- minFunc ---%
options.numDiff = 1;
options.Display = 'iter';
options.MaxFunEvals = 500000;

%--- fminunc ---%
% finite differences gradient
% options = optimset('Display','iter',...
%             'FunValCheck','on',...
%             'Diagnostics','on');

% given gradient, with check
% options = optimset('GradObj','on',...
%             'Display','iter',...
%             'FunValCheck','on',...
%             'DerivativeCheck','on',...
%             'Diagnostics','on');

% given gradient, no check
% options = optimset('GradObj','on',...
%             'Display','iter',...
%             'FunValCheck','on',...
%             'Diagnostics','on');

%--- LEARNING ---%
% [W, fval] = fminunc(fun, initialW, options);
W = minFunc(fun, initialW, options);

% disp(W);
% disp(fval);

%--- MAKE PREDICTION ---%
userW = reshape(W(1:k*Y*U), k, Y, U);
lambdaW = reshape(W(k*Y*U+1:end), k, k, U, U);
W = [];
W.userW = userW; % W.userW(1,:,:) = 1;
W.lambdaW = lambdaW;
W.usersU = usersU;
W.usersV = usersV;
W.labels = labels;
% predictor = @lflPredictor;

% [predictions, argmax, probabilities] = predictor(W);

trainErrors = testLFL(@lflPredictor, W, Tr);
testErrors = testLFL(@lflPredictor, W, Te);

format = strcat('\n train/test 0-1 error = %4.4f / %4.4f',...
    ', rmse = %4.4f / %4.4f',', mae = %4.4f / %4.4f ');
disp(sprintf(format, trainErrors.zoe, testErrors.zoe, trainErrors.rmse,...
    testErrors.rmse, trainErrors.mae, testErrors.mae))


% disp(probabilities);
% disp(predictions);
% disp(argmax);