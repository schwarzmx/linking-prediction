%--- OPTIMIZATION METHOD ---%
% method = 'bfgs';
method = 'lbfgs';

%--- LOAD DATASET (synthetic)---%
Tr = csv2struct('dataset/train_200.csv');
Te = csv2struct('dataset/test_200.csv');
Tr = csv2struct('dataset/train_100.csv');
Te = csv2struct('dataset/test_100.csv');
% Tr = csv2struct('dataset/train_50.csv');
% Te = csv2struct('dataset/test_50.csv');
% Tr = csv2struct('dataset/train_10.csv');
% Te = csv2struct('dataset/test_10.csv');
% Tr = csv2struct('dataset/train_3x3.csv');
% Te = csv2struct('dataset/test_3x3.csv');
% Tr = csv2struct('dataset/train_1x1.csv');
% Te = csv2struct('dataset/train_1x1.csv');

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
if strcmp(method, 'lbfgs')
    %--- minFunc ---%
    addpath(genpath('minFunc_2012'));
    
    options.DerivativeCheck = 'off';
    options.numDiff = 0;
    options.Corr = 500;
    options.Display = 'iter';
    options.MaxFunEvals = 1000;
elseif strcmp(method, 'bfgs')
    %--- fminunc ---%
%     finite differences gradient
%     options = optimset('Display','iter',...
%                 'FunValCheck','on',...
%                 'Diagnostics','on');

    % given gradient, with check
%     options = optimset('GradObj','on',...
%                 'Display','iter',...
%                 'FunValCheck','on',...
%                 'DerivativeCheck','on',...
%                 'Diagnostics','on');

    % given gradient, no check
    options = optimset('GradObj','on',...
                'Display','iter',...
                'Diagnostics','on');
end

%--- LEARNING ---%
if strcmp(method, 'lbfgs')
    W = minFunc(fun, initialW, options);
elseif strcmp(method, 'bfgs')
    [W, fval] = fminunc(fun, initialW, options);
end

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