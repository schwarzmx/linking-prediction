%--- OPTIMIZATION METHOD ---%
% method = 'bfgs';
method = 'lbfgs';
% withSideInfo = 1; % yes
withSideInfo = 0; % no
disp('loading dataset...');

%--- LOAD DATASET (synthetic)---%
% Tr = csv2struct('dataset/train_200.csv');
% Te = csv2struct('dataset/test_200.csv');
% Tr = csv2struct('dataset/train_100.csv');
% Te = csv2struct('dataset/test_100.csv');
% Tr = csv2struct('dataset/train_50.csv');
% Te = csv2struct('dataset/test_50.csv');
% Tr = csv2struct('dataset/train_10.csv');
% Te = csv2struct('dataset/test_10.csv');
% Tr = csv2struct('dataset/train_3x3.csv');
% Te = csv2struct('dataset/test_3x3.csv');
% Tr = csv2struct('dataset/train_2x2.csv');
% Te = csv2struct('dataset/train_2x2.csv');
% Tr = csv2struct('dataset/train_1x1.csv');
% Te = csv2struct('dataset/train_1x1.csv');
Tr = csv2struct('dataset/train_synthetic.csv');
Te = csv2struct('dataset/test_synthetic.csv');
% Tr = csv2struct('dataset/dataset_small_train.csv');
% Te = csv2struct('dataset/dataset_small_test.csv');
% Tr = csv2struct('dataset/dataset_full_train.csv');
% Te = csv2struct('dataset/dataset_full_test.csv');
if withSideInfo
    % the side info should contain as many features as necessary
    % but it's important to have as many rows as there are users
    sideInfo = loadSideInfo('dataset/side_info-synthetic.csv');
else
    sideInfo = [];
end

% number of latent features
k = 2;
% penalty
lambda = 1e-3;

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
% Y = max(labels) - min(labels) + 1;
Y = 2;

%-- INITIALIZE WEIGHTS --%
userW = 1/k * randn(k, Y, U);
%-- lambdaW(:,:,i,j) is a k by k matrix for each edge label
lambdaW = 1/k * randn(k, k);
if withSideInfo
    % vector of 2 x number features
    S = size(sideInfo,2);
    sideInfoW = 1/k * randn(S,1);
    
    initialW = [userW(:); lambdaW(:); sideInfoW];
else
    initialW = [userW(:); lambdaW(:)];
end



%--- OPTIMIZATION OPTIONS ---%
if strcmp(method, 'lbfgs')
    %--- minFunc ---%
    addpath(genpath('minFunc_2012'));
    
    options.DerivativeCheck = 'off';
    options.numDiff = 0;
    options.Corr = 100;
    options.Display = 'iter';
%     options.Display = 'final';
    options.MaxFunEvals = 1000;
    options.MaxIter = 10;
elseif strcmp(method, 'bfgs')
    %--- fminunc ---%
%     finite differences gradient
%     options = optimset('Display','iter',...
%                 'FunValCheck','on',...
%                 'Diagnostics','on');

    % given gradient, with check
    options = optimset('GradObj','on',...
                'Display','iter',...
                'FunValCheck','on',...
                'DerivativeCheck','on',...
                'Diagnostics','on');

    % given gradient, no check
%     options = optimset('GradObj','on',...
%                 'Display','iter',...
%                 'Diagnostics','on');
end

tic
disp('training...');
%--- LEARNING ---%
if strcmp(method, 'lbfgs')
    W = minFunc(@lflObjectiveFunction, initialW, options, k, Y, U, lambda,...
        usersU, usersV, sideInfo, labels, withSideInfo);
elseif strcmp(method, 'bfgs')
    fun = @(theWeights) lflObjectiveFunction(theWeights, k, Y, U, lambda,...
        usersU, usersV, sideInfo, labels, withSideInfo);
    [W, fval] = fminunc(fun, initialW, options);
end
toc

% disp(W);
% disp(fval);

%--- MAKE PREDICTION ---%

userW = reshape(W(1:k*Y*U), k, Y, U);
if withSideInfo
    lambdaStart = k*Y*U+1;
    lambdaEnd = lambdaStart + k * k - 1;
    lambdaW = reshape(W(lambdaStart : lambdaEnd), k, k);
    sideInfoW = W(lambdaEnd + 1 : end);
    
    W = [];
    W.withSideInfo = 1;
    W.sideInfoW = sideInfoW; 
    W.sideInfo = sideInfo;
else
    lambdaW = reshape(W(k*Y*U+1:end), k, k);
    W = [];
    W.withSideInfo = 0;
end

W.userW = userW; % W.userW(1,:,:) = 1;
W.lambdaW = lambdaW;
W.usersU = usersU;
W.usersV = usersV;
W.labels = labels;

% predictor = @lflPredictor;

% [predictions, argmax, probabilities] = predictor(W);

disp('evaluating...');
trainErrors = testLFL(@lflPredictor, W, Tr);
testErrors = testLFL(@lflPredictor, W, Te);

format = strcat('\n train/test 0-1 error = %4.4f / %4.4f',...
    ', f1score = %4.4f / %4.4f',', precision = %4.4f / %4.4f',...
    ', recall = %4.4f / %4.4f',...
    ', rmse = %4.4f / %4.4f',', mae = %4.4f / %4.4f ');
disp(sprintf(format, trainErrors.zoe, testErrors.zoe,...
    trainErrors.f1score, testErrors.f1score,...
    trainErrors.precision, testErrors.precision,...
    trainErrors.recall, testErrors.recall,...
    trainErrors.rmse, testErrors.rmse,...
    trainErrors.mae, testErrors.mae));

errors = [trainErrors.zoe, testErrors.zoe, trainErrors.rmse,...
    testErrors.rmse, trainErrors.mae, testErrors.mae];

if withSideInfo
    saveResults('synthetic-with_side_info.csv', errors);
else
    saveResults('synthetic-no_side_info.csv', errors);
end
% disp(probabilities);
% disp(predictions);
% disp(argmax);
