%--- OPTIMIZATION METHOD ---%
% method = 'bfgs';
method = 'lbfgs';
withSideInfo = 1; % yes
%withSideInfo = 0; % no
disp('loading dataset...');

%--- LOAD DATASET (synthetic)---%
Tr = csv2struct('dataset/final_train-val.csv');
Te = csv2struct('dataset/final_test.csv');
%Tr = csv2struct('dataset/final_train.csv');
%Te = csv2struct('dataset/final_val.csv');
% Tr = csv2struct('dataset/train_200.csv');
% Te = csv2struct('dataset/test_200.csv');
% Tr = csv2struct('dataset/train_100.csv');
% Te = csv2struct('dataset/test_100.csv');
% Tr = csv2struct('dataset/train_3x3.csv');
% Te = csv2struct('dataset/test_3x3.csv');
% Tr = csv2struct('dataset/train_1x1.csv');
% Te = csv2struct('dataset/train_1x1.csv');
% Tr = csv2struct('dataset/train_synthetic.csv');
% Te = csv2struct('dataset/test_synthetic.csv');

if withSideInfo
    % the side info should contain as many features as necessary
    % but it's important to have as many rows as there are users
    %sideInfo = loadSideInfo('dataset/side_info-synthetic_normalized.csv');
    sideInfo = loadSideInfo('dataset/final_sideinfo.csv');
else
    sideInfo = [];
end

% number of latent features
k = 10;
% penalty
lambda = 1e-2;

usersU = Tr.u;
usersV = Tr.v;
labels = Tr.y;

% Shuffle training set
n = length(usersU);
I = randperm(n);
Tr.u = Tr.u(I);
Tr.v = Tr.v(I);
Tr.y = Tr.y(I);

% Figure out number of users and number çof possible labels
U = max(usersU) - min(usersU) + 1;
% Y = max(labels) - min(labels) + 1;
Y = 2;

%-- INITIALIZE WEIGHTS --%
userUW = 1/k * randn(k, Y, U);
userVW = 1/k * randn(k, Y, U);
% userVW = userUW;
%-- lambdaW(:,:,i,j) is a k by k matrix for each edge label
lambdaW = 1/k * randn(k, k);
if withSideInfo
    % vector of 2 x number features
    S = 2 * size(sideInfo,2);
    sideInfoW = 1/k * randn(S,1);
    
    initialW = [userUW(:); userVW(:); lambdaW(:); sideInfoW];
else
    initialW = [userUW(:); userVW(:); lambdaW(:)];
end



%--- OPTIMIZATION OPTIONS ---%
if strcmp(method, 'lbfgs')
    %--- minFunc ---%
    addpath(genpath('minFunc_2012'));
    
    options.DerivativeCheck = 'off';
    options.numDiff = 0;
%     options.Corr = 100;
    options.Display = 'iter';
%     options.Display = 'final';
    options.MaxFunEvals = 10000;
    options.MaxIter = 100;
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

userUW = reshape(W(1:k*Y*U), k, Y, U);
userVW = reshape(W(k*Y*U+1:2*k*Y*U), k, Y, U);
if withSideInfo
    lambdaStart = 2 * k*Y*U+1;
    lambdaEnd = lambdaStart + k * k - 1;
    lambdaW = reshape(W(lambdaStart : lambdaEnd), k, k);
    sideInfoW = W(lambdaEnd + 1 : end);
    
    W = [];
    W.withSideInfo = 1;
    W.sideInfoW = sideInfoW; 
    W.sideInfo = sideInfo;
else
    lambdaW = reshape(W(2*k*Y*U+1:end), k, k);
    W = [];
    W.withSideInfo = 0;
end

W.userUW = userUW;
W.userVW = userVW;
W.lambdaW = lambdaW;
W.usersU = usersU;
W.usersV = usersV;
W.labels = labels;

% predictor = @lflPredictor;


disp('evaluating...');
trainErrors = testLFL(@lflPredictor, W, Tr);
testErrors = testLFL(@lflPredictor, W, Te);

format = strcat('\n train/test 0-1 error = %4.9f / %4.9f',...
    ', f1score = %4.9f / %4.9f',...
    ', accuracy = %4.9f / %4.9f',...
    ', precision = %4.9f / %4.9f',...
    ', recall = %4.9f / %4.9f',...
    ', rmse = %4.9f / %4.9f',...
    ', mae = %4.9f / %4.9f ');
disp(sprintf(format, trainErrors.zoe, testErrors.zoe,...
    trainErrors.f1score, testErrors.f1score,...
    trainErrors.accuracy, testErrors.accuracy,...
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

[predictions, argmax, probabilities] = lflPredictor(W);
% disp(probabilities);
% disp(predictions);
% disp(argmax);
