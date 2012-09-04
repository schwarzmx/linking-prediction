%--- LOAD DATASET ---%
% Tr = csv2struct('dataset/train_1x1.csv');
Tr = csv2struct('dataset/train_3x3.csv');

% number of latent features
k = 2;
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

% Figure out number of users/movies and number of possible ratings
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
% finite differences gradient
options = optimset('Display','iter',...
            'FunValCheck','on',...
            'Diagnostics','on');

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
[W, fval] = fminunc(fun, initialW, options);

disp(W);
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
predictor = @lflPredictor;

[predictions, argmax, probabilities] = predictor(W);

disp(probabilities);
disp(predictions);
disp(argmax);