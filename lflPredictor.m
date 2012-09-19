% Predictions from the LFL model using the given weight vector
% This is computed over all users U and users V (which are implicit in the weight)
% Output consists of the real-valued predictions (expected value under the
% probability model); discrete valued  'argmax predictions', viz the most
% likely label under the probability model; and the actual probabilities
% themselves
function [predictions, argmaxPredictions, probabilities] = lflPredictor(w)

    Y = size(w.userW, 2);
    U = size(w.userW, 3);
    n = length(w.usersU);
    
    probabilities = zeros(U, U, Y);
    for index = 1 : n;        
        u = w.usersU(index);
        v = w.usersV(index);
%         y = w.labels(index);
%         
%         if y == 0; continue; end
        
        uW = w.userW(:,:,u);
%         lW = w.lambdaW(:,:,y);
        lW = w.lambdaW;

        % Vector whose ith element is Pr[y = i | u, v; w]
        p = exp(diag(uW' * lW * uW));
        p = p/sum(p);
        probabilities(u, v, :) = p;
    end
%     for y = 1:Y
%         uW = squeeze(w.userW(:,y,:));
% %         lW = squeeze(w.lambdaW(:,:,y));
%         lW = squeeze(w.lambdaW);
%         probabilities(:,:,y) = exp(uW' * lW * uW);
%     end
    probabilities = bsxfun(@rdivide, probabilities, sum(probabilities, 3));
    predictions = sum(bsxfun(@times, reshape(1:Y, [1 1 Y]), probabilities), 3);
    [values, argmaxPredictions] = max(probabilities, [], 3);