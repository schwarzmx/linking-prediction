function [ errors, probabilities, predictions ] = testLFL( predictor, w, Te )
%TESTLFL Evaluates LFL model on test data

    errors = [];
    dataTe = sparse(Te.u, Te.v, Te.y);
%     n = length(Te.u);
    
    Y = 2;

    [predictions, argmaxPredictions, fullProbabilities] = predictor(w);
    predictions = nonzeros(predictions .* (dataTe > 0));
    argmaxPredictions = nonzeros(argmaxPredictions .* (dataTe > 0));

    % Compute the probabilities
    probabilities = zeros(nnz(dataTe), Y);
    for y = 1:Y
        probabilities(:,y) = nonzeros(...
            bsxfun(@times, squeeze(fullProbabilities(:,:,y)), dataTe > 0));
    end

    truth = nonzeros(dataTe);
    n = length(truth);
    
    truepos = sum((argmaxPredictions .* (argmaxPredictions == truth)) == 2);
    trueneg = sum((argmaxPredictions .* (argmaxPredictions == truth)) == 1);
    falsepos = sum((argmaxPredictions .* (argmaxPredictions ~= truth)) == 2);
    falseneg = sum((argmaxPredictions .* (argmaxPredictions ~= truth)) == 1);
    errors.precision = truepos / (truepos + falsepos);
    errors.recall = truepos / (truepos + falseneg);
    errors.f1score = 2 * (errors.precision * errors.recall) /...
        (errors.precision + errors.recall);
    errors.zoe = sum(argmaxPredictions ~= truth)/n;
    errors.mae = sum(abs(round(predictions) - truth))/n;
    errors.rmse = sqrt(sum((predictions - truth).^2)/n);
end

