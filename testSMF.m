function [ errors, predictions, discretePredictions ] = testSMF( w, Te )
%TESTSMF Evaluates SMF model on test data

    errors = [];
    dataTe = sparse(Te.u, Te.v, Te.y);
    n = length(Te.u);

    predictions = w.userW * w.lambdaW * w.userW';
    discretePredictions = round(predictions);
%     predictions = nonzeros(predictions .* (dataTe > 0));
%     discretePredictions = nonzeros(discretePredictions .* (dataTe > 0));
%     truth = nonzeros(dataTe);  
    
    discretePredictions = discretePredictions(:);
    predictions = predictions(:);
    truth = nonzeros(dataTe);

    errors.zoe = sum(discretePredictions ~= truth)/n;
    errors.mae = sum(abs(discretePredictions - truth))/n;
    errors.rmse = sqrt(sum((predictions - truth).^2)/n);
end

