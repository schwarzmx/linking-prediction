function [ fun, grad ] = lflObjectiveFunction( W, k, Y, U, lambda,...
                            usersU, usersV, labels)
%LFLOBJECTIVEFUNCTION Directed case of the lfl model with log likelihoods

% extract weights from W
    userW = reshape(W(1 : k*Y*U), k, Y, U);
    lambdaW = reshape(W(k*Y*U+1 : end), k, k, U, U);
    
    n = length(usersU);
    GuW = zeros(size(userW));
    GlW = zeros(size(lambdaW));
    
    fun = 0;

    for index = 1 : n
        u = usersU(index);
        v = usersV(index);
        y = labels(index);

        uW = userW(:,:,u);
        vW = userW(:,:,v);
        lW = lambdaW(:,:,u,v);

        % Vector whose ith element is Pr[rating = i | u, v; w]
        p = exp(diag(uW' * lW * vW));
        p = p/sum(p);
        p = p';
        
        % only take into account the prob of the current rating
        fun = fun - log(p(y)); ...
%             + (lambda / 2) * (norm(uW, 'fro'))^2 + norm(lW, 'fro'))^2;
        
        % do log gradient
        I = ((1:Y) == y); % I(y = z) in the paper
%         Gu = bsxfun(@times, lW * vW, (p - I));
%         Gl = bsxfun(@times, (uW * vW'), (p - I));
%         prod = (vW' * lW)';
        prod = lW * vW;
        Gu = -bsxfun(@times, prod, I)+ bsxfun(@times, prod, p);
        
        outerProd = uW(:,y) * vW(:,y)';
        Gl = (outerProd) - outerProd * p(y);
        
%         % Regularization
%         % Doesn't affect the bias term (nor the constant '1' term)
%         Gu(2:end-1,:) = Gu(2:end-1,:) + lambda * uW(2:end-1,:);
%         Gm(2:end-1,:) = Gm(2:end-1,:) + lambda * mW(2:end-1,:);
%         % Gu(:,2:end-1) = Gu(:,2:end-1) + lambda * uW(:,2:end-1);
%         % Gl = Gl + lambda * lW;
        
        GuW(:,:,u)=GuW(:,:,u) + Gu;
        GlW(:,:,u,v)=GlW(:,:,u,v) + Gl;
    end
    
    if nargout == 2
        grad = [GuW(:); GlW(:)];
    end

end

