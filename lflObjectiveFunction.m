function [ fun, grad ] = lflObjectiveFunction( W, k, Y, U, lambda,...
                            usersU, usersV, labels)
%LFLOBJECTIVEFUNCTION Directed case of the lfl model with log likelihoods

% extract weights from W
    userW = reshape(W(1 : k*Y*U), k, Y, U);
    lambdaW = reshape(W(k*Y*U+1 : end), k, k, U, U);
    
    n = length(usersU);
    if nargout == 2
        GuW = zeros(size(userW));
        GlW = zeros(size(lambdaW));
    end
    
    fun = 0;

    for index = 1 : n
        u = usersU(index);
        v = usersV(index);
        y = labels(index);

        % ignore unknown links (left for cross-validation)
        if y == 0 continue; end
        
        uW = userW(:,:,u);
        vW = userW(:,:,v);
        lW = lambdaW(:,:,u,v);

        % Vector whose ith element is Pr[label = i | u, v; w]
        p = exp(diag(uW' * lW * vW));
        p = p/sum(p);
        p = p';
        
        % only take into account the prob of the current label
        fun = fun - log(p(y))...
            + (lambda / 2) * ((norm(uW, 'fro'))^2 + norm(lW, 'fro')^2);
        
        % do log gradient
        if nargout == 2
            I = ((1:Y) == y); % I(y = z) in the paper
            Gu = bsxfun(@times,(lW + lW')* vW, (p - I));
            
            % TODO: do this iteratively for |Y| > 2
            Gl_ = zeros(k,k,Y);
            Gl_(:,:,1) = uW(:,1) * vW(:,1)';
            Gl_(:,:,2) = uW(:,2) * vW(:,2)';
            Gl = -Gl_(:,:,y) + Gl_(:,:,1) * p(1) + Gl_(:,:,2) * p(2);
            
            % regularization
            Gu = Gu + lambda * uW;
            Gl = Gl + lambda * lW;

            GuW(:,:,u)=GuW(:,:,u) + Gu;
            GlW(:,:,u,v)=GlW(:,:,u,v) + Gl;
        end
    end
    
    if nargout == 2
        grad = [GuW(:); GlW(:)];
    end

end

