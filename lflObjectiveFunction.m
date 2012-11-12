function [ fun, grad ] = lflObjectiveFunction( W, k, Y, U, lambda,...
                            usersU, usersV, sideInfo, labels, withSideInfo)
%LFLOBJECTIVEFUNCTION Directed case of the lfl model with log likelihoods

% extract weights from W
    userW = reshape(W(1 : k*Y*U), k, Y, U);
    if withSideInfo
        lambdaStart = k*Y*U+1;
        lambdaEnd = lambdaStart + k * k - 1;
        lambdaW = reshape(W(lambdaStart : lambdaEnd), k, k);
        sW = W(lambdaEnd + 1 : end);
    else
        lambdaW = reshape(W(k*Y*U+1 : end), k, k);
    end
    
    n = length(usersU);
    if nargout == 2
        GuW = zeros(size(userW));
        GlW = zeros(size(lambdaW));
        
        if withSideInfo
            GsW = zeros(size(sW));
        end
    end
    
    fun = 0;

    for index = 1 : n
        u = usersU(index);
        v = usersV(index);
        y = labels(index);
%         s = [sideInfo(u,:)'; sideInfo(v,:)']; 
        s = sideInfo(u,:)';
        
        % ignore unknown links (left for cross-validation)
        if y == 0 && ~withSideInfo 
            continue; 
        end
        
        uW = userW(:,:,u);

        % Vector whose ith element is Pr[label = i | u, v; w]
        if withSideInfo
            p = exp(diag(uW' * lambdaW * uW + sW' * s));
        else    
            p = exp(diag(uW' * lambdaW * uW));
        end
        p = p/sum(p);
        p = p';
        
        % only take into account the prob of the current label
        if withSideInfo
            reg = + (lambda / 2) * (norm(uW, 'fro')^2 + norm(sW, 'fro')^2);
        else
            reg = + (lambda / 2) * norm(uW, 'fro')^2;
        end
        if y == 0
            [val y] = max(p);
        end
        fun = fun - log(p(y)) + reg;

        
        % do log gradient
        if nargout == 2
%             if y ~= 0
                I = ((1:Y) == y); % I(y = z) in the paper
                Gu = bsxfun(@times,(lambdaW + lambdaW') * uW, (p - I));

                % TODO: do this iteratively for |Y| > 2
                Gl_ = zeros(k,k,Y);
                Gl_(:,:,1) = uW(:,1) * uW(:,1)';
                Gl_(:,:,2) = uW(:,2) * uW(:,2)';
                Gl = -Gl_(:,:,y) + Gl_(:,:,1) * p(1) + Gl_(:,:,2) * p(2);

                % regularization
                Gu = Gu + lambda * uW;

                GuW(:,:,u)=GuW(:,:,u) + Gu;
    %             GlW(:,:,y)=GlW(:,:,y) + Gl;
                GlW = GlW + Gl;
%             end
            if withSideInfo
                Gs = -s + s * p(1) + s * p(2);
                % regularization
                Gs = Gs + lambda * sW;
                GsW = GsW + Gs;
            end
        end
    end
    
    if nargout == 2
        GuW = GuW(:);
        GlW = GlW(:);
        if withSideInfo
            grad = [GuW; GlW; GsW];
        else
            grad = [GuW; GlW];
        end
    end

end

