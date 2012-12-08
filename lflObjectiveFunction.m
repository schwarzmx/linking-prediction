function [ fun, grad ] = lflObjectiveFunction( W, varargin)
%LFLOBJECTIVEFUNCTION Directed case of the lfl model with log likelihoods
    % load vars
    k = varargin{1};
    Y = varargin{2};
    U = varargin{3};
    lambda = varargin{4}; 
    usersU = varargin{5};
    usersV = varargin{6}; 
    sideInfo = varargin{7};
    labels = varargin{8};
    withSideInfo = varargin{9};
    
    if nargout == 2
        withGradient = 1;
    else
        withGradient = 0;
    end
    
% extract weights from W
    userUW = reshape(W(1 : k*Y*U), k, Y, U);
    userVW = reshape(W(k*Y*U + 1 : 2 * k*Y*U), k, Y, U);
    if withSideInfo
        lambdaStart = 2 * k*Y*U+1;
        lambdaEnd = lambdaStart + k * k - 1;
        lambdaW = reshape(W(lambdaStart : lambdaEnd), k, k);
        sW = W(lambdaEnd + 1 : end);
    else
        lambdaW = reshape(W(2 * k*Y*U+1 : end), k, k);
        sW = 0;
    end
    
    n = length(usersU);
    if withGradient
        GuW = zeros(size(userUW));
        GvW = zeros(size(userVW));
        GlW = zeros(size(lambdaW));
        sizeGuW = size(GuW);
        iterGuW = zeros([sizeGuW n]);
        iterGvW = zeros([sizeGuW n]);
        
        if withSideInfo
            GsW = zeros(size(sW));
        else
            GsW = 0;
        end
    end
    
    % we'll store the values of all iterations in an array
    % which we'll sum later after the loops
    fun = zeros(1,n);
    userWIterU = zeros([k Y n]);
    userWIterV = zeros([k Y n]);
    if withSideInfo
        sideInfoIter = zeros([size(sW) n]);
    end
    for i = 1 : n
        u = usersU(i);
        v = usersV(i);
        userWIterU(:,:,i) = userUW(:,:,u);
        userWIterV(:,:,i) = userVW(:,:,v);
        if withSideInfo
            sideInfoIter(:,i) = [ sideInfo(u,:)'; sideInfo(v,:)';];
        end
    end
    
    for dyad = 1 : n
        u = usersU(dyad);
        v = usersV(dyad);
        y = labels(dyad);
        
        
        % ignore unknown links (left for cross-validation)
        if y == 0
            continue; 
        end
        
        uW = userWIterU(:,:,dyad);
        vW = userWIterV(:,:,dyad);

        % Vector whose ith element is Pr[label = i | u, v; w]
        if withSideInfo
%             s = sideInfo(u,:)';
            s = sideInfoIter(:,dyad);
            
            p = exp(diag(uW' * lambdaW * vW + sW' * s));
        else    
            p = exp(diag(uW' * lambdaW * vW));
        end
        p = p/sum(p);
        p = p';
        
        % only take into account the prob of the current label
        if withSideInfo
            reg = + (lambda / 2) * ...
                (norm(uW, 'fro')^2 + norm(vW, 'fro')^2 + norm(sW, 'fro')^2);
        else
            reg = + (lambda / 2) * (norm(uW, 'fro')^2 + norm(vW, 'fro')^2);
        end
        
        fun(:,dyad) = - log(p(y)) + reg;
        
        % do log gradient
        if withGradient
                I = ((1:Y) == y); % I(y = z) in the paper
%                 Gu = bsxfun(@times,(lambdaW + lambdaW') * uW, (p - I));
                Gu = bsxfun(@times, lambdaW * vW, (p - I));
                Gv = bsxfun(@times, uW' * lambdaW, (p - I)')';
%                 if u == v
%                     Gu = bsxfun(@times,(lambdaW + lambdaW') * vW, (p - I));
%                 else
%                     
% %                     Gu_j = bsxfun(@times, (uW' *lambdaW)' , (p - I));
%                 end

                % TODO: do this iteratively for |Y| > 2
                Gl_ = zeros(k,k,Y);
                Gl_(:,:,1) = uW(:,1) * vW(:,1)';
                Gl_(:,:,2) = uW(:,2) * vW(:,2)';
                Gl = -Gl_(:,:,y) + Gl_(:,:,1) * p(1) + Gl_(:,:,2) * p(2);

                % regularization
                Gu = Gu + lambda * uW;
                Gv = Gv + lambda * vW;

%                 GuW(:,:,u)=GuW(:,:,u) + Gu;
                currentGuW = zeros(sizeGuW);
                currentGuW(:,:,u) = Gu; % only the current user is filled
                iterGuW(:,:,:,dyad) = currentGuW;
                
                currentGvW = zeros(sizeGuW);
                currentGvW(:,:,v) = Gv; % only the current user is filled
                iterGvW(:,:,:,dyad) = currentGvW;
                GlW = GlW + Gl;
            if withSideInfo
                Gs = -s + s * p(1) + s * p(2);
                % regularization
                Gs = Gs + lambda * sW;
                GsW = GsW + Gs;
            end
        end
    end
    % total function sum
    fun = sum(fun);
    
    if withGradient
        % sum accross users
        sumAccross = sum(iterGuW, 4);
        GuW = GuW + sumAccross;
        
        sumAccross = sum(iterGvW, 4);
        GvW = GvW + sumAccross;
        
        GuW = GuW(:);
        GvW = GvW(:);
        GlW = GlW(:);
        if withSideInfo
            grad = [GuW; GvW; GlW; GsW];
        else
            grad = [GuW; GvW; GlW];
        end
    end
end

