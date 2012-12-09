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
    triples = zeros([n 3]);
    for dyad = 1 : n
        triples(dyad,:) = [usersU(dyad) usersV(dyad) labels(dyad)];
    end
    % compress! ignore unknown links (left for cross-validation)
    triples = triples(triples(:,3) > 0, :);
    n = size(triples,1);
    if withGradient
        GuW = zeros(size(userUW));
        GvW = zeros(size(userVW));
        GlW = zeros(size(lambdaW));
        
        if withSideInfo
            GsW = zeros(size(sW));
        else
            GsW = 0;
        end
    else
        GuW = 0;
        GvW = 0;
        GlW = 0;
        GsW = 0;
    end
    
    fun = 0;
    
    % working in batches since parfor requires too much memory
    numBatches = 10;
    for batch = 1:numBatches % arbitrary number of batches
        % determine batch size
        defaultSize = ceil(n / numBatches);
        offset = 1 + (defaultSize * (batch - 1));
        batchSize = min([defaultSize (n - offset)]);
        
        if offset > n; break; end;
        
        % we'll store the values of all iterations in an array
        % which we'll sum later after the loops
        funBatch = zeros(1,batchSize);
        
        % create a batch of each variable
        range = offset:offset + batchSize - 1;
        usersUBatch = triples(range,1);
        usersVBatch = triples(range,2);
        labelsBatch = triples(range,3);
        userUWBatch = zeros([k Y batchSize]);
        userVWBatch = zeros([k Y batchSize]);
        sideInfoBatch = zeros([size(sW) batchSize]); % used regardless of withSideInfo
        if withGradient
            sizeGuW = size(GuW);
            GuWBatch = zeros([sizeGuW n]);
            GvWBatch = zeros([sizeGuW n]);
        else 
            sizeGuW = 0;
        end
        
        % prepare data for the batch
        for i = 1 : batchSize
            u = usersUBatch(i); %usersU(i);
            v = usersVBatch(i); %usersV(i);
            userUWBatch(:,:,i) = userUW(:,:,u);
            userVWBatch(:,:,i) = userVW(:,:,v);
            if withSideInfo
                sideInfoBatch(:,i) = [ sideInfo(u,:)'; sideInfo(v,:)';];
            end
        end
        
        parfor dyad = 1 : batchSize
            u = usersUBatch(dyad);
            v = usersVBatch(dyad);
            y = labelsBatch(dyad);
            uW = userUWBatch(:,:,dyad);
            vW = userVWBatch(:,:,dyad);

            % Vector whose ith element is Pr[label = i | u, v; w]
            if withSideInfo
    %             s = sideInfo(u,:)';
                s = sideInfoBatch(:,dyad);

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

            funBatch(:,dyad) = - log(p(y)) + reg;

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
                    GuWBatch(:,:,:,dyad) = currentGuW;

                    currentGvW = zeros(sizeGuW);
                    currentGvW(:,:,v) = Gv; % only the current user is filled
                    GvWBatch(:,:,:,dyad) = currentGvW;
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
        fun = fun + sum(funBatch);
        
        % update iterators of weights (the outputs)
        if withGradient
            GuW = GuW + sum(GuWBatch, 4);
            GvW = GvW + sum(GvWBatch, 4);
            
        end
    end
    
    
    if withGradient
        % sum accross users
%         sumAccross = sum(iterGuW, 4);
%         GuW = GuW + sumAccross;
%         
%         sumAccross = sum(iterGvW, 4);
%         GvW = GvW + sumAccross;
        
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

