function [  ] = saveResults( filename, results )
%SAVERESULTS saves a vector of results into file
    addpath('results');

    if exist(filename) == 0
        headers = {'0-1 error (train),', '0-1 error (test),', ...
            'rmse (train),', 'rmse (test),',...
            'mae (train),', 'mae (test)'};
        dlmwrite(strcat('results/', filename), headers, 'newline',...
            'unix', 'delimiter', '');
    end
    fullpath = strcat('results/', filename);
    dlmwrite(strcat('results/', filename), results, 'newline', 'unix',...
        'delimiter', ',', 'precision', '%4.4f', '-append');
end

