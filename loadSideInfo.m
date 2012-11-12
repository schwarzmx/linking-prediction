function [ info ] = loadSideInfo( filename )
%LOADSIDEINFO load side information for the current users
    
    % skip the header
    info = csvread(filename, 1, 0);
end

