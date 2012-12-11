#!/bin/sh

# make sure to have matlab in your path or symbol in /usr/local/bin/matlab
matlab -nodisplay -nodesktop -nojvm -nosplash < matlabcmd.txt \
        >> ../logs/log.txt 2>&1 &
