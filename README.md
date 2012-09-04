# Dyadic Prediction
This project contains two implementations of dyadic linking prediction algorithms:
Latent Feature Model (Menon, Elkan) and Supervised Matrix Factorization (Zhu et al.).

To run each of the algorithms execute either of the main scripts and provide their
corresponding parameters. Both algorithm implementations expect a CSV with three
columns (u,v,y), which represent edges (u,v) with their respective label. For some
syntetic sample datasets look at the `dataset` folder.

## Requirements
The code was coded in Matlab using the Optimization Toolbox, but could be used also
with Octave. Other optimization tools could be also easily adapted.

## Acknowledgements
The LFL implementation is based on A.K. Menon's [sample code](http://cseweb.ucsd.edu/~akmenon/code/).

## License
MIT License (see attached LICENSE file for details). It would be nice to give me
some acknowledgement should this code be used in any of your projects/research.
