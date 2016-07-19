# DWI-preprocessing
Processing pipeline for diffusion weighted imaging and optic radiation reconstruction.

Based on [NiPype](https://github.com/nipy/nipype).

* Grab neurological data (T1-weighted MRI, diffusion MRI)
* Normalize T1 to MNI space (affine registration)
* Realign DWI series to MNI space (affine registration)
* Tensor fitting (FSL)
* Constrained Spherical Deconvolution (MRTrix)
* Probabilistic tractography (MRTrix)
* FBC measures ([Github link](https://github.com/stephanmeesters/spuriousfibers)

