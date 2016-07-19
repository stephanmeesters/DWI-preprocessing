# DWI-preprocessing
Processing pipeline for diffusion weighted imaging and optic radiation reconstruction.

Based on [NiPype](https://github.com/nipy/nipype).

* Grab neurological data (T1-weighted MRI, diffusion MRI)
* Normalize T1 to approximate MNI space (affine registration)
* Realign DWI series to approximate MNI space (affine registration)
* Tensor fitting (FSL)
* Constrained Spherical Deconvolution (MRTrix)
* Probabilistic tractography (MRTrix)
* FBC measures ([Github link](https://github.com/stephanmeesters/spuriousfibers))

Prerequisites
=============
* Install FSL
* Install MRTrix
* Compile FBC measures ([instructions](https://github.com/stephanmeesters/spuriousfibers))

Instructions
============

* Clone repo
* Within pipeline.py, set the paths to your T1-weighted MRI and diffusion MRI datasets
* In the setting section of pipeline.py, set ```perform_tracking``` and ```perform_fbc``` to ```false```.
* Run ```python pipeline.py``` in the terminal
* Use [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) or an equivalent program te create seed ROIs for optic radiation tractography. Create seed regions for the left and right lateral geniculate nucleus, the left and right visual cortex V1, and exclude regions for the opposite brain hemispheres.
* Specify the creates ROIs in pipeline.py, set ```perform_tracking``` and ```perform_fbc``` to ```true``` and re-run the pipeline.
