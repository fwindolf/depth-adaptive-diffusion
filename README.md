# depth-adaptive-diffusion
Implementation of "A Convex Formulation of Continuous Multi-label Problems" by T.Pock et al. from ECCV 2008

## Setup
Create build directory and run cmake on the framework folder.
`mkdir build && cd build && cmake ../framework/`

This will setup a makefile for the project. To build the executable run 
`make all`

This will produce two executables: `debug` and `depth-adaptive-diffusion`. 

## Structure
The framework folder contains all the sources, images contains some stereo images from the middlebury.edu stereo image dataset 2014 (only perfectly aligned images work). 

### Included headers
Some of the code is externalized to external files in `include` with respective implementations in `source`:
- `gradient` contains code for calculating gradient and divergence (unused in master branch).
- `image` abstracts opencv code and contains some of the functions provided by the CUDA course of CV chair of TUM.
- `kernels` contains almost all kernel code.
- `reduce` contains kernel code for normalizing data.
- `timer` for some crude timing operations.
- `usage` for printing usage and reading the command line arguments.
- `util` for more readable reading/writing to data structures.


## Running
Run `./depth-adaptive-diffusion -help` to print all possible command line arguments. Arguments in brackets are optional and the defaults are mostly set to fit the Motorcycle image.

```
-help                   | Display this message
--Image parameters -----------------------------------------
-image         <path>   | specify image base name (+0/1.png)
[-gray]                 | use grayscale image
[-w            <int>]   | maximum width of image
[-h            <int>]   | maximum height of image
--Disparity calculation parameters --------------------------
[-lambda       <float>] | scaling of data term
[-tau_d        <float>] | scaling of p update step
[-tau_p        <float>] | scaling of phi update step
[-gamma_min    <int>]   | minimal disparity
[-gamma_max    <int>]   | maximum disparity
[-iterations   <int>]   | maximum number of iterations
--Diffusion parameters --------------------------------------
[-disparities  <path>]  | disparity values file
[-radius       <float>] | strength of the diffusion
[-tau          <float>] | scaling of update step
[-baseline     <float>] | camera baseline in mm
[-doffs        <float>] | difference in principal points (L/R)
[-focal_length <float>] | focal length in pixels
[-focal_plane  <float>] | focal plane between [0,1]
[-iterations   <int>]   | maximum number of iterations
-------------------------------------------------------------
```

To get usable results, a sample call (from build directory) would be
`./depth_adaptive_diffusion -image ../images/Motorcycle-perfect/im -iterations 5000 -w 1000 -lambda 50 -gamma_min 4 -gamma_max 41 -dg 2`

The `gamma_min` and `gamma_max` values can be calculated as follows:
1. Find out the image dimensions of the dataset as well as the maximum disparity values. For the motorcycle the image is 2900x2000 is with disparities ranging from 8 to 245.
2. Divide the minmal and maximum disparities by the scaling factor of the image. This would be ~3 for a set width of 1000.
3. Divide the resulting minimal and maximum disparities by the resolution of gamma. For a dg of 2, divide by another 2.

For gamma_min, this would be `23/(3 * 2) = 4` (round down).
For gamma_max, this would be `245/(3 * 2) = 41` (round up).

By decreasing lambda the disparities get alot smoother, the result however will get pretty much unusable for `lambda < 10`.

The output of the programm will be saved to build folder (`out.png`, `depths.png`, `disparities.png`)

## Issues

The energy does not converge to very little values (and sometimes does not even continually decrease at all). However the results are still pretty good. The main criterion for stopping is the number of iterations.
