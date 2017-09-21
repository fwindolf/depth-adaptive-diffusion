# depth-adaptive-diffusion
Implementation of "A Convex Formulation of Continuous Multi-label Problems" by T.Pock et al. from ECCV 2008

## Setup
Create build directory and run cmake on the framework folder.
`mkdir build && cd build && cmake ../framework/`

This will setup a makefile for the project. To build the executable run 
`make` 

## Structure
The framework folder contains all the sources, images contains some stereo images from the middlebury.edu stereo image dataset 2014 (only perfectly aligned images). 

### Other code
There also are some header files.
- `usage` for printing usage and reading the command line arguments
- `image` for some opencv abstraction
- `util` for more readable reading/writing to data structures
- `timer` for some crude timing operations

## Running
Run `./depth-adaptive-diffusion` in the build folder to start the executable. This will print the possible commandline arguments when run without some.

A sample call would be `./depth_adaptive_diffusion -i ../images/Bicycle1-perfect/im -iterations 1000`.

*NOTE: The implementation is currently flawed and doesn't do something yet!*
