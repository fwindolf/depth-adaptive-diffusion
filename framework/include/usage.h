#ifndef USAGE_H
#define USAGE_H

#include "util.h"
#include <string>
#include <iostream>

struct config {
	bool 	gray, 					// grayscale image used
			disparities_from_file; 	// dont calculate disparities
	float 	lambda, 				// weight of data term
			tau_p, 					// step size of phi update
			tau_d;					// step size of p update
	int 	gamma_min, 				// maximum disparity in -x direction
			gamma_max, 				// maximum disparity in +x direction
			max_iterations; 		// number of iterations until stopping
	std::string image,				// filename (prefix) of the image(s)
			disparities;			// filename of the disparity values
};

/**
 * Command line parameter processing
 */
template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv);

/**
 * Print the usage message
 */
void print_usage();

/**
 * Read the given parameters from command line
 */
void read_parameters(config &conf, int argc, char **argv);

#endif // USAGE_H
