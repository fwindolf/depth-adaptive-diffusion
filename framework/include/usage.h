#ifndef USAGE_H
#define USAGE_H

#include "util.h"
#include <string>
#include <iostream>

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
void read_parameters(std::string &image, bool &gray, float &lambda,
		float &tau_p, float &tau_d, int &gamma_min, int &gamma_max,
		int &iterations, int argc, char **argv);

#endif // USAGE_H
