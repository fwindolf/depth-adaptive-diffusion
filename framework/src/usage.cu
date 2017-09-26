#include "usage.h"
#include <sstream>

using namespace std;

// parameter processing
template<typename T>
bool getParam(string param, T &var, int argc, char **argv);

template<typename T>
bool getParam(string param, T &var, int argc, char **argv)
{
	const char *c_param = param.c_str();
	for (int i = argc - 1; i >= 1; i--)
	{
		if (argv[i][0] != '-')
			continue;
		if (strcmp(argv[i] + 1, c_param) == 0)
		{
			if (!(i + 1 < argc))
				continue;
			stringstream ss;
			ss << argv[i + 1];
			ss >> var;
			return (bool) ss;
		}
	}
	return false;
}

// parameter processing: template specialization for T=bool
template<>
bool getParam<bool>(string param, bool &var, int argc, char **argv)
{
	const char *c_param = param.c_str();
	for (int i = argc - 1; i >= 1; i--)
	{
		if (argv[i][0] != '-')
			continue;
		if (strcmp(argv[i] + 1, c_param) == 0)
		{
			if (!(i + 1 < argc) || argv[i + 1][0] == '-')
			{
				var = true;
				return true;
			}
			stringstream ss;
			ss << argv[i + 1];
			ss >> var;
			return (bool) ss;
		}
	}
	return false;
}

void print_usage()
{
	cout << "Depth Adaptive Diffusion" << endl;
	cout
			<< "-i <path/to/image.png>   | specify image base name. +'0.png' for Left and +'1.png' for Right"
			<< endl;
	cout << "[-gray]                  | use grayscale image" << endl;
	cout << "[-lambda <float>]        | parameter lambda" << endl;
	cout
			<< "[-tau_d      <float>]    | parameter tau for the p update step"
			<< endl;
	cout
			<< "[-tau_p      <float>]    | parameter tau for the phi update step"
			<< endl;
	cout << "[-gamma_min  <int>]      | minimal disparity" << endl;
	cout << "[-gamma_max  <int>]      | maximum disparity" << endl;
	cout << "[-iterations <int>]      | maximum number of iterations"
			<< endl;
	cout << endl;
}

// TODO: update readme with usage
void read_parameters(config &conf, int argc, char **argv)
{
	// Check if there are arguments
	if (argc <= 1)
	{
		print_usage();
	}

	// Set defaults
	conf.gray = false;
	conf.disparities_from_file = false;
	conf.lambda = 30.f;
	conf.tau_p = 1.f / sqrtf(3.f);
	conf.tau_d = 1.f / sqrtf(3.f);
	conf.gamma_min = -4;
	conf.gamma_max = 4;
	conf.max_iterations = 1000;

	// For diffusion
	conf.sigma = 4;
	conf.tau = 0.25f; // maximum possible value
	conf.baseline = 193.001f;
	conf.doffs = 124.343f;
	conf.focal_length = 3979.911f;
	conf.focal_plane = 0.5f * (conf.gamma_max - conf.gamma_min + 1);

	// Process input image
	bool ret = getParam("image", conf.image, argc, argv);
	if (!ret)
	{
		cerr << "ERROR: no image specified" << endl;
		exit(1);
	}

	conf.disparities_from_file = getParam("disparities", conf.disparities, argc,
			argv);

	getParam("gray", conf.gray, argc, argv);

	getParam("lambda", conf.lambda, argc, argv);

	getParam("tau_p", conf.tau_p, argc, argv);

	getParam("tau_d", conf.tau_d, argc, argv);

	getParam("tau", conf.tau, argc, argv);
	if(conf.tau > 0.25f)
	{
		cerr << "ERROR: tau is outside the range of possible values, this will not converge!" << endl;
	}

	getParam("gamma_min", conf.gamma_min, argc, argv);

	getParam("gamma_max", conf.gamma_max, argc, argv);

	if (conf.gamma_min >= conf.gamma_max)
	{
		cerr << "ERROR: invalid disparity range" << endl;
		exit(1);
	}

	getParam("max_iterations", conf.max_iterations, argc, argv);

	getParam("sigma", conf.sigma, argc, argv);

	getParam("focal_length", conf.focal_length, argc, argv);

	getParam("doffs", conf.doffs, argc, argv);

	getParam("baseline", conf.baseline, argc, argv);
}
