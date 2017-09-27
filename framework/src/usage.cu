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
	cout << endl;
	cout << "Depth Adaptive Diffusion" << endl;
	cout << "-help                   | Display this message" << endl;
	cout << "--Image parameters ----------------------------------------- " << endl;
	cout << "-image         <path>   | specify image base name (+0/1.png)" << endl;
	cout << "[-gray]                 | use grayscale image" << endl;
	cout << "[-w            <int>]   | maximum width of image" << endl;
	cout << "[-h            <int>]   | maximum height of image" << endl;
	cout << "--Disparity calculation parameters --------------------------" << endl;
	cout << "[-lambda       <float>] | scaling of data term" << endl;
	cout << "[-tau_d        <float>] | scaling of p update step " << endl;
	cout << "[-tau_p        <float>] | scaling of phi update step"  << endl;
	cout << "[-gamma_min    <int>]   | minimal disparity" << endl;
	cout << "[-gamma_max    <int>]   | maximum disparity" << endl;
	cout << "[-iterations   <int>]   | maximum number of iterations" << endl;
	cout << "--Diffusion parameters --------------------------------------" << endl;
	cout << "[-disparities  <path>]  | disparity values file " << endl;
	cout << "[-radius       <float>] | strength of the diffusion " << endl;
	cout << "[-tau          <float>] | scaling of update step " << endl;
	cout << "[-baseline     <float>] | camera baseline in mm " << endl;
	cout << "[-doffs        <float>] | difference in principal points (L/R)" << endl;
	cout << "[-focal_length <float>] | focal length in pixels " << endl;
	cout << "[-focal_plane  <float>] | focal plane between [0,1]" << endl;
	cout << "[-iterations   <int>]   | maximum number of iterations" << endl;
	cout << "-------------------------------------------------------------" << endl;
	cout << endl;
}

// TODO: update readme with usage
void read_parameters(config &conf, int argc, char **argv)
{
	// Check if there are arguments
	if (argc <= 1)
	{
		print_usage();
		exit(0);
	}

	bool foo;
	bool ret = getParam("help", foo, argc, argv);
	if(ret)
	{
		print_usage();
		exit(0);
	}


	// Set defaults
	conf.max_w = 3000;
	conf.max_h = 2000;
	conf.gray = false;
	conf.disparities_from_file = false;
	conf.lambda = 30.f;
	conf.tau_p = 1.f / sqrtf(3.f);
	conf.tau_d = 1.f / sqrtf(3.f);
	conf.gamma_min = 0;
	conf.gamma_max = 250;
	conf.max_iterations = 1000;

	// For diffusion
	conf.radius = 2.f;
	conf.tau = 0.25f; // maximum possible value
	conf.baseline = 193.001f;
	conf.doffs = 124.343f;
	conf.focal_length = 3979.911f;
	conf.focal_plane = 0.5f;

	// Process input image
	ret = getParam("image", conf.image, argc, argv);
	if (!ret)
	{
		cerr << "ERROR: no image specified" << endl;
		exit(1);
	}

	conf.disparities_from_file = getParam("disparities", conf.disparities, argc,
			argv);

	getParam("h", conf.max_h, argc, argv);

	getParam("w", conf.max_w, argc, argv);

	getParam("gray", conf.gray, argc, argv);

	getParam("lambda", conf.lambda, argc, argv);

	getParam("tau_p", conf.tau_p, argc, argv);

	getParam("tau_d", conf.tau_d, argc, argv);

	getParam("tau", conf.tau, argc, argv);
	if (conf.tau > 0.25f)
	{
		cerr
				<< "ERROR: tau is outside the range of possible values, this will not converge!"
				<< endl;
	}

	getParam("gamma_min", conf.gamma_min, argc, argv);

	getParam("gamma_max", conf.gamma_max, argc, argv);

	if (conf.gamma_min >= conf.gamma_max)
	{
		cerr << "ERROR: invalid disparity range" << endl;
		exit(1);
	}

	getParam("iterations", conf.max_iterations, argc, argv);

	getParam("radius", conf.radius, argc, argv);

	getParam("focal_length", conf.focal_length, argc, argv);

	getParam("focal_plane", conf.focal_plane, argc, argv);

	getParam("doffs", conf.doffs, argc, argv);

	getParam("baseline", conf.baseline, argc, argv);
}
