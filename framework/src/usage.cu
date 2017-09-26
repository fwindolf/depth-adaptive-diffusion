#include "usage.h"
#include <sstream>

// parameter processing
template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv);


template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc)) continue;
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}


// parameter processing: template specialization for T=bool
template<>
bool getParam<bool>(std::string param, bool &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc) || argv[i+1][0]=='-') { var = true; return true; }
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}



void print_usage()
{
	std::cout << "Depth Adaptive Diffusion" << std::endl;
	std::cout
			<< "-i <path/to/image.png>   | specify image base name. +'0.png' for Left and +'1.png' for Right"
			<< std::endl;
	std::cout << "[-gray]                  | use grayscale image" << std::endl;
	std::cout << "[-lambda <float>]        | parameter lambda" << std::endl;
	std::cout << "[-tau_d      <float>]    | parameter tau for the p update step" << std::endl;
	std::cout << "[-tau_p      <float>]    | parameter tau for the phi update step" << std::endl;
	std::cout << "[-gamma_min  <int>]      | minimal disparity" << std::endl;
	std::cout << "[-gamma_max  <int>]      | maximum disparity" << std::endl;
	std::cout << "[-iterations <int>]      | maximum number of iterations" << std::endl;
	std::cout << std::endl;
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
	conf.tau_p = 1.f/sqrtf(3.f);
	conf.tau_d = 1.f/sqrtf(3.f);
	conf.gamma_min = -4;
	conf.gamma_max = 4;
	conf.max_iterations = 1000;

	// Process input image
	bool ret = getParam("image", conf.image, argc, argv);
	if (!ret)
	{
		std::cerr << "ERROR: no image specified" << std::endl;
		exit(1);
	}

	conf.disparities_from_file = getParam("disparities", conf.disparities, argc, argv);


	getParam("gray", conf.gray, argc, argv);

	getParam("lambda", conf.lambda, argc, argv);

	getParam("tau_p", conf.tau_p, argc, argv);

	getParam("tau_d", conf.tau_d, argc, argv);

	getParam("gamma_min", conf.gamma_min, argc, argv);

	getParam("gamma_max", conf.gamma_max, argc, argv);

	if (conf.gamma_min >= conf.gamma_max)
	{
		std::cerr << "ERROR: invalid disparity range" << std::endl;
		exit(1);
	}

	getParam("max_iterations", conf.max_iterations, argc, argv);
}
