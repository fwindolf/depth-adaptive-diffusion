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

void read_parameters(std::string &image, bool &gray, float &lambda,
		float &tau_p, float &tau_d, int &gamma_min, int &gamma_max,
		int &iterations, int argc, char **argv)
{
	// Check if there are arguments
	if (argc <= 1)
	{
		print_usage();
	}

	// Process input image
	image = "";
	bool ret = getParam("i", image, argc, argv);
	if (!ret)
	{
		std::cerr << "ERROR: no image specified" << std::endl;
		exit(1);
	}

	gray = false;
	getParam("gray", gray, argc, argv);

	lambda = 30.f;
	getParam("lambda", lambda, argc, argv);

	tau_p = 1.f/sqrtf(3.f);
	getParam("tau_p", tau_p, argc, argv);

	tau_d = 1.f/sqrtf(3.f);
	getParam("tau_d", tau_d, argc, argv);

	gamma_min = -4;
	getParam("gamma_min", gamma_min, argc, argv);
	gamma_max = 4;
	getParam("gamma_max", gamma_max, argc, argv);
	if (gamma_min >= gamma_max)
	{
		std::cerr << "ERROR: invalid disparity range" << std::endl;
		exit(1);
	}

	iterations = 100;
	getParam("iterations", iterations, argc, argv);
}
