#include <fstream>
#include <iostream>
#include <chrono>

uint32_t get_millis()
{
	return static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
}

struct Bounds
{
	double min = -10;
	double max = 10;

	bool valid() const { return max >= min; }
	bool in_bounds(double val) const { return (min <= val) && (val < max); }
};

double get_rand_num(const Bounds& bounds)
{
	uint32_t delta = (bounds.max - bounds.min) * 100;
	double rand_num = (std::rand() % delta) / 100.0;
	return rand_num + bounds.min;
}

void fill_file(std::ofstream& file, int num_verts, const Bounds& radius_bounds)
{
	std::srand(get_millis());
	Bounds angle_bounds = { 0, 3600 };
	for (; num_verts > 0; num_verts--)
	{
		double radius = get_rand_num(radius_bounds);
		double angle = get_rand_num(angle_bounds);

		double x = cos(angle) * radius;
		double y = sin(angle) * radius;

		file << "v " << x << ' ' << y << " 0.0 1.0" << std::endl;
	}
}

int main(int argc, char** argv)
{
	std::ofstream gen_verts_file("gen_verts_file.obj"); 
	if (gen_verts_file.fail()) { return -1; }
	int num_verts = 20;

	Bounds radius_bounds{ 5.0, 10.0 };
	
	for (int i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-') { return -2; }
		switch (argv[i][1])
		{
		case 'n':
			num_verts = std::abs(std::atoi(argv[++i]));
			break;
		case 'R':
			radius_bounds.max = std::atoi(argv[++i]);
			break;
		case 'r':
			radius_bounds.min = std::atoi(argv[++i]);
			break;
		default:
			return -3;
		}
	}

	if (!radius_bounds.valid() || radius_bounds.min < 0)
	{
		return -4;
	}

	fill_file(gen_verts_file, num_verts, radius_bounds);

	gen_verts_file.close();

	return 0;
}