#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <fstream>
#include <list>
#include <math.h>
#include <algorithm>
#include <chrono>

#include "glm/glm.hpp"

struct Edge
{
	size_t vert1_i;
	size_t vert2_i;
	//glm::vec2 normal;
};


class ConvexHull
{
public:

    void add_vert(float _x, float _y)
    {
        verts.push_back({_x, _y});
    }

    size_t get_vert_num() const { return verts.size() / 2; }

	size_t find_x_extrema_vert(bool (*comp)(float, float))
	{
		size_t idx_ref = 0;
		float x_ref = verts[idx_ref].x;
		size_t size = verts.size();
		for (size_t i = 1; i < size; i++)
		{
			float next_x = verts[i].x;
			if (comp(x_ref, next_x))
			{
				x_ref = next_x;
				idx_ref = i;
			}
		}

		return idx_ref;
	}

	void swap_indeces(std::vector<size_t>& vert_idxs, size_t idx1, size_t idx2)
	{
		size_t tmp = vert_idxs[idx1];
		vert_idxs[idx1] = vert_idxs[idx2];
		vert_idxs[idx2] = tmp;
	}

	void find_solution()
	{
		size_t min_vert_i = find_x_extrema_vert([](float ref, float next){ return ref > next; });
		size_t max_vert_i = find_x_extrema_vert([](float ref, float next){ return ref < next; });

		edge_queue.push_back({min_vert_i, max_vert_i});
		edge_queue.push_back({max_vert_i, min_vert_i});

		std::vector<size_t> vert_idxs(verts.size());
		for (size_t i = 0; i < vert_idxs.size(); i++)
		{
			vert_idxs[i] = i;
		}

		size_t vert_idxs_size = vert_idxs.size();
		swap_indeces(vert_idxs, vert_idxs_size - 1, min_vert_i);
		swap_indeces(vert_idxs, vert_idxs_size - 2, max_vert_i);
		vert_idxs_size -= 2;

		std::vector<float> distances(vert_idxs.size());

		auto start = std::chrono::high_resolution_clock::now();

		for (auto edge_it = edge_queue.begin(); edge_it != edge_queue.end();)
		{
			const size_t A_vert_i = edge_it->vert1_i;
			const size_t B_vert_i = edge_it->vert2_i;
			const glm::vec2& A_vert = verts[A_vert_i];
			const glm::vec2& B_vert = verts[B_vert_i];
			
			// ----- ! GPU begin ! -----
			for (size_t i = 0; i < vert_idxs_size; i++)
			{
				size_t vert_i = vert_idxs[i];
				glm::vec2& T_vert = verts[vert_i];

				glm::vec3 AB_vec(B_vert - A_vert, 0.0f);
				glm::vec3 AT_vec(T_vert - A_vert, 0.0f);
				float A_cross = glm::cross(AB_vec, AT_vec).z;
				if (A_cross < 0) 
				{ 
					distances[i] = -1;
					continue; 
				}

				glm::vec3 T_proj = (glm::dot(AB_vec, AT_vec) / glm::dot(AB_vec, AB_vec)) * AB_vec;
				distances[i] = glm::length(AT_vec - T_proj);
			}
			// ----- ! GPU end ! -----

			size_t raw_C_vert_i = 0;
			float farthest_distance = -1;
			for (size_t i = 0; i < vert_idxs_size; i++)
			{
				if (distances[i] > farthest_distance)
				{
					farthest_distance = distances[i];
					raw_C_vert_i = i;
				}
			}

			if (farthest_distance < 0)
			{
				++edge_it;
				continue;
			}

			const size_t C_vert_i = vert_idxs[raw_C_vert_i];
			const glm::vec2& C_vert = verts[C_vert_i];

			edge_queue.push_back({A_vert_i, C_vert_i});
			edge_queue.push_back({C_vert_i, B_vert_i});
			edge_it = edge_queue.erase(edge_it);

			// ----- ! GPU begin ! -----
			for (size_t i = 0; i < vert_idxs_size; i++)
			{
				size_t vert_i = vert_idxs[i];
				if (distances[i] > 0)
				{
					const glm::vec2& T_vert = verts[vert_i];

					glm::vec3 BC_vec(C_vert - B_vert, 0.0f);
					glm::vec3 CA_vec(A_vert - C_vert, 0.0f);

					glm::vec3 BT_vec(T_vert - B_vert, 0.0f);
					glm::vec3 CT_vec(T_vert - C_vert, 0.0f);

					float B_cross = glm::cross(BC_vec, BT_vec).z;
					float C_cross = glm::cross(CA_vec, CT_vec).z;

					if (!(B_cross >= 0 && C_cross >= 0))
					{
						distances[i] = -1.0f;
					}
				}
			}
			// ----- ! GPU end ! -----
			int right_side = 0;
			int left_side = vert_idxs_size - 1;
			while (left_side >= right_side)
			{
				if (distances[left_side] < 0 && distances[right_side] >= 0)
				{
					swap_indeces(vert_idxs, left_side, right_side);
					--left_side;
					++right_side;
				}
				else
				{
					if (distances[left_side] >= 0) { --left_side; }
					if (distances[right_side] < 0) { ++right_side; }
				}
			}
			vert_idxs_size = right_side;

			if (vert_idxs_size == 0)
			{
				break;
			}
		}
	
		auto end = std::chrono::high_resolution_clock::now();

		std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

	}

	void get_result(const char* filename_in, const char* filename_out)
	{
		std::ifstream src(filename_in);
		std::ofstream dst(filename_out);

		dst << src.rdbuf();

		std::vector<Edge> convex_hull({ edge_queue.begin(), edge_queue.end() });

		std::sort(convex_hull.begin(), convex_hull.end(),
			[](const Edge& A, const Edge& B) {
				return A.vert1_i < B.vert1_i;
			});

		uint32_t start = convex_hull.begin()->vert1_i;
		uint32_t curr = start;
		uint32_t next = convex_hull.begin()->vert2_i;
		dst << std::endl << "l ";
		do
		{
			dst << (curr + 1) << " ";
			curr = next;
			auto it = std::lower_bound(convex_hull.begin(), convex_hull.end(), next,
				[](const Edge& A, const uint32_t val) {
					return A.vert1_i < val;
				});
			if (it == convex_hull.end())
			{
				std::cout << "Error: hull is not cycled. Bad idx: " << next << std::endl;
				exit(-9);
			}
			next = it->vert2_i;
		} while (curr != start);
		dst << std::endl;
	}

private:

    std::vector<glm::vec2> verts;
    std::list<Edge> edge_queue;

};

int read_file(const char* _filename, ConvexHull& _convex_hull)
{
	std::ifstream input_file;
	input_file.open(_filename);
	if (input_file.fail())
	{
		std::cout << "failed to open file: " << _filename << std::endl;
		return -3;
	}

	std::string line;
	while (std::getline(input_file, line))
	{
		float x, y;
		std::istringstream line_stream(line.substr(2));
		line_stream >> x >> y;
		_convex_hull.add_vert(x, y);
	}

	input_file.close();

	return 0;
}


const char* filename_in = "gen_verts_file.obj";
const char* filename_out = "result.obj";


int main(int argc, char** argv)
{
  	ConvexHull convex_hull;

	for (int i = 1; i < argc - 2; i += 2)
	{
		if (argv[i][0] != '-')
		{
			std::cout << "incorrect syntax: " << argv[i] << std::endl;
			return -1;
		}
		switch (argv[i][1])
		{
		case 'i':
		{
			// TODO
		}
		break;
		default:
		{
			std::cout << "unknown option: -" << argv[i][1] << std::endl;
			return -2;
		}
		}
	}

	//int res = read_file(filename_in, convex_hull);
	int res = read_file(argv[argc - 1], convex_hull);
	if (res < 0) { return res; }

	convex_hull.find_solution();
	//convex_hull.get_result(filename_in, filename_out);
	convex_hull.get_result(argv[argc - 1], argv[argc - 2]);

  	return 0;
}
