
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cooperative_groups.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <math.h>
#include <string>
#include <algorithm>
#include <chrono>

#include "glm/glm.hpp"


#define BLK_DIM 1024
//#define BLK_DIM 512
//#define BLK_DIM 256
//#define BLK_DIM 128
//#define BLK_DIM 64
//#define BLK_DIM 32

#define ELEMS_PER_THR 8
//#define ELEMS_PER_THR 6
//#define ELEMS_PER_THR 4
//#define ELEMS_PER_THR 2
//#define ELEMS_PER_THR 1

#define ELEMS_PER_BLK (BLK_DIM * ELEMS_PER_THR)


struct Edge
{
    uint32_t vert1_i;
    uint32_t vert2_i;
};

__global__ void count_distances(
	glm::vec2* verts, uint32_t* vert_idxs, float* distances,
	uint32_t A_vert_i, uint32_t B_vert_i, uint32_t num_vert_idxs

)
{
	__shared__ glm::vec2 A_vert, B_vert;
	if (threadIdx.x == 0)
	{
		A_vert = verts[A_vert_i];
		B_vert = verts[B_vert_i];
	}
	__syncthreads();

	uint32_t blk_idx = ELEMS_PER_BLK * blockIdx.x;
	uint32_t max = fminf(blk_idx + ELEMS_PER_BLK, num_vert_idxs);
	for (uint32_t idx = blk_idx + threadIdx.x; idx < max; idx += BLK_DIM)
	{
		uint32_t vert_i = vert_idxs[idx];
		glm::vec2 T_vert = verts[vert_i];

		glm::vec3 AB_vec(B_vert - A_vert, 0.0f);
		glm::vec3 AT_vec(T_vert - A_vert, 0.0f);

		float A_cross = glm::cross(AB_vec, AT_vec).z;
		if (A_cross < 0)
		{
			distances[idx] = -1.0f;
		}
		else
		{
			glm::vec3 T_proj = (glm::dot(AB_vec, AT_vec) / glm::dot(AB_vec, AB_vec)) * AB_vec;
			distances[idx] = glm::length(AT_vec - T_proj);
		}
	}
}

typedef union {
	struct { float distance; uint32_t raw_idx; } pair;
	unsigned long long int ulong;
} dist_idx_pair;

__device__ unsigned long long int atomicDistMax(unsigned long long int* addr, float distance, uint32_t raw_idx)
{
	dist_idx_pair loc, loctest;
	loc.pair.distance = distance;
	loc.pair.raw_idx = raw_idx;
	loctest.ulong = *addr;
	while (loctest.pair.distance < distance)
	{
		loctest.ulong = atomicCAS(addr, loctest.ulong, loc.ulong);
	}
	return loctest.ulong;
}

__global__ void to_default_state(dist_idx_pair* g_res)
{
	g_res->pair.distance = -1.0f;
}

__global__ void find_max_distance(float* distances, uint32_t num_vert_idxs, dist_idx_pair* g_res)
{
	__shared__ dist_idx_pair shr_res;
	if (threadIdx.x == 0)
	{
		shr_res.pair.distance = -1.0f;
	}
	__syncthreads();

	dist_idx_pair loc_res;
	loc_res.pair.distance = -1.0f;

	uint32_t blk_idx = ELEMS_PER_BLK * blockIdx.x;
	uint32_t max = fminf(blk_idx + ELEMS_PER_BLK, num_vert_idxs);
	for (uint32_t idx = blk_idx + threadIdx.x; idx < max; idx += BLK_DIM)
	{
		float tmp_dist = distances[idx];
		if (tmp_dist > loc_res.pair.distance)
		{
			loc_res.pair.distance = tmp_dist;
			loc_res.pair.raw_idx = idx;
		}
	}

	atomicDistMax(&shr_res.ulong, loc_res.pair.distance, loc_res.pair.raw_idx);
	__syncthreads();

	if (threadIdx.x == 0)
		atomicDistMax(&(g_res->ulong), shr_res.pair.distance, shr_res.pair.raw_idx);
}

__global__ void mark_inner_vets(
	glm::vec2* verts, uint32_t* vert_idxs, float* distances,
	uint32_t A_vert_i, uint32_t B_vert_i, uint32_t C_vert_i,
	uint32_t num_vert_idxs
)
{
	__shared__ glm::vec2 A_vert, B_vert, C_vert;
	if (threadIdx.x == 0)
	{
		A_vert = verts[A_vert_i];
		B_vert = verts[B_vert_i];
		C_vert = verts[C_vert_i];
	}
	__syncthreads();


	uint32_t blk_idx = ELEMS_PER_BLK * blockIdx.x;
	uint32_t max = fminf(blk_idx + ELEMS_PER_BLK, num_vert_idxs);
	for (uint32_t idx = blk_idx + threadIdx.x; idx < max; idx += BLK_DIM)
	{
		if (distances[idx] > 0)
		{
			uint32_t vert_i = vert_idxs[idx];
			glm::vec2 T_vert = verts[vert_i];

			glm::vec3 BC_vec(C_vert - B_vert, 0.0f);
			glm::vec3 CA_vec(A_vert - C_vert, 0.0f);

			glm::vec3 BT_vec(T_vert - B_vert, 0.0f);
			glm::vec3 CT_vec(T_vert - C_vert, 0.0f);

			float B_cross = glm::cross(BC_vec, BT_vec).z;
			float C_cross = glm::cross(CA_vec, CT_vec).z;

			if (!(B_cross >= 0 && C_cross >= 0))
			{
				distances[idx] = -1.0f;
			}
		}
	}
}

class ConvexHull
{
public:

	void add_vert(float _x, float _y)
	{
		verts.push_back({ _x, _y });
	}

	uint32_t get_vert_num() const { return verts.size() / 2; }

	uint32_t find_x_extrema_vert(bool (*comp)(float, float))
	{
		uint32_t idx_ref = 0;
		float x_ref = verts[idx_ref].x;
		uint32_t size = verts.size();
		for (uint32_t i = 1; i < size; i++)
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

	void swap_indeces(uint32_t* vert_idxs, uint32_t idx1, uint32_t idx2)
	{
		uint32_t tmp = vert_idxs[idx1];
		vert_idxs[idx1] = vert_idxs[idx2];
		vert_idxs[idx2] = tmp;
	}

	void find_solution()
	{

		uint32_t min_vert_i = find_x_extrema_vert([](float ref, float next) { return ref > next; });
		uint32_t max_vert_i = find_x_extrema_vert([](float ref, float next) { return ref < next; });

		//std::cout << "min_vert_i: " << min_vert_i << " - max_vert_i: " << max_vert_i << std::endl;

		edge_queue.push_back({ min_vert_i, max_vert_i });
		edge_queue.push_back({ max_vert_i, min_vert_i });

		uint32_t num_verts = verts.size();

		//std::cout << "num_verts: " << num_verts << std::endl;

		glm::vec2* dev_verts;
		uint32_t* host_vert_idxs, * dev_vert_idxs;
		float* host_distances, * dev_distances;
		dist_idx_pair* host_C_dist, *dev_C_dist;

		//std::cout << "buffers init started" << std::endl;

		cudaError_t err1 = cudaMalloc((void**)&dev_verts, num_verts * sizeof(glm::vec2));
		cudaError_t err2 = cudaMalloc((void**)&dev_vert_idxs, num_verts * sizeof(uint32_t));
		cudaError_t err3 = cudaMalloc((void**)&dev_distances, num_verts * sizeof(float));
		cudaError_t err4 = cudaMalloc((void**)&dev_C_dist, sizeof(dist_idx_pair));
		/*
		std::cout << "cudaMalloc finished: " << std::endl;
		std::cout << "\tdev_verts: " << cudaGetErrorName(err1) << std::endl;
		std::cout << "\tdev_vert_idxs: " << cudaGetErrorName(err2) << std::endl;
		std::cout << "\tdev_distances: " << cudaGetErrorName(err3) << std::endl;
		*/

		err2 = cudaHostAlloc((void**)&host_vert_idxs, num_verts * sizeof(uint32_t), cudaHostAllocDefault);
		err3 = cudaHostAlloc((void**)&host_distances, num_verts * sizeof(float), cudaHostAllocDefault);
		err3 = cudaHostAlloc((void**)&host_C_dist, sizeof(dist_idx_pair), cudaHostAllocDefault);
		/*
		std::cout << "cudaHostAlloc finished: " << std::endl;
		std::cout << "\thost_vert_idxs: " << cudaGetErrorName(err2) << std::endl;
		std::cout << "\thost_distances: " << cudaGetErrorName(err3) << std::endl;
		*/

		uint32_t num_vert_idxs = num_verts - 2;
		for (uint32_t i = 0; i < num_verts; i++) { host_vert_idxs[i] = i; }
		swap_indeces(host_vert_idxs, num_verts - 1, min_vert_i);
		swap_indeces(host_vert_idxs, num_verts - 2, max_vert_i);
		/*
		std::cout << "host_vert_idxs finished. result:" << std::endl;
		for (uint32_t i = 0; i < num_verts; i++)
		{
			std::cout << i << ':' << host_vert_idxs[i] << ' ';
		}
		std::cout << std::endl;
		*/

		err1 = cudaMemcpy(dev_verts, verts.data(), num_verts * sizeof(glm::vec2), cudaMemcpyHostToDevice);
		err2 = cudaMemcpy(dev_vert_idxs, host_vert_idxs, num_verts * sizeof(uint32_t), cudaMemcpyHostToDevice);
		/*
		std::cout << "cudaMemcpy finished: " << std::endl;
		std::cout << "\tdev_verts: " << cudaGetErrorName(err1) << std::endl;
		std::cout << "\tdev_vert_idxs: " << cudaGetErrorName(err2) << std::endl;
		*/


		cudaEvent_t start, stop;
		float delta;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		std::cout << "Start" << std::endl;


		for (auto edge_it = edge_queue.begin(); edge_it != edge_queue.end();)
		{
			uint32_t num_blocks = static_cast<uint32_t>(ceil(num_vert_idxs / (float)ELEMS_PER_BLK));
			const uint32_t A_vert_i = edge_it->vert1_i;
			const uint32_t B_vert_i = edge_it->vert2_i;

			//std::cout << std::endl << "algo start" << std::endl;

			// ----- ! GPU begin ! -----
			count_distances <<<num_blocks, BLK_DIM >>> (
				dev_verts, dev_vert_idxs, dev_distances, A_vert_i, B_vert_i, num_vert_idxs
				);
			cudaDeviceSynchronize();

			// ----- ! GPU end ! -----
			/*
			cudaMemcpy(host_distances, dev_distances, num_vert_idxs * sizeof(float), cudaMemcpyDeviceToHost);
			std::cout << "count_distances finished. result:" << std::endl;
			for (uint32_t i = 0; i < num_verts; i++)
			{
				std::cout << host_vert_idxs[i] << ":\t" << host_distances[i] << std::endl;
			}
			std::cout << std::endl;
			*/

			to_default_state <<<1, 1>>>(dev_C_dist);
			cudaDeviceSynchronize();
			find_max_distance<<<num_blocks, BLK_DIM >>>(dev_distances, num_vert_idxs, dev_C_dist);
			cudaDeviceSynchronize();

			/*
			uint32_t raw_C_vert_i = 0;
			float farthest_distance = -1;
			for (uint32_t i = 0; i < num_vert_idxs; i++)
			{
				if (host_distances[i] > farthest_distance)
				{
					farthest_distance = host_distances[i];
					raw_C_vert_i = i;
				}
			}
			*/

			cudaMemcpy(host_C_dist, dev_C_dist, sizeof(dist_idx_pair), cudaMemcpyDeviceToHost);
			if (host_C_dist->pair.distance < 0)
			{
				++edge_it;
				continue;
			}

			const uint32_t C_vert_i = host_vert_idxs[host_C_dist->pair.raw_idx];
			edge_queue.push_back({ A_vert_i, C_vert_i });
			edge_queue.push_back({ C_vert_i, B_vert_i });
			edge_it = edge_queue.erase(edge_it);

			// ----- ! GPU begin ! -----
			mark_inner_vets <<<num_blocks, BLK_DIM >>> (
				dev_verts, dev_vert_idxs, dev_distances, A_vert_i, B_vert_i, C_vert_i, num_vert_idxs
				);
			cudaDeviceSynchronize();
			cudaMemcpy(host_distances, dev_distances, num_vert_idxs * sizeof(float), cudaMemcpyDeviceToHost);
			// ----- ! GPU end ! -----
			/*
			std::cout << "mark_inner_vets finished. result:" << std::endl;
			for (uint32_t i = 0; i < num_verts; i++)
			{
				std::cout << host_vert_idxs[i] << ":\t" << host_distances[i] << std::endl;
			}
			std::cout << std::endl;
			*/

			int right_side = 0;
			int left_side = static_cast<int>(num_vert_idxs) - 1;
			while (left_side >= right_side)
			{
				if (host_distances[left_side] < 0 && host_distances[right_side] >= 0)
				{
					swap_indeces(host_vert_idxs, left_side, right_side);
					--left_side;
					++right_side;
				}
				else
				{
					if (host_distances[left_side] >= 0) { --left_side; }
					if (host_distances[right_side] < 0) { ++right_side; }
				}
			}

			/*
			std::cout << "host_vert_idxs sort finished. result:" << std::endl;
			for (uint32_t i = 0; i < num_verts; i++)
			{
				std::cout << i << ':' << host_vert_idxs[i] << ' ';
			}
			std::cout << std::endl;
			*/

			if (right_side != 0)
			{
				num_vert_idxs = right_side;
				cudaMemcpy(dev_vert_idxs, host_vert_idxs, num_vert_idxs * sizeof(uint32_t), cudaMemcpyHostToDevice);
			}
			else
			{
				break;
			}
		}


		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&delta, start, stop);

		std::cout << "Finish. Time: " << delta << std::endl;


		cudaFreeHost(host_vert_idxs);
		cudaFreeHost(host_distances);
		cudaFreeHost(host_C_dist);

		cudaFree(dev_verts);
		cudaFree(dev_vert_idxs);
		cudaFree(dev_distances);
		cudaFree(dev_C_dist);
	}

	void get_result(const char* filename_in, const char* filename_out)
	{
		std::ifstream src(filename_in);
		std::ofstream dst(filename_out);

		dst << src.rdbuf();

		std::vector<Edge> convex_hull({ edge_queue.begin(), edge_queue.end()});

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
