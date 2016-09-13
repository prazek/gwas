#include "cuda_utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <iomanip>
#include <iostream>
#include <ctime>

__device__ inline float H(float arg) {
  return -(arg) * log2f(arg);
}

__device__ inline float H2(int arg1, int arg2, float p) {
  return  H((arg1 + p) / (arg1 + arg2 + 1.0f)) +
      H((arg2 + 1.0f - p) / (arg1 + arg2 + 1.0f));
}
__device__ float compute_gig_1_2(const uint32_t* const __restrict__ v1_1,
                                 const uint32_t* const __restrict__ v1_2,
                                 const uint32_t* const __restrict__ v2_1,
                                 const uint32_t* const __restrict__ v2_2,
                                 const uint32_t* const __restrict__ ds_1,
                                 int num_objects,
                                 float p) {
  // how many objects having [d][x][y], where d [0, 1] is descriptive value,
  // x [0, 2] is v1 value and y [0, 2] v2 value
  int count[2][3][3] = {0};

#pragma unroll
  for (int obj = 0; obj < (num_objects / 32); obj++) {
#pragma unroll
    for (int d = 0; d <= 1; d++) {
#pragma unroll
      for (int v1 = 0; v1 <= 2; v1++) {
#pragma unroll
        for (int v2 = 0; v2 <= 2; v2++) {
          // No need to put __ldg  here.
          uint32_t d_b = (d == 1) ? ds_1[obj] : ~ds_1[obj];
          uint32_t v1_b = (v1 == 0) ? (~v1_1[obj] & ~v1_2[obj])
                                    : (v1 == 1 ? v1_1[obj] : v1_2[obj]);
          uint32_t v2_b = (v2 == 0) ? (~v2_1[obj] & ~v2_2[obj])
                                    : (v2 == 1 ? v2_1[obj] : v2_2[obj]);
          count[d][v1][v2] += __popc(d_b & v1_b & v2_b);
        }
      }
    }
  }

  // Do the same thing, but only for the rest of the bits.
  if (num_objects % 32) {
    uint32_t rest = (~0u) >> (32 - num_objects % 32);
    int last_obj_ind = num_objects / 32;
#pragma unroll
    for (int d = 0; d <= 1; d++) {
#pragma unroll
      for (int v1 = 0; v1 <= 2; v1++) {
#pragma unroll
        for (int v2 = 0; v2 <= 2; v2++) {
          uint32_t d_b = (d == 1) ? ds_1[last_obj_ind] : ~ds_1[last_obj_ind];
          uint32_t v1_b = (v1 == 0) ? (~v1_1[last_obj_ind] & ~v1_2[last_obj_ind])
                                    : (v1 == 1 ? v1_1[last_obj_ind] : v1_2[last_obj_ind]);
          uint32_t v2_b = (v2 == 0) ? (~v2_1[last_obj_ind] & ~v2_2[last_obj_ind])
                                    : (v2 == 1 ? v2_1[last_obj_ind] : v2_2[last_obj_ind]);
          count[d][v1][v2] += __popc(d_b & v1_b & v2_b & rest);
        }
      }
    }
  }

  int sum_n2_n3 = 0;
  int sum_n2_n3_2 = 0;
#pragma unroll
  for (int i = 0; i <= 2; i++) {
#pragma unroll
    for (int v2 = 0; v2 <= 2; v2++) {
      sum_n2_n3 += count[0][i][v2];
      sum_n2_n3_2 += count[1][i][v2];
    }
  }

  const float h_p = H2(sum_n2_n3, sum_n2_n3_2, p);
  float ig1 = h_p, ig2 = h_p;
#pragma unroll
  for (int v = 0; v <= 2; v++) {
    int sum3_1 = 0, sum3_2 = 0, sum2_1 = 0, sum2_2 = 0;
#pragma unroll
    for (int v2 = 0; v2 <= 2; v2++) {
      sum3_1 += count[0][v][v2];
      sum3_2 += count[1][v][v2];
      sum2_1 += count[0][v2][v];
      sum2_2 += count[1][v2][v];
    }
    ig1 -= (sum3_1 + sum3_2) * H2(sum3_1, sum3_2, p);
    ig2 -= (sum2_1 + sum2_2) * H2(sum2_1, sum2_2, p);
  }

  float ig12 = h_p;
#pragma unroll
  for (int v1 = 0; v1 <= 2; v1++) {
#pragma unroll
    for (int v2 = 0; v2 <= 2; v2++) {
      ig12 -= (count[0][v1][v2] + count[1][v1][v2])
          * H2(count[0][v1][v2], count[1][v1][v2], p);
    }
  }

  auto result = ig12 - ((ig1 > ig2) ? ig1 : ig2);
  return result;
}

struct GigStruct {
  float gig;
  int v1, v2;

  bool operator<(GigStruct other) const { return gig > other.gig; };

  friend std::ostream& operator<<(std::ostream& os, const GigStruct& gig) {
    return os << std::fixed << std::setprecision(3) << gig.gig << " " << gig.v1
              << " " << gig.v2;
  }
};
#define OFFSET(var_number, objects_count) \
  ((var_number) * (32 - ((objects_count) % 32)))

__global__ void compute_gig_kernel(const uint32_t* const __restrict__ vars_1,
                                   const uint32_t* const __restrict__ vars_2,
                                   const uint32_t* const __restrict__ ds_1,
                                   int num_objects,
                                   int num_vars,
                                   GigStruct* const __restrict__ r_gig,
                                   float threshold,
                                   float p,
                                   int* __restrict__ atomic_counter) {

  const int v1_p = blockDim.x * blockIdx.x + threadIdx.x;
  const int v2_p = blockDim.y * blockIdx.y + threadIdx.y;

  if (v1_p >= v2_p || v1_p >= num_vars || v2_p >= num_vars)
    return;

  const auto ind1 = (v1_p * num_objects + OFFSET(v1_p, num_objects)) / 32;
  const auto ind2 = (v2_p * num_objects + OFFSET(v2_p, num_objects)) / 32;
  assert((v1_p * num_objects + OFFSET(v1_p, num_objects)) % 32 == 0);
  assert((v2_p * num_objects + OFFSET(v2_p, num_objects)) % 32 == 0);

  const auto result =
      compute_gig_1_2(&vars_1[ind1], &vars_2[ind1], &vars_1[ind2],
                      &vars_2[ind2], ds_1, num_objects, p);

  if (!(result > threshold))
    return;  // not large enough result.

  const int index = atomicAdd(atomic_counter, 1);
  r_gig[index] = {result, v1_p, v2_p};
}

int getCounterValue(int* atomic_counter_d) {
  int result = -1;
  checkCudaErrors(cudaMemcpy(&result, atomic_counter_d, sizeof(int),
                             cudaMemcpyDeviceToHost));
  return result;
}

int* getDeviceCounter() {
  int* atomic_counter_d;
  int init = 0;
  checkCudaErrors(cudaMalloc((void**)&atomic_counter_d, sizeof(int)));
  checkCudaErrors(
      cudaMemcpy(atomic_counter_d, &init, sizeof(int), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaDeviceSynchronize());
  return atomic_counter_d;
}

template <typename Allocator>
void setBit(std::vector<uint32_t, Allocator>& bitmask, int index) {
  bitmask.at(index / 32) |= (1u << (index % 32));
}

int getBitmaskSize(int objects) {
  return (objects + 31) / 32;
}

struct KernelLaunchInfo {
  int num_vars;
  int num_objects;
  float threshold;
  float a_priori;

  KernelLaunchInfo(int num_vars,
                   int num_objects,
                   float threshold,
                   float a_priori)
      : num_vars(num_vars),
        num_objects(num_objects),
        threshold(threshold),
        a_priori(a_priori) {}
};

using pinnedBitmaskVector = std::vector<uint32_t, pinned_allocator<uint32_t>>;
std::vector<GigStruct, pinned_allocator<GigStruct> > launch_kernel(
    pinnedBitmaskVector ds,
                                     pinnedBitmaskVector vars_1,
                                     pinnedBitmaskVector vars_2,
                                     KernelLaunchInfo info,
                                     int result_size) {
  const auto vars1_d = make_device_vector(vars_1),
                       vars2_d = make_device_vector(vars_2);
  const auto ds1_d = make_device_vector(ds);
  auto gigs_d = make_device_vector<GigStruct>(result_size);

  auto* atomic_counter_d = getDeviceCounter();
  const int BlockSize = 16; // It is possible that it will be faster with 32.
  auto gridSize = (info.num_vars + BlockSize - 1) / BlockSize;
  dim3 grid(gridSize, gridSize);
  dim3 block(BlockSize, BlockSize);

  compute_gig_kernel<<<grid, block>>>(vars1_d.dev(), vars2_d.dev(), ds1_d.dev(), info.num_objects,
                                      info.num_vars, gigs_d.dev(), info.threshold,
                                      info.a_priori, atomic_counter_d);
  checkCudaErrors(cudaDeviceSynchronize());

  const int computed_result_size = getCounterValue(atomic_counter_d);
  if (computed_result_size > result_size)
    printf("%d %d ub happend\n", computed_result_size, result_size);
  auto gigs = make_vector<GigStruct, pinned_allocator<GigStruct>>
      (gigs_d, std::min(result_size, computed_result_size));
  std::sort(gigs.begin(), gigs.end());
  return gigs;
}

float calculateThreshold(const pinnedBitmaskVector& ds,
                         pinnedBitmaskVector vars1,
                         pinnedBitmaskVector vars2,
                         int num_vars,
                         int num_objects,
                         int normal_result_size,
                         float experimentSize,
                         float a_priori) {

  KernelLaunchInfo info(num_vars, num_objects,
                        -std::numeric_limits<float>::infinity(), a_priori);
  int current_result_size = (num_vars * (num_vars - 1)) / 2;  // want to get gig for every
  // unique pair in smaller vars.
  auto gigs = launch_kernel(ds, std::move(vars1), std::move(vars2),
                            info, current_result_size);
  const float cutoffPercent = experimentSize * experimentSize;
  int cutoffLevel = normal_result_size * cutoffPercent;
  return gigs[cutoffLevel].gig;
}

std::vector<int> getIndexes(int ofSize, int outOfSize) {
  std::vector<int> indexes(outOfSize);
  std::iota(indexes.begin(), indexes.end(), 0);
  std::random_shuffle(indexes.begin(), indexes.end());
  indexes.resize(ofSize);

  std::sort(indexes.begin(), indexes.end());
  return indexes;
}

void init() {
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  std::srand(time(nullptr));
}

// Function for fast reading int.
uint32_t getInt() {
  char c;
  while(true) {
    c = getchar_unlocked();
    if ('0' <= c && c <= '9')
      break; // got digit
  }
  uint32_t result = c - '0';

  while (true) {
    c = getchar_unlocked();
    if (!('0' <= c && c <= '9'))
      break; // finished reading digits
    result *= 10;
    result += c - '0';
  }
  return result;
}

int un;
int main() {
  init();
  std::ios_base::sync_with_stdio(false);
  int num_objects, num_vars, result_size;
  float a_priori;
  const float sampleSizeMultiplier = 0.10;  // 10%
  un = scanf("%d %d %d %f", &num_objects, &num_vars, &result_size, &a_priori);

  const int LastBitmaskElement =
      num_vars * num_objects + OFFSET(num_vars - 1, num_objects);
  const auto varsBitmaskSize = getBitmaskSize(LastBitmaskElement);
  std::vector<uint32_t, pinned_allocator<uint32_t>> vars_1(varsBitmaskSize);
  std::vector<uint32_t, pinned_allocator<uint32_t>> vars_2(varsBitmaskSize);

  const auto DescriptiveMaskSize = getBitmaskSize(num_objects);
  pinnedBitmaskVector ds(DescriptiveMaskSize);

  const int smallVarsRow = num_vars * sampleSizeMultiplier;
  const int sampleVarsSize = smallVarsRow * num_objects;

  std::vector<int8_t> small_vars(sampleVarsSize, -1);
  auto indexes = getIndexes(smallVarsRow, num_vars);

  // Firstly read num_object descriptive values.
  for (int i = 0; i < num_objects; ++i) {
    int d;
    d = getInt();

    if (d == 1)
      setBit(ds, i);
    assert(d == 1 || d == 0);
  }

  int smallIndex = 0;
  // Then read num_vars lines of num_objects vars.
  for (int j = 0; j < num_vars; ++j) {
    bool useForSampling = false;
    if (smallIndex < indexes.size() && indexes[smallIndex] == j) {
      useForSampling = true;
    }

    for (int i = 0; i < num_objects; ++i) {
      int var;
      var = getInt();
      const int offset = OFFSET(j, num_objects);
      if (var == 1)
        setBit(vars_1, j * num_objects + i + offset);
      else if (var == 2)
        setBit(vars_2, j * num_objects + i + offset);
      else
        assert(var == 0);

      if (useForSampling) {
        const int newIndex = smallIndex * num_objects + i;
        assert(small_vars.at(newIndex) == -1);
        small_vars.at(newIndex) = var;
      }
    }

    if (useForSampling)
      smallIndex++;
    assert(smallIndex == smallVarsRow && smallIndex == indexes.size()
               && "not all small_vars filled");
  }


  for (int i = 0; i < indexes.size(); i++) {
    // random shuffle values of each type.
    std::random_shuffle(small_vars.begin() + i * num_objects,
                        small_vars.begin() + (i + 1) * num_objects);
  }
  const int lastSmallBitmaskElement =
      sampleVarsSize * num_objects + OFFSET(sampleVarsSize - 1, num_objects);
  auto smallVarsBitmaskSize = getBitmaskSize(lastSmallBitmaskElement);
  pinnedBitmaskVector small_vars_1(smallVarsBitmaskSize),
      small_vars_2(smallVarsBitmaskSize);

  int varNum = 0;
  for (int i = 0; i < small_vars.size(); i++) {
    if (i % num_objects == 0)
      varNum++;

    int var = small_vars.at(i);
    if (var == 1)
      setBit(small_vars_1, i + OFFSET(varNum, num_objects));
    else if (var == 2)
      setBit(small_vars_2, i + OFFSET(varNum, num_objects));
    else
      assert(var == 0);
  }
  const float threshold = calculateThreshold(
      ds, std::move(small_vars_1), std::move(small_vars_2), smallVarsRow,
      num_objects, result_size, sampleSizeMultiplier, a_priori);
  KernelLaunchInfo info(num_vars, num_objects, threshold, a_priori);

  // We should not get more than 2.25x results greater than threshold.
  const float EXTRA_ELEMENTS_MULTIPLIER = 2.5;
  auto gigs = launch_kernel(std::move(ds), std::move(vars_1), std::move(vars_2),
                            info, result_size * EXTRA_ELEMENTS_MULTIPLIER);

  if (gigs.size() > result_size)
    gigs.resize(result_size);
  for (int i = 0; i < gigs.size(); ++i)
    std::cout << gigs[i] << '\n';

  return 0;
}
