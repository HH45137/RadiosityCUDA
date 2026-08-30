#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>

#include <glm/ext.hpp>
#include <glm/glm.hpp>

namespace RadCu {
void CheckCudaError(cudaError err);

struct vertex_s {
  glm::vec3 position{};
  glm::vec3 normal{};
  glm::vec2 uv{};
};

struct face_s {
  vertex_s vertices[3]{};
  glm::vec3 radiosity{};
  glm::vec3 emission{};
  float reflectivity{};
};

__device__ glm::vec3 CalculateTriangleCentroid(const face_s &face);

__device__ glm::vec3 CalculateTriangleNormal(const face_s &face);

__device__ float CalculateTriangleArea(const glm::vec3 vertices[3]);

__device__ float CalculateFormFactor(const face_s &face_i, const face_s &face_j);

__global__ void CalculateLighting(face_s *faces, int face_count, glm::vec3 *in_rad, glm::vec3 *out_rad,
                                  int bounce_count);
} // namespace RadCu