#pragma once

#include <fstream>
#include <glm/glm.hpp>
#include <glm/ext.hpp>


#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (line: %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    }

struct vertex_s {
  glm::vec3 position{};
  glm::vec3 normal{};
  glm::vec2 uv{};
};

struct face_s {
  vertex_s *vertices{};
  glm::vec3 radiosity{};
  glm::vec3 emission{};
  float reflectivity{};
  int vertex_count{};
};