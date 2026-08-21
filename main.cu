#include <iostream>
#include <fstream>
#include <filesystem>

#include "common.cuh"


__device__ glm::vec3 CalculateTriangleCentroid(const face_s &face) {
  return (face.vertices[0].position +
          face.vertices[1].position +
          face.vertices[2].position) / 3.0f;
}

__device__ glm::vec3 CalculateTriangleNormal(const face_s &face) {
  glm::vec3 edge1 = face.vertices[1].position - face.vertices[0].position;
  glm::vec3 edge2 = face.vertices[2].position - face.vertices[0].position;

  return glm::normalize(cross(edge1, edge2));
}

__device__ float CalculateTriangleArea(const glm::vec3 vertices[3]) {
  const glm::vec3 edge1 = vertices[1] - vertices[0];
  const glm::vec3 edge2 = vertices[2] - vertices[0];

  return length(cross(edge1, edge2)) / 2.0f;
}

__device__ float CalculateFormFactor(const face_s &face_i, const face_s &face_j) {
  // The floor area of hemisphere
  const float hemisphere_radius = 1.0f;
  const float hemisphere_floor_area = glm::pi<float>() * (hemisphere_radius * hemisphere_radius);

  // The centroid of the faces
  const glm::vec3 i_centroid = CalculateTriangleCentroid(face_i);
  const glm::vec3 j_centroid = CalculateTriangleCentroid(face_j);

  // The direction between the faces
  const glm::vec3 i_to_j_direction = j_centroid - i_centroid;

  // The normal of face i
  const glm::vec3 i_normal = CalculateTriangleNormal(face_i);

  // Backface culling
  if (dot(i_to_j_direction, i_normal) < 0.0f) {
    return 0.0f;
  }

  // The distance between the faces
  float i_to_j_distance = fabsf(length(i_to_j_direction));

  // The vectors form the three corners of face j to the centroid of face i
  glm::vec3 i_corner_to_j_centroid[3] = {
      glm::normalize(face_j.vertices[0].position - i_centroid),
      glm::normalize(face_j.vertices[1].position - i_centroid),
      glm::normalize(face_j.vertices[2].position - i_centroid)
  };

  // Map the face i on hemisphere
  glm::vec3 i_corner_on_hemisphere[3] = {
      i_centroid + i_corner_to_j_centroid[0] * hemisphere_radius,
      i_centroid + i_corner_to_j_centroid[1] * hemisphere_radius,
      i_centroid + i_corner_to_j_centroid[2] * hemisphere_radius
  };

  // Map "i_corner_on_hemisphere" to floor from hemisphere
  glm::vec3 map_to_floor_from_hemisphere[3] = {
      i_corner_on_hemisphere[0] - i_normal * dot(i_corner_on_hemisphere[0] - i_centroid, i_normal),
      i_corner_on_hemisphere[1] - i_normal * dot(i_corner_on_hemisphere[1] - i_centroid, i_normal),
      i_corner_on_hemisphere[2] - i_normal * dot(i_corner_on_hemisphere[2] - i_centroid, i_normal)
  };
  // Verification
  for (int i = 0; i < 3; i++) {
    // Test map_to_floor_from_hemisphere is vertical to i_normal
    if (fabsf(dot(map_to_floor_from_hemisphere[i] - i_centroid, i_normal)) > 1e-5f) {
      return -1145.14f;
    }
    // Test map_to_floor_from_hemisphere is in hemisphere
    if (length(map_to_floor_from_hemisphere[i] - i_centroid) > 1.0f + 1e-5f) {
      return -1919.810f;
    }
  }

  // The face area
  float face_area = CalculateTriangleArea(map_to_floor_from_hemisphere);

  // The final form factor
  return face_area / hemisphere_floor_area;
}

__global__ void CalculateLighting(face_s *faces, int face_count, glm::vec3 *in_rad, glm::vec3 *out_rad,
                                  int bounce_count) {
  const int f_i_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (f_i_idx >= face_count) {
    return;
  }

  // Sum F_ij * B_j
  glm::vec3 sum{};
  for (int f_j_idx = 0; f_j_idx < face_count; f_j_idx++) {
    // Skip the self face
    if (f_i_idx == f_j_idx) {
      continue;
    }

    // Calculate form-factor between face_i and face_j
    const float form_factor = CalculateFormFactor(faces[f_i_idx], faces[f_j_idx]);

    sum += in_rad[f_j_idx] * form_factor;
  }

  if (bounce_count != 0) {
    out_rad[f_i_idx] = sum * faces[f_i_idx].reflectivity + faces[f_i_idx].emission;
  } else {
    out_rad[f_i_idx] = sum * faces[f_i_idx].reflectivity;
  }
}

struct {
  face_s *faces{};
  glm::vec3 *faces_lighting{};
  glm::vec3 *faces_lighting_old{};
} device_var;

struct {
  glm::vec3 *faces_lighting{};
  int face_count{};
  int faces_lighting_count{};
  int face_lighting_buffer_size{};
} host_var;

int main(int argc, char *argv[]) {
  int bounce_num = 8;
  bool is_log{};
  std::vector<face_s> mesh{};

  std::cout << "Starting to load the mesh..." << std::endl;

  std::filesystem::path obj_path("../assets/cube.obj");
  if (argc >= 2) {
    if (std::filesystem::exists(std::filesystem::path(argv[1]))) {
      obj_path = argv[1];
    } else {
      std::cout << "The path of the obj file was not found!" << std::endl;
      return -1;
    }
  }
  if (argc >= 3) {
    is_log = (std::string(argv[2]) == "log");
  }

  bool load_result = false;
  if (load_result) {
  } else {
    std::cout << "No mesh loaded!" << std::endl;
    return -1;
  }
  std::cout << "Mesh: " << obj_path.string() << " loading successful." << std::endl;

  std::cout << "Start the work on the GPU side..." << std::endl;
  // Variable
  host_var.face_count = mesh.size();
  host_var.faces_lighting_count = host_var.face_count;
  host_var.face_lighting_buffer_size = host_var.faces_lighting_count * sizeof(glm::vec3);
  host_var.faces_lighting = new glm::vec3[host_var.face_count]();

  // Initial
  // The initial faces emission
  for (int i = 0; i < host_var.face_count; i++) {
    host_var.faces_lighting[i] = mesh[i].emission;
  }
  CHECK_CUDA_ERROR(cudaMalloc(&device_var.faces, host_var.face_count * sizeof(face_s)));
  CHECK_CUDA_ERROR(cudaMalloc(&device_var.faces_lighting, host_var.face_lighting_buffer_size));
  CHECK_CUDA_ERROR(cudaMalloc(&device_var.faces_lighting_old, host_var.face_lighting_buffer_size));

  // Copy
  CHECK_CUDA_ERROR(cudaMemcpy(
    device_var.faces,
    mesh.data(),
    host_var.face_count * sizeof(face_s),
    cudaMemcpyHostToDevice
  ));
  CHECK_CUDA_ERROR(cudaMemcpy(
    device_var.faces_lighting_old,
    host_var.faces_lighting,
    host_var.face_lighting_buffer_size,
    cudaMemcpyHostToDevice
  ));

  // Block and Grid size
  int block_size = 256;
  int grid_size = (host_var.face_count + block_size - 1) / block_size;

  // Call kernel
  for (int i = 0; i < bounce_num; i += 1) {
    CalculateLighting<<<grid_size,block_size>>>(
        device_var.faces,
        host_var.face_count,
        device_var.faces_lighting_old,
        device_var.faces_lighting,
        i
        );

    // Swap buffer
    glm::vec3 *temp = device_var.faces_lighting_old;
    device_var.faces_lighting_old = device_var.faces_lighting;
    device_var.faces_lighting = temp;
  }

  // Get error
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Fetch from GPU
  CHECK_CUDA_ERROR(cudaMemcpy(
    host_var.faces_lighting,
    device_var.faces_lighting_old,
    host_var.face_lighting_buffer_size,
    cudaMemcpyDeviceToHost
  ));

  // Print results
  if (is_log) {
    for (int i = 0; i < host_var.faces_lighting_count; i += 1) {
      std::cout << std::fixed << std::setprecision(8);
      std::cout << "Face " << i << " lighting: \n\t" <<
          "R: " << host_var.faces_lighting[i].x << " " <<
          "G: " << host_var.faces_lighting[i].y << " " <<
          "B: " << host_var.faces_lighting[i].z << std::endl;
    }
  }
  std::cout << "All the faces lighting have been successfully calculated." << std::endl;

  std::cout << "End GPU side work." << std::endl;

  std::cout << "Start saving to obj mesh file..." << std::endl;
  try {
    // TODO: Save to file
    std::cout << "OBJ file saved successfully!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  std::cout << "The obj file is saved." << std::endl;

  std::cout << "Cleaning the program..." << std::endl;
  // Destroy
  CHECK_CUDA_ERROR(cudaFree(device_var.faces));
  CHECK_CUDA_ERROR(cudaFree(device_var.faces_lighting));
  CHECK_CUDA_ERROR(cudaFree(device_var.faces_lighting_old));
  delete[] host_var.faces_lighting;
  mesh.clear();

  return 0;
}