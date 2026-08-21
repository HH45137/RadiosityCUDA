#include <RadiosityCuda.cuh>
#include <filesystem>
#include <fstream>
#include <iostream>

static struct {
  RadCu::face_s *faces{};
  glm::vec3 *faces_lighting{};
  glm::vec3 *faces_lighting_old{};
} device_var;

static struct {
  glm::vec3 *faces_lighting{};
  int face_count{};
  int faces_lighting_count{};
  int face_lighting_buffer_size{};
} host_var;

int main(int argc, char *argv[]) {
  int bounce_num = 8;
  bool is_log{};
  std::vector<RadCu::face_s> mesh{};

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
  RadCu::CheckCudaError(cudaMalloc(&device_var.faces, host_var.face_count * sizeof(RadCu::face_s)));
  RadCu::CheckCudaError(cudaMalloc(&device_var.faces_lighting, host_var.face_lighting_buffer_size));
  RadCu::CheckCudaError(cudaMalloc(&device_var.faces_lighting_old, host_var.face_lighting_buffer_size));

  // Copy
  RadCu::CheckCudaError(
      cudaMemcpy(device_var.faces, mesh.data(), host_var.face_count * sizeof(RadCu::face_s), cudaMemcpyHostToDevice));
  RadCu::CheckCudaError(cudaMemcpy(device_var.faces_lighting_old, host_var.faces_lighting,
                                   host_var.face_lighting_buffer_size, cudaMemcpyHostToDevice));

  // Block and Grid size
  int block_size = 256;
  int grid_size = (host_var.face_count + block_size - 1) / block_size;

  // Call kernel
  for (int i = 0; i < bounce_num; i += 1) {
    CalculateLighting<<<grid_size, block_size>>>(device_var.faces, host_var.face_count, device_var.faces_lighting_old,
                                                 device_var.faces_lighting, i);

    // Swap buffer
    glm::vec3 *temp = device_var.faces_lighting_old;
    device_var.faces_lighting_old = device_var.faces_lighting;
    device_var.faces_lighting = temp;
  }

  // Get error
  RadCu::CheckCudaError(cudaGetLastError());
  RadCu::CheckCudaError(cudaDeviceSynchronize());

  // Fetch from GPU
  RadCu::CheckCudaError(cudaMemcpy(host_var.faces_lighting, device_var.faces_lighting_old,
                                   host_var.face_lighting_buffer_size, cudaMemcpyDeviceToHost));

  // Print results
  if (is_log) {
    for (int i = 0; i < host_var.faces_lighting_count; i += 1) {
      std::cout << std::fixed << std::setprecision(8);
      std::cout << "Face " << i << " lighting: \n\t" << "R: " << host_var.faces_lighting[i].x << " "
                << "G: " << host_var.faces_lighting[i].y << " " << "B: " << host_var.faces_lighting[i].z << std::endl;
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
  RadCu::CheckCudaError(cudaFree(device_var.faces));
  RadCu::CheckCudaError(cudaFree(device_var.faces_lighting));
  RadCu::CheckCudaError(cudaFree(device_var.faces_lighting_old));
  delete[] host_var.faces_lighting;
  mesh.clear();

  return 0;
}
