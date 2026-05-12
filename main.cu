// #define OBJL_CONSOLE_OUTPUT

#include <iostream>
#include <fstream>
#include <filesystem>

#include "OBJ_Loader.h"

#include "math.cu"

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (line: %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    }

struct vertex_s {
    float3 position{};
    float3 normal{};
    float2 uv{};
};

struct face_s {
    vertex_s vertices[3]{};
};


struct {
    face_s *faces{};
    float *form_factors_area{};
} device_var;

struct {
    float *form_factors_area{};
    int face_count{};
    int form_factor_area_count{};
} host_var;


__device__ float3 CalculateTriangleCentroid(face_s &face) {
    return (face.vertices[0].position +
            face.vertices[1].position +
            face.vertices[2].position) / 3.0f;
}

__device__ float CalculateFormFactor(face_s &face_i, face_s &face_j) {
    // The centroid of the faces
    float3 f_i_centroid = CalculateTriangleCentroid(face_i);
    float3 f_j_centroid = CalculateTriangleCentroid(face_j);

    // The distance between the faces
    float f_ij_distance = fabsf(length(f_i_centroid - f_j_centroid));

    return f_ij_distance;
}

__global__ void Calculate(
    face_s *faces,
    int face_count,
    float *form_factors_area
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= face_count * face_count) {
        return;
    }

    // The face "i" index
    const int f_i_idx = idx / face_count;

    // The face "j" index
    for (int f_j_idx = 0; f_j_idx < face_count; f_j_idx++) {
        // Skip the current self face
        if (f_i_idx == f_j_idx) {
            continue;
        }

        // To calculate form-factor between the tow face
        form_factors_area[idx] = CalculateFormFactor(faces[f_i_idx], faces[f_j_idx]);
    }
}

int main(int argc, char *argv[]) {
    std::vector<face_s> mesh{};

    std::cout << "Starting to load the mesh..." << std::endl;
    objl::Loader obj_loader;

    std::filesystem::path obj_path("../assets/cube.obj");
    if (argc > 1) {
        if (std::filesystem::exists(std::filesystem::path(argv[1]))) {
            obj_path = argv[1];
        } else {
            std::cout << "The path of the obj file was not found!" << std::endl;
            return -1;
        }
    }
    bool load_result = obj_loader.LoadFile(obj_path.string());
    if (load_result) {
        for (int i = 0; i < obj_loader.LoadedMeshes.size(); i++) {
            objl::Mesh curMesh = obj_loader.LoadedMeshes[i];

            // std::cout << "Mesh" << ": " << curMesh.MeshName << "\n";

            for (int j = 0; j < curMesh.Indices.size(); j += 3) {
                unsigned int cur_face_idx = curMesh.Indices[j];

                objl::Vertex v1 = curMesh.Vertices[cur_face_idx];
                objl::Vertex v2 = curMesh.Vertices[cur_face_idx + 1];
                objl::Vertex v3 = curMesh.Vertices[cur_face_idx + 2];

                face_s cur_face{};

                // Position
                cur_face.vertices[0].position = float3(v1.Position.X, v1.Position.Y, v1.Position.Z);
                cur_face.vertices[1].position = float3(v2.Position.X, v2.Position.Y, v2.Position.Z);
                cur_face.vertices[2].position = float3(v3.Position.X, v3.Position.Y, v3.Position.Z);
                // Normal
                cur_face.vertices[0].normal = float3(v1.Normal.X, v1.Normal.Y, v1.Normal.Z);
                cur_face.vertices[1].normal = float3(v2.Normal.X, v2.Normal.Y, v2.Normal.Z);
                cur_face.vertices[2].normal = float3(v3.Normal.X, v3.Normal.Y, v3.Normal.Z);
                // UV
                cur_face.vertices[0].uv = float2(v1.TextureCoordinate.X, v1.TextureCoordinate.Y);
                cur_face.vertices[1].uv = float2(v2.TextureCoordinate.X, v2.TextureCoordinate.Y);
                cur_face.vertices[2].uv = float2(v3.TextureCoordinate.X, v3.TextureCoordinate.Y);

                mesh.push_back(cur_face);
            }
        }
    } else {
        std::cout << "No mesh loaded!" << std::endl;
        return -1;
    }
    std::cout << "Mesh: " << obj_path.string() << " loading successful." << std::endl;

    std::cout << "Start the work on the GPU side..." << std::endl;
    // Variable
    host_var.face_count = mesh.size();
    host_var.form_factor_area_count = host_var.face_count * host_var.face_count;
    host_var.form_factors_area = new float[host_var.form_factor_area_count]();

    // Initial
    CHECK_CUDA_ERROR(cudaMalloc(&device_var.faces, host_var.face_count * sizeof(face_s)));
    CHECK_CUDA_ERROR(cudaMalloc(&device_var.form_factors_area, host_var.form_factor_area_count * sizeof(float)));

    // Copy
    CHECK_CUDA_ERROR(cudaMemcpy(
        device_var.faces,
        mesh.data(),
        host_var.face_count * sizeof(face_s),
        cudaMemcpyHostToDevice
    ));

    // Block and Grid size
    int block_size = 256;
    int grid_size = (host_var.form_factor_area_count + block_size - 1) / block_size;

    // Call kernel
    Calculate<<<grid_size,block_size>>>(
        device_var.faces,
        host_var.face_count,
        device_var.form_factors_area
    );

    // Get error
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Fetch from GPU
    CHECK_CUDA_ERROR(cudaMemcpy(
        host_var.form_factors_area,
        device_var.form_factors_area,
        host_var.form_factor_area_count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // Print
    for (int i = 0; i < host_var.form_factor_area_count; i += host_var.face_count) {
        std::cout << "Face first index = " << i << "\tfirst value = " << host_var.form_factors_area[i] << std::endl;
    }
    std::cout << "All the form-factors have been successfully calculated." << std::endl;

    std::cout << "End GPU side work." << std::endl;

    std::cout << "Cleaning the program..." << std::endl;
    // Destroy
    CHECK_CUDA_ERROR(cudaFree(device_var.faces));
    CHECK_CUDA_ERROR(cudaFree(device_var.form_factors_area));
    delete[] host_var.form_factors_area;
    mesh.clear();

    return 0;
}
