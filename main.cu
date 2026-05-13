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
    float reflectivity{};
    float3 radiosity{};
    float3 emission{};
};


struct {
    face_s *faces{};
    float3 *faces_lighting{};
} device_var;

struct {
    float3 *faces_lighting{};
    int face_count{};
    int faces_lighting_count{};
    int face_lighting_buffer_size{};
} host_var;


__device__ float3 CalculateTriangleCentroid(const face_s &face) {
    return (face.vertices[0].position +
            face.vertices[1].position +
            face.vertices[2].position) / 3.0f;
}

__device__ float3 CalculateTriangleNormal(const face_s &face) {
    float3 edge1 = face.vertices[1].position - face.vertices[0].position;
    float3 edge2 = face.vertices[2].position - face.vertices[0].position;

    return normalized(cross(edge1, edge2));
}

__device__ float CalculateTriangleArea(const float3 vertices[3]) {
    const float3 edge1 = vertices[1] - vertices[0];
    const float3 edge2 = vertices[2] - vertices[0];

    return length(cross(edge1, edge2)) / 2.0f;
}

__device__ float CalculateFormFactor(const face_s &face_i, const face_s &face_j) {
    // The floor area of hemisphere
    const float hemisphere_radius = 1.0f;
    const float hemisphere_floor_area = CUDART_PI_F * (hemisphere_radius * hemisphere_radius);

    // The centroid of the faces
    const float3 i_centroid = CalculateTriangleCentroid(face_i);
    const float3 j_centroid = CalculateTriangleCentroid(face_j);

    // The direction between the faces
    const float3 i_to_j_direction = j_centroid - i_centroid;

    // The normal of face i
    const float3 i_normal = CalculateTriangleNormal(face_i);

    // Backface culling
    if (dot(i_to_j_direction, i_normal) < 0.0f) {
        return 0.0f;
    }

    // The distance between the faces
    float i_to_j_distance = fabsf(length(i_to_j_direction));

    // The vectors form the three corners of face j to the centroid of face i
    float3 i_corner_to_j_centroid[3] = {
        normalized(face_j.vertices[0].position - i_centroid),
        normalized(face_j.vertices[1].position - i_centroid),
        normalized(face_j.vertices[2].position - i_centroid)
    };

    // Map the face i on hemisphere
    float3 i_corner_on_hemisphere[3] = {
        i_centroid + i_corner_to_j_centroid[0] * hemisphere_radius,
        i_centroid + i_corner_to_j_centroid[1] * hemisphere_radius,
        i_centroid + i_corner_to_j_centroid[2] * hemisphere_radius
    };

    // Map "i_corner_on_hemisphere" to floor from hemisphere
    float3 map_to_floor_from_hemisphere[3] = {
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

__device__ float3 CalculateLighting(face_s *faces, int face_count, int f_i_idx) {
    // Sum F_ij * B_j
    float3 sum{};
    for (int f_j_idx = 0; f_j_idx < face_count; f_j_idx++) {
        // Skip the self face
        if (f_i_idx == f_j_idx) {
            continue;
        }

        // Calculate form-factor between face_i and face_j
        const float form_factor = CalculateFormFactor(faces[f_i_idx], faces[f_j_idx]);

        sum += faces[f_j_idx].radiosity * form_factor;
    }

    float3 accumulated_lighting{};
    // Accumulate lighting
    accumulated_lighting += sum * faces[f_i_idx].reflectivity + faces[f_i_idx].emission;

    return accumulated_lighting;
}

__global__ void Calculate(
    face_s *faces,
    int face_count,
    float3 *faces_lighting
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= face_count) {
        return;
    }

    float3 lighting = CalculateLighting(faces, face_count, idx);

    faces_lighting[idx] = lighting;
}

int main(int argc, char *argv[]) {
    bool is_log{};
    std::vector<face_s> mesh{};

    std::cout << "Starting to load the mesh..." << std::endl;
    objl::Loader obj_loader;

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
    bool load_result = obj_loader.LoadFile(obj_path.string());
    if (load_result) {
        for (int i = 0; i < obj_loader.LoadedMeshes.size(); i++) {
            objl::Mesh cur_mesh = obj_loader.LoadedMeshes[i];

            std::cout << "Mesh" << ": " << cur_mesh.MeshName << "\n";

            for (int j = 0; j < cur_mesh.Indices.size(); j += 3) {
                unsigned int i1 = cur_mesh.Indices[j];
                unsigned int i2 = cur_mesh.Indices[j + 1];
                unsigned int i3 = cur_mesh.Indices[j + 2];

                objl::Vertex v1 = cur_mesh.Vertices[i1];
                objl::Vertex v2 = cur_mesh.Vertices[i2];
                objl::Vertex v3 = cur_mesh.Vertices[i3];

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

                if (!cur_mesh.MeshName.compare("LIGHT_MESH")) {
                    cur_face.reflectivity = 0.5f;
                    cur_face.emission = {1.0f, 0.8f, 0.2f};
                } else {
                    cur_face.reflectivity = 0.2f;
                    cur_face.emission = {0.0, 0.0, 0.0};
                }


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
    host_var.faces_lighting_count = host_var.face_count;
    host_var.face_lighting_buffer_size = host_var.faces_lighting_count * sizeof(float3);
    host_var.faces_lighting = new float3[host_var.face_count]();

    // Initial
    CHECK_CUDA_ERROR(cudaMalloc(&device_var.faces, host_var.face_count * sizeof(face_s)));
    CHECK_CUDA_ERROR(cudaMalloc(&device_var.faces_lighting, host_var.face_lighting_buffer_size));

    // Copy
    CHECK_CUDA_ERROR(cudaMemcpy(
        device_var.faces,
        mesh.data(),
        host_var.face_count * sizeof(face_s),
        cudaMemcpyHostToDevice
    ));

    // Block and Grid size
    int block_size = 256;
    int grid_size = (host_var.face_count + block_size - 1) / block_size;

    // Call kernel
    Calculate<<<grid_size,block_size>>>(
        device_var.faces,
        host_var.face_count,
        device_var.faces_lighting
    );

    // Get error
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Fetch from GPU
    CHECK_CUDA_ERROR(cudaMemcpy(
        host_var.faces_lighting,
        device_var.faces_lighting,
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

    std::cout << "Cleaning the program..." << std::endl;
    // Destroy
    CHECK_CUDA_ERROR(cudaFree(device_var.faces));
    CHECK_CUDA_ERROR(cudaFree(device_var.faces_lighting));
    delete[] host_var.faces_lighting;
    mesh.clear();

    return 0;
}
