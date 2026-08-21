#include <RadiosityCuda.cuh>

namespace RadCu {

glm::vec3 CalculateTriangleCentroid(const face_s &face) {
  return (face.vertices[0].position + face.vertices[1].position + face.vertices[2].position) / 3.0f;
}

glm::vec3 CalculateTriangleNormal(const face_s &face) {
  glm::vec3 edge1 = face.vertices[1].position - face.vertices[0].position;
  glm::vec3 edge2 = face.vertices[2].position - face.vertices[0].position;

  return glm::normalize(cross(edge1, edge2));
}

float CalculateTriangleArea(const glm::vec3 vertices[3]) {
  const glm::vec3 edge1 = vertices[1] - vertices[0];
  const glm::vec3 edge2 = vertices[2] - vertices[0];

  return length(cross(edge1, edge2)) / 2.0f;
}

float CalculateFormFactor(const face_s &face_i, const face_s &face_j) {
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

  // The vectors form the three corners of face j to the centroid of face
  // i
  glm::vec3 i_corner_to_j_centroid[3] = {glm::normalize(face_j.vertices[0].position - i_centroid),
                                         glm::normalize(face_j.vertices[1].position - i_centroid),
                                         glm::normalize(face_j.vertices[2].position - i_centroid)};

  // Map the face i on hemisphere
  glm::vec3 i_corner_on_hemisphere[3] = {i_centroid + i_corner_to_j_centroid[0] * hemisphere_radius,
                                         i_centroid + i_corner_to_j_centroid[1] * hemisphere_radius,
                                         i_centroid + i_corner_to_j_centroid[2] * hemisphere_radius};

  // Map "i_corner_on_hemisphere" to floor from hemisphere
  glm::vec3 map_to_floor_from_hemisphere[3] = {
      i_corner_on_hemisphere[0] - i_normal * dot(i_corner_on_hemisphere[0] - i_centroid, i_normal),
      i_corner_on_hemisphere[1] - i_normal * dot(i_corner_on_hemisphere[1] - i_centroid, i_normal),
      i_corner_on_hemisphere[2] - i_normal * dot(i_corner_on_hemisphere[2] - i_centroid, i_normal)};
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
} // namespace RadCu
