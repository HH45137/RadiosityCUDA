#pragma once

#include <math_constants.h>
#include <math_functions.h>

static __device__ __host__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
    return float3{
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

static __device__ __host__ float3 operator+(const float3 &a, const float &b) {
    return float3{
        a.x + b,
        a.y + b,
        a.z + b
    };
}

static __device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
    return float3{
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

static __device__ __host__ float3 operator-(const float3 &a, const float &b) {
    return float3{
        a.x - b,
        a.y - b,
        a.z - b
    };
}

static __device__ __host__ float3 operator*(const float3 &a, const float3 &b) {
    return float3{
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    };
}

static __device__ __host__ float3 operator*(const float3 &a, const float &b) {
    return float3{
        a.x * b,
        a.y * b,
        a.z * b
    };
}

static __device__ __host__ float3 operator/(const float3 &a, const float3 &b) {
    return float3{
        a.x / b.x,
        a.y / b.y,
        a.z / b.z
    };
}

static __device__ __host__ float3 operator/(const float3 &a, const float &b) {
    return float3{
        a.x / b,
        a.y / b,
        a.z / b
    };
}

static __device__ __host__ float length(const float3 &a) {
    return sqrtf(powf(a.x, 2) + powf(a.y, 2) + powf(a.z, 2));
}

static __device__ __host__ float3 normalized(const float3 &a) {
    float len = length(a);
    if (len < 1e-8f) {
        return float3{0.0f, 0.0f, 0.0f};
    }
    return a / len;
}

static __device__ __host__ float3 cross(const float3 &a, const float3 &b) {
    return {
        (a.y * b.z) - (a.z * b.y),
        (a.z * b.x) - (a.x * b.z),
        (a.x * b.y) - (a.y * b.z)
    };
}
