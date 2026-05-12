#pragma once


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
