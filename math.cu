#pragma once


static __device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ float3 add3(float3 a, float3 b) {
    return float3{
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

static __device__ float3 sub3(float3 a, float3 b) {
    return float3{
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

static __device__ float3 mul3(float3 a, float3 b) {
    return float3{
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    };
}

static __device__ float3 div3(float3 a, float3 b) {
    return float3{
        a.x / b.x,
        a.y / b.y,
        a.z / b.z
    };
}

static __device__ float dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static __device__ float4 add4(float4 a, float4 b) {
    return float4{
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        a.w + b.w
    };
}

static __device__ float4 sub4(float4 a, float4 b) {
    return float4{
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
        a.w - b.w
    };
}

static __device__ float4 mul4(float4 a, float4 b) {
    return float4{
        a.x * b.x,
        a.y * b.y,
        a.z * b.z,
        a.w * b.w
    };
}

static __device__ float4 div4(float4 a, float4 b) {
    return float4{
        a.x / b.x,
        a.y / b.y,
        a.z / b.z,
        a.w / b.w
    };
}

static __device__ float length3(const float3 a) {
    return sqrtf(powf(a.x, 2) + powf(a.y, 2) + powf(a.z, 2));
}
