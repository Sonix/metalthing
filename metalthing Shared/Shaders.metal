//
//  Shaders.metal
//  metalthing Shared
//
//  Created by Johannes Loepelmann on 16.06.24.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
    float4 color;
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]])
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
    out.texCoord = in.texCoord;
    out.color = float4(in.position + 0.5f, 1.0);
    return out;
}

fragment float4 backfaceFragment(ColorInOut in [[stage_in]])
{
    return float4(in.color);
}

constant int STEPS = 256;

float4 trace(float4 start, float4 end, texture3d<half> volume)
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);
    
    float4 acc = float4(0.f);
    float4 dist = end - start;
    for(int i = 0; i < STEPS; ++i) {
        float3 pos = (start + dist * i/STEPS).xyz;
        float current = volume.sample(colorSampler, pos).r;
        acc += current / STEPS;
    }
    return acc;
}

float4 trace(float4 start, float4 end)
{
    return start + end;
}



fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               texture2d<half> colorMap     [[ texture(TextureIndexColor) ]],
                               texture3d<half> volume     [[ texture(TextureIndexVolume) ]])
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    float4 rayEnd =  float4(colorMap.sample(colorSampler, in.position.xy / uniforms.screenSize));
    float4 rayStart = in.color;

    return trace(rayStart, rayEnd, volume);
}

