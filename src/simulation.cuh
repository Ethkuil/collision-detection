#pragma once
#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include "data.cuh"

// Grid structure for spatial partitioning
struct GridData {
    int* d_gridIDs;           // Grid ID for each particle
    int* d_gridIndices;       // Particle indices in each grid cell
    int* d_gridCounts;        // Particle count in each grid cell
    int3 gridDim;
    float3 gridMin;
    float cellSize;
};

// Initialize grid data structures
GridData initializeGrid(int3 gridDim, float3 gridMin, float cellSize);

// Free grid data structures
void freeGrid(GridData& grid);

// Main simulation step - orchestrates the entire pipeline
void simulationStep(Sphere* d_spheres, int numSpheres,
                    Object* d_objects, int numObjects,
                    float deltaTime, float3 boxMin, float3 boxMax,
                    GridData& grid,
                    Collision* d_collisions, int* d_collisionCount);

#endif // SIMULATION_CUH
