#include "simulation.cuh"

#include "data.cuh"

constexpr int NUM_PARTICLES_PER_CELL_MAX = 1024;

constexpr float GRAVITY = 9.81f;

// 粒子分为小球和物体
// 小球是可移动的，物体是不可移动的

// 1. 空间划分：构建均匀网格，每个网格单元记录落入其中的粒子（小球/物体） ID 列表。

// 每个线程处理一个小球，计算其所在网格 ID 并存储到 gridIDs 中。
__global__ void assignSpheresToGridKernel(Sphere* d_spheres, int numSpheres,
                                      float3 gridMin, float cellSize, int3 gridDim,
                                      int* gridIDs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSpheres) return;

    Sphere sphere = d_spheres[idx];
    int3 gridPos;
    gridPos.x = static_cast<int>((sphere.position.x - gridMin.x) / cellSize);
    gridPos.y = static_cast<int>((sphere.position.y - gridMin.y) / cellSize);
    gridPos.z = static_cast<int>((sphere.position.z - gridMin.z) / cellSize);
    // Clamp to grid dimensions
    gridPos.x = max(0, min(gridPos.x, gridDim.x - 1));
    gridPos.y = max(0, min(gridPos.y, gridDim.y - 1));
    gridPos.z = max(0, min(gridPos.z, gridDim.z - 1));
    int gridID = gridPos.z * gridDim.y * gridDim.x +
                 gridPos.y * gridDim.x +
                 gridPos.x;
    gridIDs[idx] = gridID;
}

// 每个线程处理一个物体，计算其所在网格 ID 并存储到 gridIDs 中。跟在小球后面。
__global__ void assignObjectsToGridKernel(Object* d_objects, int numObjects,
                                    float3 gridMin, float cellSize, int3 gridDim,
                                    int* gridIDs, int numSpheres) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObjects) return;

    Object object = d_objects[idx];
    int3 gridPos;
    gridPos.x = static_cast<int>((object.position.x - gridMin.x) / cellSize);
    gridPos.y = static_cast<int>((object.position.y - gridMin.y) / cellSize);
    gridPos.z = static_cast<int>((object.position.z - gridMin.z) / cellSize);
    // Clamp to grid dimensions
    gridPos.x = max(0, min(gridPos.x, gridDim.x - 1));
    gridPos.y = max(0, min(gridPos.y, gridDim.y - 1));
    gridPos.z = max(0, min(gridPos.z, gridDim.z - 1));
    int gridID = gridPos.z * gridDim.y * gridDim.x +
                 gridPos.y * gridDim.x +
                 gridPos.x;
    gridIDs[numSpheres + idx] = gridID;
}

// ### `buildGridStructure`
// 按 `gridID` 构建 `gridIndices` 和 `gridCounts`。
// 使用 `atomicAdd` 统计每个网格内的粒子数，并根据返回值确定粒子在 `gridIndices`
// 中的索引，以防止写冲突。
// gridIndices 的布局为二维数组，每个网格对应一行，存储该网格内粒子的索引。
// hack: 假设每个网格内的粒子数不超过 `NUM_PARTICLES_PER_CELL_MAX`。
__global__ void buildGridStructureKernel(int* gridIDs, int numParticles,
                                   int* gridIndices, int* gridCounts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int gridID = gridIDs[idx];
    int count = atomicAdd(&gridCounts[gridID], 1);
    gridIndices[gridID * NUM_PARTICLES_PER_CELL_MAX + count] = idx;
}

// ### `detectCollisions`
// 每个线程处理一个小球。为了避免重复检测，仅检测 `i < j` 的对。
// 2. 碰撞检测：对每个小球，对其所在及相邻的27个网格执行碰撞检测（距离 < 半径和）。
__global__ void detectCollisionsKernel(Sphere* d_spheres, int numSpheres,
                                 Object* d_objects, int numObjects,
                                 int *gridIDs,
                                 int *gridIndices, int* gridCounts,
                                 Collision* d_collisions, int* d_collisionCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSpheres) return; // 仅处理小球
    // trick: 由于 gridIDs 的布局是小球在前，物体在后，因此小球在 gridIDs 中的索引即为其在 d_spheres 中的索引

    Sphere sphereA = d_spheres[idx];

    int3 gridPos;
    int gridID = gridIDs[idx];
    // 计算 gridPos
    gridPos.z = gridID / (gridCounts[0] * gridCounts[1]);
    gridPos.y = (gridID / gridCounts[0]) % gridCounts[1];
    gridPos.x = gridID % gridCounts[0];
    // 遍历相邻27个网格
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
                // 计算 neighborGridID
                int neighborGridID = neighborPos.z * gridCounts[0] * gridCounts[1] +
                                     neighborPos.y * gridCounts[0] +
                                     neighborPos.x;
                // 获取该网格内粒子数
                int count = gridCounts[neighborGridID];
                for (int i = 0; i < count; i++) {
                    int particleIdx = gridIndices[neighborGridID * NUM_PARTICLES_PER_CELL_MAX + i];
                    if (particleIdx <= idx) continue; // 避免重复检测
                    if (particleIdx < numSpheres) {
                        // 碰撞检测小球 vs 小球
                        Sphere sphereB = d_spheres[particleIdx];
                        float3 diff = make_float3(sphereB.position.x - sphereA.position.x,
                                                  sphereB.position.y - sphereA.position.y,
                                                  sphereB.position.z - sphereA.position.z);
                        float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        float radiusSum = sphereA.radius + sphereB.radius;
                        if (distSq < radiusSum * radiusSum) {
                            // 记录碰撞
                            int collIdx = atomicAdd(d_collisionCount, 1);
                            d_collisions[collIdx].sphereAIdx = idx;
                            d_collisions[collIdx].sphereBIdx = particleIdx;
                        }
                    } else {
                        // 碰撞检测小球 vs 物体
                        int objectIdx = particleIdx - numSpheres;
                        Object objectB = d_objects[objectIdx];
                        float3 diff = make_float3(objectB.position.x - sphereA.position.x,
                                                  objectB.position.y - sphereA.position.y,
                                                  objectB.position.z - sphereA.position.z);
                        float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        float radiusSum = sphereA.radius + objectB.radius;
                        if (distSq < radiusSum * radiusSum) {
                            // 记录碰撞
                            int collIdx = atomicAdd(d_collisionCount, 1);
                            d_collisions[collIdx].sphereAIdx = idx;
                            d_collisions[collIdx].sphereBIdx = COLLISION_OBJECT_IDX;
                        }   
                    }
                }
            }
        }
    }
}

// ### `resolveCollisions`
// 基于碰撞检测的结果更新小球的速度。
// 每个线程处理一个小球。
__global__ void resolveCollisionsKernel(Sphere* d_spheres, int numSpheres,
                              Collision* d_collisions, int numCollisions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSpheres) return;
    Sphere& sphereA = d_spheres[idx];

    for (int i = 0; i < numCollisions; i++) {
        Collision coll = d_collisions[i];
        if (coll.sphereAIdx == idx) {
            if (coll.sphereBIdx != COLLISION_OBJECT_IDX) {
                // 小球 vs 小球碰撞
                Sphere& sphereB = d_spheres[coll.sphereBIdx];
                float3 normal = make_float3(sphereB.position.x - sphereA.position.x,
                                            sphereB.position.y - sphereA.position.y,
                                            sphereB.position.z - sphereA.position.z);
                float dist = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
                normal.x /= dist; normal.y /= dist; normal.z /= dist;

                float3 relativeVelocity = make_float3(sphereA.velocity.x - sphereB.velocity.x,
                                                      sphereA.velocity.y - sphereB.velocity.y,
                                                      sphereA.velocity.z - sphereB.velocity.z);
                float velAlongNormal = relativeVelocity.x * normal.x +
                                      relativeVelocity.y * normal.y +
                                      relativeVelocity.z * normal.z;
                if (velAlongNormal > 0) continue; // 已远离

                float e = min(sphereA.elasticity, sphereB.elasticity);
                float j = -(1 + e) * velAlongNormal;
                j /= (1 / sphereA.mass + 1 / sphereB.mass);

                float3 impulse = make_float3(j * normal.x, j * normal.y, j * normal.z);
                sphereA.velocity.x += impulse.x / sphereA.mass;
                sphereA.velocity.y += impulse.y / sphereA.mass;
                sphereA.velocity.z += impulse.z / sphereA.mass;

                sphereB.velocity.x -= impulse.x / sphereB.mass;
                sphereB.velocity.y -= impulse.y / sphereB.mass;
                sphereB.velocity.z -= impulse.z / sphereB.mass;
            } else {
                // 小球 vs 物体碰撞
                sphereA.velocity.x -= 2 * sphereA.velocity.x * sphereA.elasticity;
                sphereA.velocity.y -= 2 * sphereA.velocity.y * sphereA.elasticity;
                sphereA.velocity.z -= 2 * sphereA.velocity.z * sphereA.elasticity;
            }
        }
    }
}

// ### `sphereMove`
// 每个线程处理一个小球，更新小球的位置。
__global__ void sphereMoveKernel(Sphere* d_spheres, int numSpheres, float deltaTime,
                            float3 boxMin, float3 boxMax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSpheres) return;

    Sphere& sphere = d_spheres[idx];
    sphere.position.x += sphere.velocity.x * deltaTime;
    sphere.position.y += sphere.velocity.y * deltaTime;
    sphere.position.z += sphere.velocity.z * deltaTime;

    // 重力影响
    sphere.velocity.y -= GRAVITY * deltaTime;

    // 撞墙反弹
    if (sphere.position.x - sphere.radius < boxMin.x) {
        sphere.position.x = boxMin.x + sphere.radius;
        sphere.velocity.x = -sphere.velocity.x * sphere.elasticity;
    } else if (sphere.position.x + sphere.radius > boxMax.x) {
        sphere.position.x = boxMax.x - sphere.radius;
        sphere.velocity.x = -sphere.velocity.x * sphere.elasticity;
    }
    if (sphere.position.y - sphere.radius < boxMin.y) {
        sphere.position.y = boxMin.y + sphere.radius;
        sphere.velocity.y = -sphere.velocity.y * sphere.elasticity;
    } else if (sphere.position.y + sphere.radius > boxMax.y) {
        sphere.position.y = boxMax.y - sphere.radius;
        sphere.velocity.y = -sphere.velocity.y * sphere.elasticity;
    }
    if (sphere.position.z - sphere.radius < boxMin.z) {
        sphere.position.z = boxMin.z + sphere.radius;
        sphere.velocity.z = -sphere.velocity.z * sphere.elasticity;
    } else if (sphere.position.z + sphere.radius > boxMax.z) {
        sphere.position.z = boxMax.z - sphere.radius;
        sphere.velocity.z = -sphere.velocity.z * sphere.elasticity;
    }
}


GridData initializeGrid(int3 gridDim, float3 gridMin, float cellSize) {
    GridData grid;
    grid.gridDim = gridDim;
    grid.gridMin = gridMin;
    grid.cellSize = cellSize;
    
    int totalCells = gridDim.x * gridDim.y * gridDim.z;
    cudaMalloc(&grid.d_gridIDs, sizeof(int) * (gridDim.x * gridDim.y * gridDim.z * NUM_PARTICLES_PER_CELL_MAX));
    cudaMalloc(&grid.d_gridIndices, sizeof(int) * totalCells * NUM_PARTICLES_PER_CELL_MAX);
    cudaMalloc(&grid.d_gridCounts, sizeof(int) * totalCells);
    cudaMemset(grid.d_gridCounts, 0, sizeof(int) * totalCells);
    
    return grid;
}

void freeGrid(GridData& grid) {
    cudaFree(grid.d_gridIDs);
    cudaFree(grid.d_gridIndices);
    cudaFree(grid.d_gridCounts);
}

void assignParticlesToGrid(Sphere* d_spheres, int numSpheres,
                           Object* d_objects, int numObjects,
                           GridData& grid) {
    cudaMemset(grid.d_gridCounts, 0, sizeof(int) * grid.gridDim.x * grid.gridDim.y * grid.gridDim.z);
    
    int blockSize = 256;
    int gridSize = (numSpheres + blockSize - 1) / blockSize;
    assignSpheresToGridKernel<<<gridSize, blockSize>>>(d_spheres, numSpheres,
                                                   grid.gridMin, grid.cellSize, grid.gridDim,
                                                   grid.d_gridIDs);
    
    gridSize = (numObjects + blockSize - 1) / blockSize;
    assignObjectsToGridKernel<<<gridSize, blockSize>>>(d_objects, numObjects,
                                                  grid.gridMin, grid.cellSize, grid.gridDim,
                                                  grid.d_gridIDs, numSpheres);
    
    int totalParticles = numSpheres + numObjects;
    gridSize = (totalParticles + blockSize - 1) / blockSize;
    buildGridStructureKernel<<<gridSize, blockSize>>>(grid.d_gridIDs, totalParticles,
                                                grid.d_gridIndices, grid.d_gridCounts);
}

void detectCollisions(Sphere* d_spheres, int numSpheres,
                          Object* d_objects, int numObjects,
                          GridData& grid,
                          Collision* d_collisions, int* d_collisionCount) {
    int blockSize = 256;
    int gridSize = (numSpheres + blockSize - 1) / blockSize;
    detectCollisionsKernel<<<gridSize, blockSize>>>(d_spheres, numSpheres,
                                              d_objects, numObjects,
                                              grid.d_gridIDs,
                                              grid.d_gridIndices, grid.d_gridCounts,
                                              d_collisions, d_collisionCount);
}

void resolveCollisions(Sphere* d_spheres, int numSpheres,
                           Collision* d_collisions, int* d_collisionCount) {
    int collisionCount;
    cudaMemcpy(&collisionCount, d_collisionCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    int blockSize = 256;
    int gridSize = (numSpheres + blockSize - 1) / blockSize;
    resolveCollisionsKernel<<<gridSize, blockSize>>>(d_spheres, numSpheres,
                                               d_collisions, collisionCount);
}

void sphereMove(Sphere* d_spheres, int numSpheres, float deltaTime,
                 float3 boxMin, float3 boxMax) {
    int blockSize = 256;
    int gridSize = (numSpheres + blockSize - 1) / blockSize;
    sphereMoveKernel<<<gridSize, blockSize>>>(d_spheres, numSpheres, deltaTime, boxMin, boxMax);
}

void simulationStep(Sphere* d_spheres, int numSpheres,
                    Object* d_objects, int numObjects,
                    float deltaTime, float3 boxMin, float3 boxMax,
                    GridData& grid,
                    Collision* d_collisions, int* d_collisionCount) {
    // Reset collision count
    cudaMemset(d_collisionCount, 0, sizeof(int));
    
    // 1. Assign particles to grid
    assignParticlesToGrid(d_spheres, numSpheres, d_objects, numObjects, grid);
    
    // 2. Detect collisions
    detectCollisions(d_spheres, numSpheres, d_objects, numObjects, grid, d_collisions, d_collisionCount);
    
    // 3. Resolve collisions
    resolveCollisions(d_spheres, numSpheres, d_collisions, d_collisionCount);
    
    // 4. Move spheres
    sphereMove(d_spheres, numSpheres, deltaTime, boxMin, boxMax);
}
