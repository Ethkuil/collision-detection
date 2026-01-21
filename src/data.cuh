#pragma once
#ifndef DATA_H
#define DATA_H

// 粒子分为小球和物体
// 小球是可移动的，物体是不可移动的
// 各个小球、物体可能有不同的半径、质量、初速度和弹性系数

struct Particle {
    float3 position;
    float radius;
    float mass;
    float elasticity;
    bool is_sphere; // true表示小球，false表示物体
};

struct Sphere : Particle {
    float3 velocity;

    Sphere() { is_sphere = true; }
};

struct Object : Particle {
  // 物体没有速度
    Object() { is_sphere = false; }
};

struct Collision {
    int sphereAIdx;
    int sphereBIdx;
};

// 用于标识与物体的碰撞
constexpr int COLLISION_OBJECT_IDX = -1;

#endif // DATA_H