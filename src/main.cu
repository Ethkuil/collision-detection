#include <cstdlib>
#include <ctime>
#include <sstream>
#include <iomanip>

#include <GL/glut.h>

#include "data.cuh"
#include "simulation.cuh"

// 空间大小
constexpr int SPACE_SIZE = 100;

// 时间步长
constexpr float TIME_STEP = 0.016f; // 16ms

// 固定小球为白色，物体为红色
using Color = float3;
constexpr Color SPHERE_COLOR = {1.0f, 1.0f, 1.0f};
constexpr Color OBJECT_COLOR = {1.0f, 0.0f, 0.0f};
// 粒子数量
constexpr int NUM_SPHERES = 128;
constexpr int NUM_OBJECTS = 32;
constexpr int NUM_PARTICLES = NUM_SPHERES + NUM_OBJECTS;
// 粒子半径范围
constexpr float MIN_RADIUS = 0.5f;
constexpr float MAX_RADIUS = 2.0f;
// 粒子质量范围
constexpr float MIN_MASS = 1.0f;
constexpr float MAX_MASS = 10.0f;
// 粒子弹性系数范围
constexpr float MIN_ELASTICITY = 0.5f;
constexpr float MAX_ELASTICITY = 1.0f;
// 小球初速度范围
constexpr float MAX_INITIAL_VELOCITY = 10.0f;

// 全局粒子数据
Sphere* h_spheres;
Object* h_objects;
Sphere* d_spheres;
Object* d_objects;

// Grid and collision data
GridData g_gridData;
Collision* d_collisions;
int* d_collisionCount;

constexpr int MAX_COLLISIONS = 100000;
constexpr int3 GRID_DIM = {10, 10, 10};
constexpr float CELL_SIZE = SPACE_SIZE / 10.0f;

// 时间和FPS相关全局变量
static int g_lastTime = 0;
static float g_frameTime = 0.016f;
static int g_frameCount = 0;
static float g_fps = 0.0f;
static int g_fpsUpdateTime = 0;

void initializeParticles() {
    // 随机初始化小球和物体的位置等属性
    h_spheres = new Sphere[NUM_SPHERES];
    h_objects = new Object[NUM_OBJECTS];
    for (int i = 0; i < NUM_SPHERES; ++i) {
        Sphere sphere;
        sphere.position = {
            static_cast<float>(rand() % SPACE_SIZE),
            static_cast<float>(rand() % SPACE_SIZE),
            static_cast<float>(rand() % SPACE_SIZE)
        };
        sphere.radius = MIN_RADIUS + static_cast<float>(rand()) / RAND_MAX * (MAX_RADIUS - MIN_RADIUS);
        sphere.mass = MIN_MASS + static_cast<float>(rand()) / RAND_MAX * (MAX_MASS - MIN_MASS);
        sphere.elasticity = MIN_ELASTICITY + static_cast<float>(rand()) / RAND_MAX * (MAX_ELASTICITY - MIN_ELASTICITY);
        sphere.velocity = {
            static_cast<float>(rand()) / RAND_MAX * MAX_INITIAL_VELOCITY,
            static_cast<float>(rand()) / RAND_MAX * MAX_INITIAL_VELOCITY,
            static_cast<float>(rand()) / RAND_MAX * MAX_INITIAL_VELOCITY             
        };
        h_spheres[i] = sphere;
    }
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        Object object;
        object.position = {
            static_cast<float>(rand() % SPACE_SIZE),
            static_cast<float>(rand() % SPACE_SIZE),
            static_cast<float>(rand() % SPACE_SIZE)};
        object.radius = MIN_RADIUS + static_cast<float>(rand()) / RAND_MAX * (MAX_RADIUS - MIN_RADIUS);
        object.mass = MIN_MASS + static_cast<float>(rand()) / RAND_MAX * (MAX_MASS - MIN_MASS);
        object.elasticity = MIN_ELASTICITY + static_cast<float>(rand()) / RAND_MAX * (MAX_ELASTICITY - MIN_ELASTICITY);
        h_objects[i] = object;
    }

    cudaMalloc(&d_spheres, sizeof(Sphere) * NUM_SPHERES);
    cudaMalloc(&d_objects, sizeof(Object) * NUM_OBJECTS);
    cudaMemcpy(d_spheres, h_spheres, sizeof(Sphere) * NUM_SPHERES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_objects, h_objects, sizeof(Object) * NUM_OBJECTS, cudaMemcpyHostToDevice);
}

void drawBackground() {
  // 绘制墙壁
  glColor3f(0.5f, 0.5f, 0.5f);
  glBegin(GL_QUADS);
  // 底面
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(SPACE_SIZE, 0.0f, 0.0f);
  glVertex3f(SPACE_SIZE, 0.0f, SPACE_SIZE);
  glVertex3f(0.0f, 0.0f, SPACE_SIZE);
  // 顶面
  glVertex3f(0.0f, SPACE_SIZE, 0.0f);
  glVertex3f(SPACE_SIZE, SPACE_SIZE, 0.0f);
  glVertex3f(SPACE_SIZE, SPACE_SIZE, SPACE_SIZE);
  glVertex3f(0.0f, SPACE_SIZE, SPACE_SIZE);
  // 四壁
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, SPACE_SIZE, 0.0f);
  glVertex3f(0.0f, SPACE_SIZE, SPACE_SIZE);
  glVertex3f(0.0f, 0.0f, SPACE_SIZE);
  glVertex3f(SPACE_SIZE, 0.0f, 0.0f);
  glVertex3f(SPACE_SIZE, SPACE_SIZE, 0.0f);
  glVertex3f(SPACE_SIZE, SPACE_SIZE, SPACE_SIZE);
  glVertex3f(SPACE_SIZE, 0.0f, SPACE_SIZE);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(SPACE_SIZE, 0.0f, 0.0f);
  glVertex3f(SPACE_SIZE, SPACE_SIZE, 0.0f);
  glVertex3f(0.0f, SPACE_SIZE, 0.0f);
  glVertex3f(0.0f, 0.0f, SPACE_SIZE);
  glVertex3f(SPACE_SIZE, 0.0f, SPACE_SIZE);
  glVertex3f(SPACE_SIZE, SPACE_SIZE, SPACE_SIZE);
  glVertex3f(0.0f, SPACE_SIZE, SPACE_SIZE);
  glEnd();
}

void drawParticles(const Sphere* spheres, const Object* objects, int numSpheres, int numObjects) {
    // 绘制小球
    for (int i = 0; i < numSpheres; ++i) {
        const Sphere& sphere = spheres[i];
        glColor3f(SPHERE_COLOR.x, SPHERE_COLOR.y, SPHERE_COLOR.z);
        glPushMatrix();
        glTranslatef(sphere.position.x, sphere.position.y, sphere.position.z);
        glutSolidSphere(sphere.radius, 20, 20);
        glPopMatrix();
    }
    // 绘制物体
    for (int i = 0; i < numObjects; ++i) {
        const Object& object = objects[i];
        glColor3f(OBJECT_COLOR.x, OBJECT_COLOR.y, OBJECT_COLOR.z);
        glPushMatrix();
        glTranslatef(object.position.x, object.position.y, object.position.z);
        glutSolidCube(object.radius * 2); // 用立方体表示物体. 不过碰撞检测仍然是基于球体的
        glPopMatrix();
    }
}

void drawFPS() {
    // 设置2D投影用于绘制文字
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, 800, 600, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glColor3f(0.0f, 1.0f, 0.0f);

    // 使用GLUT位图字体绘制FPS
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << "FPS: " << g_fps;
    std::string fpsStr = oss.str();

    glRasterPos2f(10.0f, 30.0f);
    for (char c : fpsStr) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
    }

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void simulateStep(Sphere* d_spheres, Object* d_objects, int numSpheres, int numObjects, float timeStep) {
    simulationStep(d_spheres, numSpheres, d_objects, numObjects, timeStep,
                   make_float3(0.0f, 0.0f, 0.0f),
                   make_float3(SPACE_SIZE, SPACE_SIZE, SPACE_SIZE),
                   g_gridData, d_collisions, d_collisionCount);
    cudaDeviceSynchronize();
}

void displayCallback() {
    // 计算实际帧时间
    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    if (g_lastTime == 0) {
        g_lastTime = currentTime;
    }
    g_frameTime = (currentTime - g_lastTime) / 1000.0f;
    g_lastTime = currentTime;

    // 更新FPS计数
    g_frameCount++;
    if (currentTime - g_fpsUpdateTime >= 1000) {
        g_fps = g_frameCount * 1000.0f / (currentTime - g_fpsUpdateTime);
        g_fpsUpdateTime = currentTime;
        g_frameCount = 0;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushMatrix();

    drawBackground();

    // 模拟一步
    simulateStep(d_spheres, d_objects, NUM_SPHERES, NUM_OBJECTS, g_frameTime);

    // 绘制粒子
    cudaMemcpy(h_spheres, d_spheres, sizeof(Sphere) * NUM_SPHERES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_objects, d_objects, sizeof(Object) * NUM_OBJECTS, cudaMemcpyDeviceToHost);
    drawParticles(h_spheres, h_objects, NUM_SPHERES, NUM_OBJECTS);

    glPopMatrix();

    drawFPS();

    glutSwapBuffers();
    glutPostRedisplay();
}

void cleanup() {
    freeGrid(g_gridData);
    cudaFree(d_collisions);
    cudaFree(d_collisionCount);
    cudaFree(d_spheres);
    cudaFree(d_objects);
    delete[] h_spheres;
    delete[] h_objects;
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Collision Detection Simulation");
    
    initializeParticles();
    
    // Initialize grid and collision data
    g_gridData = initializeGrid(GRID_DIM, make_float3(0.0f, 0.0f, 0.0f), CELL_SIZE);
    cudaMalloc(&d_collisions, sizeof(Collision) * MAX_COLLISIONS);
    cudaMalloc(&d_collisionCount, sizeof(int));
    
    // Register callbacks
    glutDisplayFunc(displayCallback);
    
    // Setup OpenGL
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    // Start main loop
    glutMainLoop();
    
    cleanup();
    return 0;
}
