# 基于GPU的碰撞检测算法

## 程序运行环境和编程环境

> 由于我的本地电脑缺乏NVIDIA独立显卡，故租用了一台 Windows GPU云主机 来调试和运行。

操作系统: Windows Server 2022 Datacenter, 64位

CPU: AMD EPYC 7542 32-Core Processor, 2.90 GHz, x64

GPU: NVIDIA GeForce RTX4090

编程环境: 

1. 在 WSL (Ubuntu) 下使用 Visual Studio Code (clangd) + CUDA Toolkit + freeglut 完成初步开发，并测试了移除`drawBackground()`和`drawParticles()`后，`drawFPS()`能够正常显示FPS。
2. 之后传输到Windows云主机进一步开发测试，安装了 Visual Studio 2022（2026版与CUDA ToolKit尚不兼容）、NVIDIA Driver、CUDA Toolkit 13.1、freeglut.

### 从源码构建

#### 额外依赖库

本项目使用 `GLUT` 渲染，具体使用 `freeglut`.

安装好 `vcpkg` 后，可使用 `vcpkg` 安装所需依赖（`vcpkg.json`已配置好）：

```sh
vcpkg install 
```

Ubuntu 中也可使用系统级的包管理器安装:

```sh
sudo apt-get install build-essential freeglut3 freeglut3-dev binutils-gold
```

#### 构建

```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## 各个程序模块之间的逻辑关系

- `data.cuh` 定义基本的数据结构，小球、物体、碰撞。
- `simulation.cuh` 和 `simulation.cu` 实现了所需的 GPU kernels，进一步封装了 launch 参数，再进一步将多个阶段整合为单一函数接口对外暴露。
- `main.cu` 包含程序的顶层逻辑，指定具体参数，执行初始化、模拟、渲染。
- 渲染库为 `OpenGL GLUT`。

## 程序运行的主要流程

1. 程序启动后，搭建GUI，完成内存分配，初始化模拟状态，执行 `initializeParticles()` 函数随机设置小球、物体的属性。
2. 渲染新一帧前，程序会执行 `displayCallback()` 函数。在其中，程序依次完成 绘制背景、模拟一步（调用`simulation.cuh`中的`simulationStep()`函数，等待执行完成）、根据模拟结果绘制小球和物体。程序还会计算实时FPS并绘制出来。
3. 在`simulationStep()`函数中，程序依次执行 空间划分、碰撞检测、碰撞处理、小球移动。
4. 空间划分中，首先检查小球的位置，其次检查物体的位置，再综合构建反向索引。
5. 碰撞检测会记录所发生的碰撞（小球 vs 小球，小球 vs 物体），在碰撞处理中加以处理，更新小球的速度。
6. 小球移动阶段，根据小球的速度和时间差更新小球的位置，并考虑到重力、空间限制这些额外因素。

## 各个功能的演示方法

启动程序后无需额外操作。

白色的是小球，红色的是物体。用立方体表示物体. 不过碰撞检测仍然是基于球体的。

> 目前测试时只要有绘图就会发生段错误。

## 算法性能测试结果分析

GPU: NVIDIA GeForce RTX4090

表格，FPS关于小球数量

<!-- TODO -->

## 参考文献和引用代码出处

1. [空间数据结构(四叉树/八叉树/BVH树/BSP树/k-d树) - KillerAery - 博客园](https://www.cnblogs.com/KillerAery/p/10878367.html)
2. [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
3. [Chapter 32. Broad-Phase Collision Detection with CUDA | NVIDIA Developer](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
