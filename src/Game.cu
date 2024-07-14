#include "Game.cuh"
#include <cuda_runtime.h>
#include <thread>
#include <chrono> 

__device__ int getNeighborCount(Point* points, int width, int height, int x, int y) {
    int count = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) continue;
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (!points[ny * width + nx].isDead()) {
                    count++;
                }
            }
        }
    }
    return count;
}

__global__ void updateGrid(Point* points, Point* newPoints, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int neighbors = getNeighborCount(points, width, height, x, y);
        bool isDead = points[y * width + x].isDead();
        bool newDeadState = isDead;

        if (isDead && neighbors == 3) {
            newDeadState = false; // Canlanma
        } else if (!isDead && (neighbors < 2 || neighbors > 3)) {
            newDeadState = true; // Ölüm
        }

        newPoints[y * width + x] = Point(x, y, newDeadState);
    }
}

Game::Game(int width, int height, float duration, float period) : duration(duration), period(period) {
    currentGrid = new Grid(width, height);
    nextGrid = new Grid(width, height);
}

Game::~Game() {
    delete currentGrid;
    delete nextGrid;
}

void Game::start() {
    float time = 0;
    while (time < duration) {
        auto start_time = std::chrono::steady_clock::now();

        update();

        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = end_time - start_time;
        std::this_thread::sleep_for(std::chrono::duration<float>(period) - elapsed);

        time += period;
    }
}

void Game::update() {
    Point* d_points = currentGrid->getPoints();
    Point* d_newPoints = nextGrid->getPoints();
    int width = currentGrid->getWidth();
    int height = currentGrid->getHeight();

    // Kernel çağrısı
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    updateGrid<<<gridSize, blockSize>>>(d_points, d_newPoints, width, height);

    cudaDeviceSynchronize();

    // currentGrid ve nextGrid pointerlarını değiştir
    Grid* temp = currentGrid;
    currentGrid = nextGrid;
    nextGrid = temp;

    // Güncellenmiş grid'i ekrana bas
    currentGrid->printGrid();
}
