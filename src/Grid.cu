#include "Grid.cuh"
#include <iostream>
#include <cuda_runtime.h>

Grid::Grid(int w, int h) : width(w), height(h) {
    cudaMallocManaged(&points, width * height * sizeof(Point));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            points[i * width + j] = Point(i, j, rand() % 2 == 0);
        }
    }
}

Grid::~Grid() {
    cudaFree(points);
}

void Grid::printGrid() {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << (points[i * width + j].isDead() ? "." : "O") << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "----------------" << std::endl; 
}

Point* Grid::getPoints() {
    return points;
}

int Grid::getWidth() {
    return width;
}

int Grid::getHeight() {
    return height;
}
