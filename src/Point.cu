#include "Point.cuh"

// nokta oluşturmak için constructor
__host__ __device__ Point::Point(int x, int y, bool dead) : x(x), y(y), dead(dead){}

__host__ __device__ Point::~Point() {}

// noktanın durumunu elde etmek için gerekli method
__host__ __device__ bool Point::isDead() { return dead; }

// noktanın durumunu değiştirmek için gerekli method
__host__ __device__ void Point::setDead(bool state) { dead = state; }