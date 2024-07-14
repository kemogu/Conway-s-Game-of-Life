// grid üzerinde bulunacak noktaları tanımlayan nesne
class Point
{
private:
    int x;
    int y;
    bool dead;
public:
    __host__ __device__ Point(int x, int y, bool dead);
    __host__ __device__ ~Point();
    // nokta durumları ayarlama
    __host__ __device__ bool isDead();
    __host__ __device__ void setDead(bool state);
};

