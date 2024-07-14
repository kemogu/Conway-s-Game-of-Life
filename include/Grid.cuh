// noktaların üzerinde bulunacağı ızgara nesnesine ait sınıf
// noktaları oluşturmak ve noktaların durumlarını yönetmek için oluşturuldu

#include "Point.cuh"
#include <vector>

class Grid {
private:
    int width;
    int height;
    Point* points; // CUDA uyumlu pointer
public:
    Grid(int w, int h);
    ~Grid();
    void printGrid();
    Point* getPoints();
    int getWidth();
    int getHeight();
};
