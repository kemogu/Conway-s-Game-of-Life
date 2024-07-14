#include "Grid.cuh"

class Game {
private:
    float duration;
    float period;
    Grid* currentGrid;
    Grid* nextGrid;
public:
    Game(int width, int height, float duration, float period);
    ~Game();
    void start();
    void update();
};
