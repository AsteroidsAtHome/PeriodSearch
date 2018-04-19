#pragma once

class Beta {
private:
    double *ptr;
    int length;

public:
    Beta();
    ~Beta();
    void Init(int n);
    void set(int element, double value);
    double get(size_t x);

};
