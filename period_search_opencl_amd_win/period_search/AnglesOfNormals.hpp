#pragma once
/*  direction angles of normal of the triangle vertices of Gaussian image sphere
    20th September 2018
*/

#include <vector>
#include <iostream>

struct AnglesOfNormals {
    std::vector<double> theta;
    std::vector<double> phi;
    size_t numberFacets = 0;

    AnglesOfNormals();
    ~AnglesOfNormals();
};

inline AnglesOfNormals::AnglesOfNormals()
{
    std::cout << "constructor AnglesOfNormals" << std::endl;
}

inline AnglesOfNormals::~AnglesOfNormals()
{
    std::cout << "destructor AnglesOfNormals" << std::endl;
}

