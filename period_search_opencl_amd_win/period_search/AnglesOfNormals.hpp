#pragma once
/*  direction angles of normal of the triangle vertices of Gaussian image sphere
    20th September 2018
*/

#include <vector>

struct AnglesOfNormals {
    std::vector<double> theta_angle;
    std::vector<double> phi_angle;
    size_t number_facets;
};
