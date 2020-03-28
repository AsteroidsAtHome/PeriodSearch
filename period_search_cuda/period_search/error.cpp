#include "stdafx.h"
#include <cstdio>
#include <cstdlib>

void ErrorFunction(const char* buffer, int no_conversions) {
    fprintf(stderr, "An error occurred. You entered:\n%s\n", buffer);
    fprintf(stderr, "%d successful conversions", no_conversions);
    exit(EXIT_FAILURE);
}