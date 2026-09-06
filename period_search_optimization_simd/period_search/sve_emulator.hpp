// SVE emulation
// useful for running tests on an x64 hardware without need for a specialized HW or virtualization
// period_search_BOINC --optimization 8
//
// Only the subset of the ACLE that the *_sve.cpp files use is provided, but the
// semantics that matter are reproduced faithfully:
//   - a predicated load zeroes the inactive lanes,
//   - a predicated store leaves the inactive lanes of the memory untouched,
//   - a comparison / reduction never looks at a lane the governing predicate masks off,
//   - the "_x" (don't care) forms compute every lane, like the unpredicated
//     instruction real hardware picks for them.
#pragma once

#include <cstdint>
#include <cmath>

#ifndef SVE_VLEN
  #define SVE_VLEN 64                   /* emulated vector length in bytes */
#endif

#define svcntd() (SVE_VLEN / 8)
#define svcntw() (SVE_VLEN / 4)
#define svcnth() (SVE_VLEN / 2)

// One byte of predicate per byte of data, exactly like a real SVE predicate register,
// so lane i of a double vector is governed by x[8 * i].
struct svbool_t {
    bool x[SVE_VLEN];
};

struct svfloat64_t {
    double x[svcntd()];
};

static inline svbool_t svptrue_b64() {
    svbool_t r{};
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = true;
    return r;
}

static inline svbool_t svwhilelt_b64(int64_t begin, int64_t end) {
    svbool_t r{};
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = i + begin < end;
    return r;
}

static inline svfloat64_t svdup_n_f64(double x) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = x;
    return r;
}

static inline svfloat64_t svdup_f64(double x) {
    return svdup_n_f64(x);
}

static inline svfloat64_t svld1_f64(svbool_t t, double const *x) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (t.x[8 * i] ? x[i] : 0.);
    return r;
}

static inline void svst1_f64(svbool_t t, double *x, svfloat64_t y) {
    for (int i = 0; i < svcntd(); ++i) {
        if (t.x[8 * i])
            x[i] = y.x[i];
    }
}

static inline svfloat64_t svadd_f64_x(svbool_t, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = a.x[i] + b.x[i];
    return r;
}

static inline svfloat64_t svsub_f64_x(svbool_t, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = a.x[i] - b.x[i];
    return r;
}

static inline svfloat64_t svmul_f64_x(svbool_t, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = a.x[i] * b.x[i];
    return r;
}

static inline svfloat64_t svdiv_f64_x(svbool_t, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = a.x[i] / b.x[i];
    return r;
}

// a + b * c
static inline svfloat64_t svmla_f64_x(svbool_t, svfloat64_t a, svfloat64_t b, svfloat64_t c) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = a.x[i] + b.x[i] * c.x[i];
    return r;
}

static inline svfloat64_t svmla_n_f64_x(svbool_t, svfloat64_t a, svfloat64_t b, double c) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = a.x[i] + b.x[i] * c;
    return r;
}

// a - b * c
static inline svfloat64_t svmls_f64_x(svbool_t, svfloat64_t a, svfloat64_t b, svfloat64_t c) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = a.x[i] - b.x[i] * c.x[i];
    return r;
}

static inline svfloat64_t svmls_n_f64_x(svbool_t, svfloat64_t a, svfloat64_t b, double c) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = a.x[i] - b.x[i] * c;
    return r;
}

static inline double svaddv_f64(svbool_t t, svfloat64_t a) {
    double r = 0.;
    for (int i = 0; i < svcntd(); ++i) {
        if (t.x[8 * i])
            r += a.x[i];
    }
    return r;
}

static inline svfloat64_t svsel_f64(svbool_t m, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (m.x[8 * i] ? a.x[i] : b.x[i]);
    return r;
}

static inline svfloat64_t svabs_f64_x(svbool_t, svfloat64_t a) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = std::fabs(a.x[i]);
    return r;
}

static inline svbool_t svcmpge_f64(svbool_t t, svfloat64_t a, svfloat64_t b) {
    svbool_t r{};
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = t.x[8 * i] && a.x[i] >= b.x[i];
    return r;
}

static inline svbool_t svcmplt_f64(svbool_t t, svfloat64_t a, svfloat64_t b) {
    svbool_t r{};
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = t.x[8 * i] && a.x[i] < b.x[i];
    return r;
}

static inline svbool_t svcmpgt_f64(svbool_t t, svfloat64_t a, svfloat64_t b) {
    svbool_t r{};
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = t.x[8 * i] && a.x[i] > b.x[i];
    return r;
}

static inline svbool_t svand_z(svbool_t t, svbool_t a, svbool_t b) {
    svbool_t r{};
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = t.x[8 * i] && a.x[8 * i] && b.x[8 * i];
    return r;
}

static inline bool svptest_any(svbool_t t, svbool_t a) {
    for (int i = 0; i < svcntd(); ++i) {
        if (t.x[8 * i] && a.x[8 * i])
            return true;
    }
    return false;
}

/*
static void print_vector(svbool_t t, svfloat64_t a) {
    for (int i = 0; i < svcntd(); ++i) {
        printf("[%d]=%2f, ", i, t.x[8 * i] ? a.x[i] : 0.);
    }
    printf("\n");
}

static void print_bool(svbool_t a) {
    for (int i = 0; i < svcntd(); ++i) {
        printf("[%d]=%d, ", i, a.x[8 * i]);
    }
    printf("\n");
}
*/
