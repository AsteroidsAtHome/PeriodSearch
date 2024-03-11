// SVE emulation
// useful for running tests on an x64 hardware without need for a specialized HW or virtualization
// period_search_BOINC.exe --optimization 8

#include <cstdint>

#define SVE_VLEN 64
#define svcntd() (SVE_VLEN / 8)
#define svcntw() (SVE_VLEN / 4)
#define svcnth() (SVE_VLEN / 2)

struct svbool_t {
    bool x[SVE_VLEN];
};

static svbool_t svptrue_b64() {
    svbool_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = true;
    return r;
}

static svbool_t svwhilelt_b64(int64_t begin, int64_t end) {
    svbool_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = i + begin < end;
    return r;
}

struct svfloat64_t {
    double x[svcntd()];
};

static svfloat64_t svdup_f64(double x) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = x;
    return r;
}

static svfloat64_t svdup_n_f64(double x) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = x;
    return r;
}

static svfloat64_t svld1_f64(svbool_t t, double const *x) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (t.x[8 * i] ? x[i] : 0.);
    return r;
}

static void svst1_f64(svbool_t t, double *x, svfloat64_t y) {
    for (int i = 0; i < svcntd(); ++i) {
        x[i] = t.x[8 * i] ? y.x[i] : 0.;
    }
}

static svfloat64_t svadd_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (t.x[8 * i] ? a.x[i] + b.x[i] : a.x[i]);
    return r;
}

static svfloat64_t svmla_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b, svfloat64_t c) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (t.x[8 * i] ? a.x[i] + b.x[i] * c.x[i] : a.x[i]);
    return r;
}

static svfloat64_t svmla_n_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b, float c) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (t.x[8 * i] ? a.x[i] + b.x[i] * c : a.x[i]);
    return r;
}

static svfloat64_t svmls_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b, svfloat64_t c) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (t.x[8 * i] ? a.x[i] - b.x[i] * c.x[i] : a.x[i]);
    return r;
}

static svfloat64_t svmul_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (t.x[8 * i] ? a.x[i] * b.x[i] : a.x[i]);
    return r;
}

static svfloat64_t svdiv_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (t.x[8 * i] ? a.x[i] / b.x[i] : a.x[i]);
    return r;
}

static svfloat64_t svsub_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (t.x[8 * i] ? a.x[i] - b.x[i] : a.x[i]);
    return r;
}

static double svaddv_f64(svbool_t t, svfloat64_t a) {
    double r = 0.;
    for (int i = 0; i < svcntd(); ++i) {
        r += t.x[8 * i] ? a.x[i] : 0.;
    }
    return r;
}

static svfloat64_t svsel_f64(svbool_t m, svfloat64_t a, svfloat64_t b) {
    svfloat64_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[i] = (m.x[8 * i] ? a.x[i] : b.x[i]);
    return r;
}

static svbool_t svcmpgt_f64(svbool_t m, svfloat64_t a, svfloat64_t b) {
    svbool_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = a.x[i] > b.x[i] ? true : false;
    return r;
}

static svbool_t svand_z(svbool_t t, svbool_t a, svbool_t b) {
    svbool_t r;
    for (int i = 0; i < svcntd(); ++i)
        r.x[8 * i] = a.x[8 * i] && b.x[8 * i];

    return r;
}

static bool svptest_any(svbool_t t, svbool_t a) {
    bool r = false;
    for (int i = 0; i < svcntd(); ++i)
        r |= a.x[8 * i];

    return r;
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