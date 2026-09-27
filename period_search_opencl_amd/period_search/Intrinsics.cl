/* WORKAROUND(rusticl / aco): runtime f64 '/' returns results with ~3*2^-29
   relative error (verified by [DIVTEST]); fma() and '*' are exact.
   Markstein sequence: two Newton steps refine the reciprocal, the final
   fused correction restores correct rounding.
   NATIVE_DIV_OK=1 (set by the host after the startup probe) replaces the
   helper with plain '/' on drivers whose division is correctly rounded;
   both paths then produce identical bits, so determinism is preserved. */
#ifndef NATIVE_DIV_OK
#define NATIVE_DIV_OK 0
#endif
#if NATIVE_DIV_OK
#define ddiv(a, b) ((a) / (b))
#else
inline double ddiv(double a, double b)
{
    double r = 1.0 / b;
    double e = fma(-b, r, 1.0);
    r = fma(r, e, r);
    e = fma(-b, r, 1.0);
    r = fma(r, e, r);
    double q = a * r;
    return fma(fma(-b, q, a), r, q);
}
#endif

/*
    FROM stackoverflow: https://stackoverflow.com/questions/42856717/intrinsics-equivalent-to-the-cuda-type-casting-intrinsics-double2loint-doub
    You can express these operations via a union. This will not create extra overhead with modern compilers as long as optimization is on (nvcc -O3 ...).
*/

//struct HiLo
//{
//    int lo;
//    int hi;
//};
//
//typedef struct HiLo hilo;
//
//union U {
//    double val;
//    hilo hiLo;
//};
//
//double HiLoint2double(int hi, int lo)
//{
//    union U u;
//
//    u.hiLo.hi = hi;
//    u.hiLo.lo = lo;
//
//    return u.val;
//}

typedef union {
    double val;
    struct {
        int lo;
        int hi;
    };
} un;

double HiLoint2double(int hi, int lo)
{
    /*union {
        double val;
        struct {
            int lo;
            int hi;
        };
    } u;*/
    un u;

    u.hi = hi;
    u.lo = lo;
    return u.val;
}


int double2hiint(double val)
{
    un u;
    u.val = val;
    return u.hi;
}

int double2loint(double val)
{
    un u;
    u.val = val;
    return u.lo;
}

//int __double2hiint(double val)
//{
//    union {
//        double val;
//        struct {
//            int lo;
//            int hi;
//        };
//    } u;
//    u.val = val;
//
//    return u.hi;
//}
//
//int __double2loint(double val)
//{
//    union {
//        double val;
//        struct {
//            int lo;
//            int hi;
//        };
//    } u;
//    u.val = val;
//
//    return u.lo;
//}
//
//int2 __double2int2(double val) {
//    int2 result;
//
//    result.x = __double2hiint(val);
//    result.y = __double2loint(val);
//
//    return result;
//}

