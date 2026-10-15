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

