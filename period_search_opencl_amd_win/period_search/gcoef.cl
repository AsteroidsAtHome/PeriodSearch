__kernel void gcoef(__global double *dyda, const int ncoef0, const int *incl, const int incl_count, const double *dbr,
    const double *dg, const int ydg, const double scale)
{
    uint i = get_global_id(0);
    double _tmpdyda = 0;

    for (int j = 0; j < incl_count; j++) {
        const int q = incl[j];
        _tmpdyda = _tmpdyda + dbr[j] * dg[q * ydg + i];
    }
    dyda[i] = scale * _tmpdyda;
}