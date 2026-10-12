#!/usr/bin/env python3
"""Generate src/ps_core.cpp from period_search/period_search_BOINC.cpp.

The main() of period_search is turned into  int ps_run_impl(const char*, PsHooks*)
with all BOINC I/O removed. The numerical code is left untouched, so the viewer computes
exactly what period_search computes. What changes:
  * input file opened directly, output rows passed to a callback instead of a file
  * optional override of the period interval and of the convexity weight
    (a fixed weight skips the convexity-weight optimisation, like PS_FIXED_CONW)
  * a cancel flag checked in every loop, progress/iteration callbacks
  * the Gaussian image (facet areas) of the best solution is kept
  * fatal errors throw instead of calling exit()
Re-run this script if period_search_BOINC.cpp changes:
  python3 scripts/gen_ps_core.py ../../period_search/period_search_BOINC.cpp src/ps_core.cpp
"""
import re, sys

src = open(sys.argv[1]).read()   # text mode: CRLF -> LF
dst = sys.argv[2]

def take(a, b):
    i = src.index(a); j = src.index(b, i) + len(b)
    return src[i:j]

def rep(s, old, new, count=1):
    n = s.count(old)
    assert n >= count, f"pattern not found ({n}x): {old[:60]!r}"
    return s.replace(old, new) if count == -1 else s.replace(old, new, count)

globals_block = take("/* global parameters */", "Weight[MAX_N_OBS+1];")
body = take("int main(int argc, char **argv) {", "    boinc_finish(0);\n}")

body = rep(body, "int main(int argc, char **argv) {", "int ps_run_impl(const char* in_path, PsHooks* H) {")
body = rep(body, "isnan(", "std::isnan(", -1)
i = body.index("    retval = boinc_init();")
j = body.index("    boinc_register_timer_callback(update_shmem);\n#endif\n") + len("    boinc_register_timer_callback(update_shmem);\n#endif\n")
body = body[:i] + """    infile = fopenUtf8(in_path, "r");
    if (!infile) throw std::runtime_error(std::string("cannot open ") + in_path);
    out.H = H;
    n_start_from = 1;
""" + body[j:]
body = rep(body, "fscanf(infile, \"%lf\", &Alamda_start);", "fscanf(infile, \"%lf\", &Alamda_start);   PS_OVERRIDES;")
body = rep(body, "    new_conw = 0;\n", "    new_conw = H->conw > 0 ? 1 : 0;   /* fixed weight: skip optimisation */\n")
body = rep(body, "boinc_is_standalone()", "0", -1)
body = rep(body, "       boinc_fraction_done(fraction_done);",
           "       if (H->onPeriod) H->onPeriod(n, (int) ((freq_start - freq_end) / freq_step) + 1, 1 / (freq_start - (n - 1) * freq_step));")
i = body.index("          if (boinc_time_to_checkpoint()) {")
j = body.index("boinc_checkpoint_completed();\n            }\n", i) + len("boinc_checkpoint_completed();\n            }\n")
body = body[:i] + body[j:]
body = rep(body, "exit(2); }", "throw std::runtime_error(\"period_search limit exceeded\"); }", -1)
body = rep(body, "            fscanf(infile, \"%d\", &lc_number);\n            fscanf(infile, \"%lf\", &weight_lc[lc_number]);",
           "            if (fscanf(infile, \"%d\", &lc_number) != 1 || lc_number < 1 || lc_number > Lcurves) break;\n"
           "            if (fscanf(infile, \"%lf\", &weight_lc[lc_number]) != 1) break;")
# cancel checks
body = rep(body, "while (((Niter < n_iter_max) && (iter_diff > iter_diff_max)) || (Niter < n_iter_min))",
           "while ((((Niter < n_iter_max) && (iter_diff > iter_diff_max)) || (Niter < n_iter_min)) && !PS_CANCEL)", -1)
body = rep(body, "for (m = 1; m <= N_POLES; m++)", "for (m = 1; m <= PS_NPOLES && !PS_CANCEL; m++)", -1)
# optional single starting pole (e.g. the pole of a known solution): much faster
body = rep(body, "    ia_lambda_pole = ia_beta_pole = 1;",
           "    ia_lambda_pole = ia_beta_pole = 1;\n"
           "    if (H->singlePole) { lambda_pole[1] = H->poleLambda; beta_pole[1] = H->poleBeta; }")
body = rep(body, "for (n = 1; n <= max_test_periods; n++)", "for (n = 1; n <= max_test_periods && !PS_CANCEL; n++)")
body = rep(body, "for (n = n_start_from; n <= (int) ((freq_start - freq_end) / freq_step) + 1; n++)",
           "for (n = n_start_from; n <= (int) ((freq_start - freq_end) / freq_step) + 1 && !PS_CANCEL; n++)")
body = rep(body, "while ( (new_conw != 1) && ( (conw_r * escl * escl) < 10.0 ) )",
           "while ( (new_conw != 1) && ( (conw_r * escl * escl) < 10.0 ) && !PS_CANCEL )")
# iteration callbacks: 1st occurrence = convexity-weight optimisation, 2nd = main period loop
k = "dev_new = sqrt(rchisq / (ndata - 3));"
a = body.index(k); b = body.index(k, a + 1)
body = body[:b] + k + " if (H->onIter) H->onIter(1, n, m, Niter, dev_new);" + body[b + len(k):]
body = body[:a] + k + " if (H->onIter) H->onIter(0, n, m, Niter, dev_new);" + body[a + len(k):]
if hasattr(str, "x"): pass
# keep the Gaussian image of the best solution (main loop only = 2nd occurrence)
key = "\t   if (dev_new < dev_best)\n\t   {"
a = body.index(key); b = body.index(key, a + 1)
body = body[:b] + """\t   if (H->onPole) H->onPole(n, m, 24 * prd, la_tmp < 0 ? la_tmp + 360 : la_tmp, be_tmp, dev_new);
\t   if (dev_new < H->bestDev)
\t   {
\t       H->bestDev = dev_new; H->bestPeriod = 24 * prd; H->bestLambda = la_tmp < 0 ? la_tmp + 360 : la_tmp; H->bestBeta = be_tmp;
\t       /* Area[] holds the last *trial* step of Levenberg-Marquardt, which may have been
\t          rejected: recompute it from the accepted parameters cg (and restore it afterwards) */
\t       std::vector<double> keep(Area, Area + Numfac + 1);
\t       curv(cg);
\t       H->area.assign(Area + 1, Area + 1 + Numfac);
\t       std::copy(keep.begin(), keep.end(), Area);
\t       H->normal.resize(Numfac);
\t       for (i = 1; i <= Numfac; i++) { H->normal[i-1][0] = Nor[i][1]; H->normal[i-1][1] = Nor[i][2]; H->normal[i-1][2] = Nor[i][3]; }
\t       H->ncoef = Ncoef; H->nphpar = Nphpar;
\t       H->cg.assign(cg + 1, cg + 1 + Ncoef + 5 + Nphpar);
\t       H->conwUsed = conw_r * escl * escl;
\t   }
""" + body[b:]
# end of run: free memory instead of boinc_finish
tail = body[body.index("    out.close();\n"):]
body = body[:body.index("    out.close();\n")] + """    out.close();
    fclose(infile);
    free(str_temp);
    deallocate_matrix((void *) ee, MAX_N_OBS);
    deallocate_matrix((void *) ee0, MAX_N_OBS);
    deallocate_matrix((void *) covar, MAX_N_PAR);
    deallocate_matrix((void *) aalpha, MAX_N_PAR);
    deallocate_matrix((void *) ifp, MAX_N_FAC);
    deallocate_vector((void *) tim);
    deallocate_vector((void *) brightness);
    deallocate_vector((void *) sig);
    deallocate_vector((void *) cg);
    deallocate_vector((void *) cg_first);
    deallocate_vector((void *) t);
    deallocate_vector((void *) f);
    deallocate_vector((void *) at);
    deallocate_vector((void *) af);
    deallocate_vector((void *) ia);
    deallocate_vector((void *) al);
    deallocate_vector((void *) weight_lc);
    return PS_CANCEL ? 1 : 0;
}
"""
header = '''// GENERATED by scripts/gen_ps_core.py from period_search/period_search_BOINC.cpp - do not edit.
// The numerical code is period_search's own; see the generator for the (I/O only) changes.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include "ps_core.h"
#include "u8.h"

extern "C" {
#include "constants.h"
#include "declarations.h"

''' + globals_block + '''
}

#define PS_CANCEL (H->cancel && H->cancel->load())
#define PS_NPOLES (H->singlePole ? 1 : N_POLES)
#define PS_OVERRIDES \\
   if (H->perStart > 0) { per_start = H->perStart; per_end = H->perEnd; } \\
   if (H->conw > 0) conw = H->conw;

namespace {
struct MFILE {  // replaces BOINC's MFILE: rows go to H->onRow
  PsHooks* H = nullptr;
  int printf(const char* fmt, ...) {
    char b[512];
    va_list a;
    va_start(a, fmt);
    int r = vsnprintf(b, sizeof b, fmt, a);
    va_end(a);
    if (H && H->onRow) H->onRow(b);
    return r;
  }
  void close() {}
};
}  // namespace

'''
extra = r"""

/* Gaussian image of a fitted model on a finer triangulation (nrows rows per octant,
   8*nrows^2 facets, at most MAX_N_FAC) - same curvature function exp(sum cg*Y_lm),
   evaluated more densely: used for the smooth model. Uses the period_search globals. */
int ps_gauss_image(const double* cg0, int ncoef, int lmax, int mmax, int nrows,
                   std::vector<double>& area, std::vector<std::array<double, 3>>& normal) {
   int i, j, k, ndir;
   double dth, dph;
   if (8 * nrows * nrows > MAX_N_FAC || lmax > MAX_LM || mmax > MAX_LM) return 1;
   int **ifp = matrix_int(MAX_N_FAC, 4);
   double *t = vector_double(MAX_N_FAC), *f = vector_double(MAX_N_FAC),
          *at = vector_double(MAX_N_FAC), *af = vector_double(MAX_N_FAC),
          *cg = vector_double(MAX_N_PAR);
   for (i = 1; i <= ncoef; i++) cg[i] = cg0[i - 1];
   Lmax = lmax; Mmax = mmax;
   /* same construction of the triangulated sphere as in period_search */
   dth = PI / (2 * nrows);
   k = 1; t[1] = 0; f[1] = 0;
   for (i = 1; i <= nrows; i++) {
      dph = PI / (2 * i);
      for (j = 0; j <= 4 * i - 1; j++) { k++; t[k] = i * dth; f[k] = j * dph; }
   }
   for (i = nrows - 1; i >= 1; i--) {
      dph = PI / (2 * i);
      for (j = 0; j <= 4 * i - 1; j++) { k++; t[k] = PI - i * dth; f[k] = j * dph; }
   }
   ndir = k + 1;
   t[ndir] = PI; f[ndir] = 0;
   Numfac = 8 * nrows * nrows;
   trifac(nrows, ifp);
   areanorm(t, f, ndir, Numfac, ifp, at, af);
   sphfunc(Numfac, at, af);
   curv(cg);
   area.assign(Area + 1, Area + 1 + Numfac);
   normal.resize(Numfac);
   for (i = 1; i <= Numfac; i++) { normal[i-1][0] = Nor[i][1]; normal[i-1][1] = Nor[i][2]; normal[i-1][2] = Nor[i][3]; }
   deallocate_matrix((void *) ifp, MAX_N_FAC);
   deallocate_vector((void *) t); deallocate_vector((void *) f);
   deallocate_vector((void *) at); deallocate_vector((void *) af); deallocate_vector((void *) cg);
   return 0;
}
"""
open(dst, "w").write(header + body + extra + "\n")
print("wrote", dst)
