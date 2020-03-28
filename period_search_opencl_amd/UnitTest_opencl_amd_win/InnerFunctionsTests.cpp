#include "stdafx.h"
#include "CppUnitTest.h"
#include <memory>
#include <math.h>
#include <codecvt>
#include "arrangers.hpp"
#include "declarations.h"
#include "constants.h"
#include "globals.h"
#include <CL/cl.hpp>
//#include "../period_search/arrayHelpers.hpp"
#include "../period_search/memory.c"
#include "../period_search/areanorm.c"
#include "../period_search/trifac.c"
#include "../period_search/sphfunc.c"
#include "../period_search/ludcmp.c"
#include "../period_search/lubksb.c"
#include "../period_search/ellfit.c"
#include "../period_search/mrqmin.c"
#include "../period_search/mrqcof.c"
#include "../period_search/bright.c"
#include "../period_search/conv.c"
#include "../period_search/host_dot_product.c"
#include "../period_search/gauss_errc.c"
#include "../period_search/blmatrix.c"
#include "../period_search/curv.c"
#include "../period_search/matrix.c"
#include "../period_search/phasec.c"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

int Lmax, Mmax, Niter, Lastcall, Ncoef, Numfac, Lcurves, Nphpar, Deallocate, n_iter;
int *indx, *_indx, **ifp, **_ifp, *ia, *_ia;
int Lpoints[MAX_LC + 1], Inrel[MAX_LC + 1];
double Alamda, Alamda_incr, Alamda_start, Ochisq, Chisq, _Chisq, Phi_0, Scale, _scale;
double Blmat[4][4], _blmat[4][4], Dblm[3][4][4], _dblm[3][4][4];
double Xx1[4], Xx2[4], _xx1[4], _xx2[4];
double tmat[4][4], dtm[4][4][4], _tmat[4][4], _dtm[4][4][4];
double php[N_PHOT_PAR + 1], dphp[N_PHOT_PAR + 1], _dphp[N_PHOT_PAR + 1];
double *t, *f, *at, *af, *_at, *_af, *sig, *cg_first, *_cg_first, *d, *_d, **fitmat, **_fitmat, *fitvec, *_fitvec;
double **ee, **_ee, **ee0, **_ee0, *tim, *_tim, *brightness, *_brightness, *cg, *_cg, **covar, **_covar, **aalpha, **_aalpha;
double Sclnw[MAX_LC + 1];
double Yout[MAX_N_OBS + 1];
double Weight[MAX_N_OBS + 1];
static double *atry, *beta, *da; //beta, da are zero indexed

double Fc[MAX_N_FAC + 1][MAX_LM + 1];
double Fs[MAX_N_FAC + 1][MAX_LM + 1];
double Tc[MAX_N_FAC + 1][MAX_LM + 1];
double Ts[MAX_N_FAC + 1][MAX_LM + 1];
double Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
double _dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
double Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
double _pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];

double Area[MAX_N_FAC + 1], Darea[MAX_N_FAC + 1], 
//Nor[MAX_N_FAC + 1][3], 
Nor[3][MAX_N_FAC + 1],
Dg[MAX_N_FAC + 1][MAX_N_PAR + 1],
_area[MAX_N_FAC + 1], _darea[MAX_N_FAC + 1], _nor[3][MAX_N_FAC + 1], _dg[MAX_N_FAC + 1][MAX_N_PAR + 1],
Dyda[MAX_N_PAR + 1], _dyda[MAX_N_PAR + 1];

// Dyda[MAX_N_PAR + 8], _dyda[MAX_N_PAR + 8];
// __declspec(align(32)) double Nor[3][MAX_N_FAC + 4], Area[MAX_N_FAC + 4], Darea[MAX_N_FAC + 4], Dg[MAX_N_FAC + 8][MAX_N_PAR + 4],
// _nor[3][MAX_N_FAC + 4], _area[MAX_N_FAC + 4], _darea[MAX_N_FAC + 4], _dg[MAX_N_FAC + 8][MAX_N_PAR + 4],
// Dyda[MAX_N_PAR + 8], _dyda[MAX_N_PAR + 8]; //Zero indexed for aligned memory access

namespace UnitTest_avx_win
{

    TEST_CLASS(UnitTest1)
    {
    public:

        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        char cmsg[500];
        const wchar_t *wmsg;

        TEST_CLASS_INITIALIZE(Init)
        {
            t = vector_double(MAX_N_FAC);
            f = vector_double(MAX_N_FAC);
            at = vector_double(MAX_N_FAC);
            af = vector_double(MAX_N_FAC);
            _at = vector_double(MAX_N_FAC);
            _af = vector_double(MAX_N_FAC);
            sig = vector_double(MAX_N_OBS);
            tim = vector_double(MAX_N_OBS);
            _tim = vector_double(MAX_N_OBS);
            cg_first = vector_double(MAX_N_PAR);
            _cg_first = vector_double(MAX_N_PAR);
            brightness = vector_double(MAX_N_OBS);
            _brightness = vector_double(MAX_N_OBS);
            ifp = matrix_int(MAX_N_FAC, 4);
            _ifp = matrix_int(MAX_N_FAC, 4);
            ee = matrix_double(MAX_N_OBS, 3);
            _ee = matrix_double(MAX_N_OBS, 3);
            ee0 = matrix_double(MAX_N_OBS, 3);
            _ee0 = matrix_double(MAX_N_OBS, 3);
            cg = vector_double(MAX_N_PAR);
            _cg = vector_double(MAX_N_PAR);
            ia = vector_int(MAX_N_PAR);
            _ia = vector_int(MAX_N_PAR);
            covar = matrix_double(MAX_N_PAR, MAX_N_PAR);
            _covar = matrix_double(MAX_N_PAR, MAX_N_PAR);
            aalpha = matrix_double(MAX_N_PAR, MAX_N_PAR + 4);
            _aalpha = matrix_double(MAX_N_PAR, MAX_N_PAR + 4);
            beta = vector_double(24);
            da = vector_double(24);

            int ncoef = 16;
            fitvec = vector_double(ncoef);
            _fitvec = vector_double(ncoef);
            fitmat = matrix_double(ncoef, ncoef);
            _fitmat = matrix_double(ncoef, ncoef);
            indx = vector_int(ncoef);
            _indx = vector_int(ncoef);
            d = vector_double(1);
            _d = vector_double(1);

        }

        TEST_CLASS_CLEANUP(Clean)
        {
            deallocate_matrix_int(ifp, MAX_N_FAC);
            deallocate_matrix_int(_ifp, MAX_N_FAC);
            deallocate_vector(t);
            deallocate_vector(f);
            deallocate_vector(at);
            deallocate_vector(af);
            deallocate_vector(_at);
            deallocate_vector(_af);
            deallocate_vector(sig);
            deallocate_vector(cg);
            deallocate_vector(_cg);
            deallocate_vector(cg_first);
            deallocate_vector(_cg_first);
            deallocate_vector(tim);
            deallocate_vector(_tim);
            deallocate_vector(brightness);
            deallocate_vector(_brightness);
            deallocate_matrix_double(ee, MAX_N_OBS);
            deallocate_matrix_double(_ee, MAX_N_OBS);
            deallocate_matrix_double(ee0, MAX_N_OBS);
            deallocate_matrix_double(_ee0, MAX_N_OBS);
            deallocate_vector(ia);
            deallocate_vector(_ia);
            deallocate_matrix_double(covar, MAX_N_PAR);
            deallocate_matrix_double(_covar, MAX_N_PAR);
            deallocate_matrix_double(aalpha, MAX_N_PAR);
            deallocate_matrix_double(_aalpha, MAX_N_PAR);
            deallocate_vector((void *)beta);
            deallocate_vector((void *)da);

            int ncoef = 16;
            deallocate_vector(static_cast<void *>(indx));
            deallocate_vector(static_cast<void *>(_indx));
            deallocate_vector(static_cast<void *>(d));
            deallocate_vector(static_cast<void *>(_d));
            deallocate_matrix_double(fitmat, ncoef);
            deallocate_matrix_double(_fitmat, ncoef);
            deallocate_vector(static_cast<void *>(fitvec));
            deallocate_vector(static_cast<void *>(_fitvec));
        }

        TEST_METHOD(TestTrifac)
        {
            // Arange
            const int ntri = 288;
            int nrows = 6;          // nr.of triangulation rows per octant

            get_ifp(_ifp);
            
            // Act
            //makes indices to triangle vertices
            trifac(nrows, ifp);
            
            // Assert
            int o = 0, p = 0;
            try {
                for (p = 1; p <= 3; p++)
                    for (o = 1; o <= ntri; o++)
                        if (ifp[o][p] == _ifp[o][p])
                        {
                            continue;
                        }
                        else
                        {
                            throw std::exception();
                        }
                Assert::IsTrue(1 == 1);
            }
            catch (std::exception &e)
            {
                /*char c[500];*/
                sprintf_s(cmsg, "Elements ifp[%d][%d] = %d and _ifp[%d][%d] = %d are NOT EQUAL!", o, p, ifp[o][p], o, p, _ifp[o][p]);
                char *msg = &cmsg[0u];
                Logger::WriteMessage(msg);
                Assert::Fail();
            }

        }

        TEST_METHOD(TestAreanorm)
        {
            // Arrange
            const int ndir = 146;
            int nrows = 6;          // nr.of triangulation rows per octant
            Numfac = 8 * nrows * nrows;

            get_t(t);
            get_f(f);
            get_at(_at);
            get_af(_af);

            // Act
            // makes indices to triangle vertices
            trifac(nrows, ifp);

            // areas and normals of the triangulated Gaussian image sphere
            areanorm(t, f, ndir, Numfac, ifp, at, af);

            for (int i = 1; i <= Numfac; i++)
            {

                sprintf_s(cmsg, "Element _at[%d]<%.30f>; element at[%d]<%.30f>", i, _at[i], i, at[i]);
                std::wstring wide = converter.from_bytes(cmsg);
                const wchar_t* wat = wide.c_str();
                sprintf_s(cmsg, "Element _af[%d]<%.30f>; element af[%d]<%.30f>", i, _af[i], i, af[i]);
                wide = converter.from_bytes(cmsg);
                const wchar_t* waf = wide.c_str();

                Assert::IsTrue(_at[i] - at[i] < DBL_EPSILON, wat);
                Assert::IsTrue(_af[i] - af[i] < DBL_EPSILON, waf);

                /*double x = ((_at[i] - at[i]) / (_at[i] + at[i])) / 2.0;
                double xx = _at[i] - at[i];
                char cc[40];
                sprintf_s(cc, "(%d)\t%.30f\n", i, xx);
                Logger::WriteMessage(cc);*/

                //Assert::IsTrue(x < 0.1, w);
            }
        }

        TEST_METHOD(TestSphfunc)
        {
            // Arrange
            const int ndir = 146;
            int nrows = 6;          // nr.of triangulation rows per octant
            Numfac = 8 * nrows * nrows;
            Lmax = Mmax = 3;
            int kMax = 16;

            get_t(t);
            get_f(f);

            // Act
            // makes indices to triangle vertices
            trifac(nrows, ifp);

            // areas and normals of the triangulated Gaussian image sphere
            areanorm(t, f, ndir, Numfac, ifp, at, af);

            // Precompute some function values at each normal direction
            sphfunc(Numfac, at, af);

            get_pleg();
            get_dsph();

            // Assert
            // Pleg[][]
            for (int k = 1; k <= kMax; k++)
            {
                for (int j = 0; j <= kMax; j++) {
                    for (int i = 1; i < ndir; i++)
                    {
                        sprintf_s(cmsg, "Element _pleg[%d][%d][%d]<%.30f>; element Pleg[%d][%d][%d]<%.30f>", i, j, k, _pleg[i][j][k], i, j, k, Pleg[i][j][k]);
                        std::wstring wide = converter.from_bytes(cmsg);
                        wmsg = wide.c_str();
                        Assert::IsTrue(_pleg[i][j][k] - Pleg[i][j][k] < DBL_EPSILON, wmsg);
                    }
                }
            }

            // Dsph[][]
            for (int k = 1; k <= kMax; k++)
            {
                for (int i = 1; i < ndir; i++)
                {
                    sprintf_s(cmsg, "Element _dsph[%d][%d]<%.30f>; element Dsph[%d][%d]<%.30f>", i, k, _dsph[i][k], i, k, Dsph[i][k]);
                    std::wstring wide = converter.from_bytes(cmsg);
                    wmsg = wide.c_str();
                    Assert::IsTrue(_dsph[i][k] - Dsph[i][k] < DBL_EPSILON, wmsg);
                }
            }

            // Fc[][]

            // Fs[][]
        }

        TEST_METHOD(TestLudcmp)
        {
            // Arrange
            int ncoef = 16;

            get_fitmat_a(fitmat);
            get_fitmat_b(_fitmat);
            get_indx(_indx);
            *_d = 1.0;

            // Act

            ludcmp(fitmat, ncoef, indx, d);

            // Assert
            Assert::IsTrue(*_d == *d);

            int i;
            for (int j = 0; j <= ncoef; j++) {
                for (i = 0; i <= ncoef; i++)
                {
                    Assert::IsTrue(_fitmat[i][j] - fitmat[i][j] < DBL_EPSILON);
                }
            }

            for (i = 0; i <= ncoef; i++)
            {
                Assert::AreEqual(_indx[i], indx[i]);
            }
        }

        TEST_METHOD(TestLubksb)
        {
            // Arrange
            int ncoef = 16;

            get_fitmat_a(fitmat);
            get_fitmat_b(_fitmat);
            get_fitvec_a(fitvec);
            get_fitvec_b(_fitvec);
            get_indx(_indx);
            *_d = 1.0;

            // Act
            ludcmp(fitmat, ncoef, indx, d);
            lubksb(_fitmat, ncoef, indx, fitvec);

            // Assert
            for (int i = 0; i <= ncoef; i++)
            {
                Assert::IsTrue(_fitvec[i] - fitvec[i] < DBL_EPSILON);
            }
        }

        TEST_METHOD(TestEllfit)
        {
            // Arrange

            int onlyrel = 0;
            int k2 = 0;
            Ncoef = 16;
            const int ndir = 146;
            int nrows = 6;          // nr.of triangulation rows per octant
            Numfac = 8 * nrows * nrows;
            //int Lpoints[MAX_LC + 1];
            double a0 = 1.05, b0 = 1.00, c0 = 0.95, a, b, c_axis;

            a = a0; b = b0; c_axis = c0;
            double ave = 0.382008772881355651573898057904;
            double al0 = 0.069555451868686835048549710336;
            int ial0 = 1;
            Lpoints[1] = 118;
            Alamda = -1;
            Alamda_incr = 5.0;
            Alamda_start = 0.1;
            Lmax = 3; //degree and order of the Laplace series
            Mmax = 3;

            for (int j = 1; j <= Lpoints[1]; j++)
            {
                k2++;
                sig[k2] = ave;
            }

            /* Initial shape guess */
            double rfit = sqrt(2 * sig[ial0] / (0.5 * PI * (1 + cos(al0))));
            double escl = rfit / sqrt((a * b + b * c_axis + a * c_axis) / 3);
            if (onlyrel == 0)
                escl *= 0.8;
            a = a * escl;
            b = b * escl;
            c_axis = c_axis * escl;

            get_t(t);
            get_f(f);
            get_cg_first(Ncoef, _cg_first);

            // Act
            // makes indices to triangle vertices
            trifac(nrows, ifp);

            // areas and normals of the triangulated Gaussian image sphere
            areanorm(t, f, ndir, Numfac, ifp, at, af);

            // Precompute some function values at each normal direction
            sphfunc(Numfac, at, af);

            //get_dsph();

            ellfit(cg_first, a, b, c_axis, Numfac, Ncoef, at, af);

            //Assert
            for (int i = 1; i <= Ncoef; i++)
            {
                sprintf_s(cmsg, "Element _cg_first[%d]<%.30f>; element cg_first[%d]<%.30f>", i, _cg_first[i], i, cg_first[i]);
                std::wstring wide = converter.from_bytes(cmsg);
                wmsg = wide.c_str();

                Assert::IsTrue(_cg_first[i] - cg_first[i] < DBL_EPSILON, wmsg);
            }
        }

        TEST_METHOD(TestCurv)
        {
            // Arrange
            int onlyrel = 0;
            int k2 = 0;
            Ncoef = 16;
            const int ndir = 288;
            int nrows = 6;          // nr.of triangulation rows per octant
            Numfac = 8 * nrows * nrows;
            //int Lpoints[MAX_LC + 1];
            double a0 = 1.05, b0 = 1.00, c0 = 0.95, a, b, c_axis;

            a = a0; b = b0; c_axis = c0;
            double ave = 0.382008772881355651573898057904;
            double al0 = 0.069555451868686835048549710336;
            int ial0 = 1;
            Lpoints[1] = 118;
            Alamda = -1;
            Alamda_incr = 5.0;
            Alamda_start = 0.1;
            Lmax = Mmax = 3; //degree and order of the Laplace series
            double par[4] = { 0, 0.5, 0.1, -0.5 };
            int ia_par[4] = { 0, 1, 1, 1 };
            double cl = 0.1;
            int ia_cl = 0;

            for (int j = 1; j <= Lpoints[1]; j++)
            {
                k2++;
                sig[k2] = ave;
            }

            /* Initial shape guess */
            double rfit = sqrt(2 * sig[ial0] / (0.5 * PI * (1 + cos(al0))));
            double escl = rfit / sqrt((a * b + b * c_axis + a * c_axis) / 3);
            if (onlyrel == 0)
                escl *= 0.8;
            a = a * escl;
            b = b * escl;
            c_axis = c_axis * escl;

            get_t(t);
            get_f(f);
            get_cg_first(Ncoef, _cg_first);

            // Act
            /* makes indices to triangle vertices */
            trifac(nrows, ifp);
            /* areas and normals of the triangulated Gaussian image sphere */
            areanorm(t, f, ndir, Numfac, ifp, at, af);
            /* Precompute some function values at each normal direction*/
            sphfunc(Numfac, at, af);
            ellfit(cg_first, a, b, c_axis, Numfac, Ncoef, at, af);

            // 17.208208 0.5 100 1 period_start period_step period_end fixed/free
            double per_start = 17.208208;
            double per_end = 100;
            double per_step_coef = 0.5;

            double freq_start = 1 / per_start;
            double freq_end = 1 / per_end;
            double jd_max = 2455920.693431000225245952606201171875;
            double jd_min = 2450746.833715000189840793609619140625;
            double freq_step = 0.5 / (jd_max - jd_min) / 24 * per_step_coef;
            int n = 1;
            double freq = freq_start - (n - 1) * freq_step;
            double prd = 1 / freq;

            /* starts from the initial ellipsoid */
            for (int i = 1; i <= Ncoef; i++)
                cg[i] = cg_first[i];

            cg[Ncoef + 1] = 0.0; // beta_pole[m];
            cg[Ncoef + 2] = 0.0; // lambda_pole[m];

            /* The formulas use beta measured from the pole */
            cg[Ncoef + 1] = 90 - cg[Ncoef + 1];
            /* conversion of lambda, beta to radians */
            cg[Ncoef + 1] = DEG2RAD * cg[Ncoef + 1];
            cg[Ncoef + 2] = DEG2RAD * cg[Ncoef + 2];

            /* Use omega instead of period */
            cg[Ncoef + 3] = 24 * 2 * PI / prd;

            for (int i = 1; i <= Nphpar; i++)
            {
                cg[Ncoef + 3 + i] = par[i];
                ia[Ncoef + 3 + i - 1] = ia_par[i];
            }
            /* Lommel-Seeliger part */
            cg[Ncoef + 3 + Nphpar + 2] = 1;
            /* Use logarithmic formulation for Lambert to keep it positive */
            cg[Ncoef + 3 + Nphpar + 1] = log(cl);

            get_dg();   // prepare _dg[][]
            get_area(); // prepare _area[]

            curv(cg);

            // Assert
            // Area[]
            for (int p = 0; p <= Numfac; p++)
            {
                Assert::IsTrue(_area[p] - Area[p] < DBL_EPSILON);
            }

            // Dg[][]
            for (int q = 0; q <= Ncoef; q++)
            {
                for (int p = 0; p <= Numfac; p++)
                {
                    sprintf_s(cmsg, "Element _dg[%d][%d]<%.30f>; element Dg[%d][%d]<%.30f>", p, q, _dg[p][q], p, q, Dg[p][q]);
                    std::wstring wide = converter.from_bytes(cmsg);
                    wmsg = wide.c_str();
                    Assert::IsTrue(_dg[p][q] - Dg[p][q] < DBL_EPSILON, wmsg);
                }
            }
        }

        TEST_METHOD(TestBlmatrix)
        {
            // Arrange
            int onlyrel = 0;
            int k2 = 0;
            Ncoef = 16;
            const int ndir = 288;
            int nrows = 6;          // nr.of triangulation rows per octant
            Numfac = 8 * nrows * nrows; //288
            //int Lpoints[MAX_LC + 1];
            double a0 = 1.05, b0 = 1.00, c0 = 0.95, a, b, c_axis;

            a = a0; b = b0; c_axis = c0;
            double ave = 0.382008772881355651573898057904;
            double al0 = 0.069555451868686835048549710336;
            int ial0 = 1;
            Lpoints[1] = 118;
            Alamda = -1;
            Alamda_incr = 5.0;
            Alamda_start = 0.1;
            Lmax = Mmax = 3; //degree and order of the Laplace series
            double par[4] = { 0, 0.5, 0.1, -0.5 };
            int ia_par[4] = { 0, 1, 1, 1 };
            double cl = 0.1;
            int ia_cl = 0;

            for (int j = 1; j <= Lpoints[1]; j++)
            {
                k2++;
                sig[k2] = ave;
            }

            // Initial shape guess
            double rfit = sqrt(2 * sig[ial0] / (0.5 * PI * (1 + cos(al0))));
            double escl = rfit / sqrt((a * b + b * c_axis + a * c_axis) / 3);
            if (onlyrel == 0)
                escl *= 0.8;
            a = a * escl;
            b = b * escl;
            c_axis = c_axis * escl;

            get_t(t);
            get_f(f);
            get_cg_first(Ncoef, _cg_first);

            // Act
            // makes indices to triangle vertices
            trifac(nrows, ifp);
            // areas and normals of the triangulated Gaussian image sphere 
            areanorm(t, f, ndir, Numfac, ifp, at, af);
            // Precompute some function values at each normal direction
            sphfunc(Numfac, at, af);
            ellfit(cg_first, a, b, c_axis, Numfac, Ncoef, at, af);

            // 17.208208 0.5 100 1 period_start period_step period_end fixed/free
            double per_start = 17.208208;
            double per_end = 100;
            double per_step_coef = 0.5;

            double freq_start = 1 / per_start;
            double freq_end = 1 / per_end;
            double jd_max = 2455920.693431000225245952606201171875;
            double jd_min = 2450746.833715000189840793609619140625;
            double freq_step = 0.5 / (jd_max - jd_min) / 24 * per_step_coef;
            int n = 1;
            double freq = freq_start - (n - 1) * freq_step;
            double prd = 1 / freq;

            // starts from the initial ellipsoid 
            for (int i = 1; i <= Ncoef; i++)
                cg[i] = cg_first[i];

            cg[Ncoef + 1] = 0.0; // beta_pole[m];
            cg[Ncoef + 2] = 0.0; // lambda_pole[m];

            // The formulas use beta measured from the pole 
            cg[Ncoef + 1] = 90 - cg[Ncoef + 1];
            // conversion of lambda, beta to radians
            cg[Ncoef + 1] = DEG2RAD * cg[Ncoef + 1];
            cg[Ncoef + 2] = DEG2RAD * cg[Ncoef + 2];

            // Use omega instead of period
            cg[Ncoef + 3] = 24 * 2 * PI / prd;

            for (int i = 1; i <= Nphpar; i++)
            {
                cg[Ncoef + 3 + i] = par[i];
                ia[Ncoef + 3 + i - 1] = ia_par[i];
            }
            // Lommel-Seeliger part
            cg[Ncoef + 3 + Nphpar + 2] = 1;
            // Use logarithmic formulation for Lambert to keep it positive
            cg[Ncoef + 3 + Nphpar + 1] = log(cl);

            curv(cg);
            get_blmat();
            get_dblm();
            int ma = Ncoef + 5 + Nphpar;

            // Act

            blmatrix(cg[ma - 4 - Nphpar], cg[ma - 3 - Nphpar]);

            // Assert
            // Blmat
            for (int q = 0; q <= 3; q++)
            {
                for (int p = 0; p <= 3; p++)
                {
                    Assert::IsTrue(_blmat[p][q] - Blmat[p][q] < DBL_EPSILON);
                }
            }

            // Dblm
            for (int r = 0; r <= 3; r++)
            {
                for (int q = 0; q <= 3; q++)
                {
                    for (int p = 0; p <= 2; p++)
                    {
                        Assert::IsTrue(_dblm[p][q][r] - Dblm[p][q][r] < DBL_EPSILON);
                    }
                }
            }
        }

        TEST_METHOD(TestPhasec)
        {
            // Arrange
            int ncoef0, i, j, k, incl_count = 0;
            int ndata = 118;
            double cos_alpha, br, cl, cls, alpha;

            get_ee(ndata, ee);
            get_ee0(ndata, ee0);
            Nphpar = 3;
            Ncoef = 16;
            int ma = Ncoef + 5 + Nphpar;
            int ncoef = ma;
            get_cg24(cg);
            double ee_a = cg[ma - 4 - Nphpar];
            double ee0_a = cg[ma - 3 - Nphpar];

            get_xx1();
            get_xx2();
            cos_alpha = host_dot_product(_xx1, _xx2);
            alpha = acos(cos_alpha);
            ncoef0 = ncoef - 2 - Nphpar;
            for (i = 1; i <= Nphpar; i++)
                php[i] = cg[ncoef0 + i];
            _scale = 1.214621163362573641464337015350;
            get_dphp();  // _dphp[]

            // Act
            // computes also Scale
            phasec(dphp, alpha, php);

            // Assert
            for (i = 0; i <= 5; i++)
            {
                Assert::IsTrue(_dphp[i] - dphp[i] < DBL_EPSILON);
            }

            Assert::IsTrue(_scale - Scale < DBL_EPSILON);
        }

        TEST_METHOD(TestMatrix)
        {
            // Arrange
            int ndata = 118;
            Nphpar = 3;
            Ncoef = 16;
            int ma = Ncoef + 5 + Nphpar;
            int ncoef = ma;
            int ncoef0 = ncoef - 2 - Nphpar;
            get_cg24(cg);
            get_tim(ndata, tim);
            int np = 1;
            double t = tim[np];

            get_tmat();     // _tmat[4][4]
            get_dtm();      // _dtm[4][4][4]
            get_blmat_m();  // Blmat[4][4]
            get_dblm_m();   // Dblm[3][4][4]

            // Act
            matrix(cg[ncoef0], t, tmat, dtm);

            // Assert
            // tmat[4][4]
            for (int j = 0; j <= 3; j++)
            {
                for (int i = 0; i <= 3; i++)
                {
                    sprintf_s(cmsg, "Element _tmat[%d][%d]<%.30f>; element tmat[%d][%d]<%.30f>", i, j, _tmat[i][j], i, j, tmat[i][j]);
                    std::wstring wide = converter.from_bytes(cmsg);
                    wmsg = wide.c_str();
                    Assert::IsTrue(_tmat[i][j] - tmat[i][j] < DBL_EPSILON, wmsg);
                }
            }

            // dtm[4][4][4]
            for (int k = 0; k <= 3; k++) {
                for (int j = 0; j <= 3; j++)
                {
                    for (int i = 0; i <= 3; i++)
                    {
                        sprintf_s(cmsg, "Element _dtm[%d][%d][%d]<%.30f>; element dtm[%d][%d][%d]<%.30f>", i, j, k, _dtm[i][j][k], i, j, k, dtm[i][j][k]);
                        std::wstring wide = converter.from_bytes(cmsg);
                        wmsg = wide.c_str();
                        Assert::IsTrue(_dtm[i][j][k] - dtm[i][j][k] < DBL_EPSILON, wmsg);
                    }
                }
            }
        }

        TEST_METHOD(TestBright)
        {
            // Arrange
            int ncoef0, i, j, k, incl_count = 0;
            double ymod, _ymod;

            // Note: Prepare Nor[][] 
            const int ndir = 146;
            int nrows = 6;          // nr.of triangulation rows per octant
            Numfac = 8 * nrows * nrows;

            get_t(t);
            get_f(f);
            /*get_at(_at);
            get_af(_af);*/

            // Act
            // makes indices to triangle vertices
            trifac(nrows, ifp);

            // areas and normals of the triangulated Gaussian image sphere
            areanorm(t, f, ndir, Numfac, ifp, at, af);

            // Nore: Prepare Dg[]
            get_dg();
            for (j = 0; j <= 16; j++) {
                for (i = 0; i <= 288; i++)
                {
                    Dg[i][j] = _dg[i][j];
                }
            }

            int ndata = 118;
            double cos_alpha, br, cl, cls, alpha;

            get_ee(ndata, ee);
            get_ee0(ndata, ee0);
            Nphpar = 3;
            Ncoef = 24;
            int ma = Ncoef + 5 + Nphpar;
            int ncoef = ma;
            get_cg24(cg);
            double ee_a = cg[ma - 4 - Nphpar];
            double ee0_a = cg[ma - 3 - Nphpar];

            get_xx1(true);
            get_xx2(true);
            /*cos_alpha = dot_product(Xx1, Xx2);
            alpha = acos(cos_alpha);
            ncoef0 = ncoef - 2 - Nphpar;
            for (i = 1; i <= Nphpar; i++)
                php[i] = cg[ncoef0 + i];*/
                ////_scale = 1.214621163362573641464337015350;
                //get_dphp();  // dphp[]

                //phasec(dphp, alpha, php);

            get_cg24(cg);
            get_tim(ndata, tim);
            int np = 1;
            double t = tim[np];
            double tolerance = 0.1;

            //get_tmat(true);     // tmat[4][4]
            //get_dtm(true);      // dtm[4][4][4]
            get_blmat_m();      // Blmat[4][4]
            get_dblm_m();       // Dblm[3][4][4]
            get_area(true);

            //matrix(cg[ncoef0], t, tmat, dtm);

            //Scale = 1.214621163362573641464337015350;
            _ymod = 0.34207493045125292;
            k = (MAX_N_PAR + 1) - 1;
            get_dyda();    // _dyda[]

            // Act
            ymod = bright(Xx1, Xx2, t, cg, Dyda, Ncoef);

            // Assert
            sprintf_s(cmsg, "Element _ymod<%.30f>; element ymod<%.30f>", _ymod, ymod);
            std::wstring wide = converter.from_bytes(cmsg);
            wmsg = wide.c_str();
            Assert::IsTrue(_ymod - ymod < DBL_EPSILON, wmsg);

            for (i = 0; i <= k; i++)
            {
                sprintf_s(cmsg, "Element _dyda[%d]<%.30f>; element Dyda[%d]<%.30f>", i, _dyda[i], i, Dyda[i]);
                std::wstring wide = converter.from_bytes(cmsg);
                wmsg = wide.c_str();
                Assert::AreEqual(_dyda[i], Dyda[i], 1e-9, wmsg);
            }

        }

        TEST_METHOD(TestConv)
        {
            int ncoef0, i, j, k, incl_count = 0;
            double ymod, _ymod;

            // Note: Prepare Nor[][] 
            const int ndir = 146;
            int nrows = 6;          // nr.of triangulation rows per octant
            Numfac = 8 * nrows * nrows;  // 288
            Ncoef = 16;

            get_t(t);
            get_f(f);
            
            // Act
            // makes indices to triangle vertices
            trifac(nrows, ifp);

            // areas and normals of the triangulated Gaussian image sphere
            areanorm(t, f, ndir, Numfac, ifp, at, af);

            // Nore: Prepare Dg[]
            get_dg();
            for (j = 0; j <= 16; j++) {
                for (i = 0; i <= 288; i++)
                {
                    Dg[i][j] = _dg[i][j];
                }
            }

            int ndata = 118;
            double cos_alpha, br, cl, cls, alpha;
            k = (MAX_N_PAR + 8) - 1;
            int ma = 24;

            //get_ee(ndata, ee);
            //get_ee0(ndata, ee0);
            //Nphpar = 3;
            //Ncoef = 24;
            //int ma = Ncoef + 5 + Nphpar;
            //int ncoef = ma;
            //get_cg24(cg);
            //double ee_a = cg[ma - 4 - Nphpar];
            //double ee0_a = cg[ma - 3 - Nphpar];

            get_xx1(true);
            get_xx2(true);
            get_cg24(cg);
            get_tim(ndata, tim);
            int np = 1;
            double t = tim[np];
            _ymod = 6.2124784483419404e-17;
            

            get_blmat_m();              // Blmat[4][4]
            get_dblm_m();               // Dblm[3][4][4]
            get_area(true);             // Area[288]
            get_dyda_after_conv();      // _dyda[201]
            get_dyda_before_conv(true); // Dyda[201]
            get_dg_curv(true);          // Dg[288][16]
            get_nor_0(true);            // Nor[0][288]        
            get_darea(true);            // Darea[288]

            // Act
            ymod = conv(1, Dyda, ma);

            // Assert

            Assert::AreEqual(_ymod, ymod, DBL_EPSILON);

            for(i = 0; i < 201; i++)
            {
                sprintf_s(cmsg, "Element _dyda[%d]<%.30f>; element Dyda[%d]<%.30f>", i, _dyda[i], i, Dyda[i]);
                std::wstring wide = converter.from_bytes(cmsg);
                wmsg = wide.c_str();
                Assert::AreEqual(_dyda[i], Dyda[i], DBL_EPSILON, wmsg);
            }
        }

        //TEST_METHOD(TestMrqcof)
        //{
        //    // Arrange
        //    // x1,x2,x3,y,sig,a,ia,ma,alpha
        //    int ndata = 118;
        //    static int mfit, lastone, lastma; /* it is set in the first call*/
        //    Ncoef = 16;
        //    int Nphar = 3;
        //    int ma = Ncoef + 5 + Nphpar;
        //    double _chisq = 7.8099025458229479;
        //    get_ee(ndata, ee);
        //    get_ee0(ndata, ee0);
        //    get_tim(ndata, tim);
        //    get_brightness(brightness);
        //    get_sig(sig);
        //    get_cg24(cg);
        //    get_ia24(ia);
        //    
        //    // Act
        //    mrqcof(ee, ee0, tim, brightness, sig, cg, ia, ma, aalpha, beta, mfit, lastone, lastma);
        //
        //    // Assert
        //    Assert::AreEqual(_chisq, Chisq, DBL_EPSILON);
        //}

        //TEST_METHOD(TestMrqmin)
        //{
        //    // Arrange
        //    int onlyrel = 0;
        //    int k2 = 0;
        //    Ncoef = 16;
        //    const int ndir = 146;
        //    int nrows = 6;          // nr.of triangulation rows per octant
        //    Numfac = 8 * nrows * nrows;
        //    double a0 = 1.05, b0 = 1.00, c0 = 0.95, a, b, c_axis;
        //
        //    a = a0; b = b0; c_axis = c0;
        //    double ave = 0.382008772881355651573898057904;
        //    double al0 = 0.069555451868686835048549710336;
        //    int ial0 = 1;
        //    Lpoints[1] = 118;
        //
        //    for (int j = 1; j <= Lpoints[1]; j++)
        //    {
        //        k2++;
        //        sig[k2] = ave;
        //    }
        //
        //    /* Initial shape guess */
        //    double rfit = sqrt(2 * sig[ial0] / (0.5 * PI * (1 + cos(al0))));
        //    double escl = rfit / sqrt((a * b + b * c_axis + a * c_axis) / 3);
        //    if (onlyrel == 0)
        //        escl *= 0.8;
        //    a = a * escl;
        //    b = b * escl;
        //    c_axis = c_axis * escl;
        //    Niter = 0;
        //
        //    get_t(t);
        //    get_f(f);
        //
        //    int ndata = 118;
        //    int n = 22;
        //    Alamda = -1;
        //    Alamda_incr = 5.0;
        //    Alamda_start = 0.1;
        //    Lmax = 3; //degree and order of the Laplace series
        //    Mmax = 3;
        //    //Ncoef = 16;
        //    Nphpar = 3;
        //    Lcurves = 1;
        //    Phi_0 = 0.0;
        //    _Chisq = 7.8099025458229479;
        //    Lcurves = Lcurves + 1;
        //    Lpoints[Lcurves] = 3;
        //    brightness[1] = 3.452391e-01;
        //
        //
        //    get_ee(ndata, ee);
        //    get_ee0(ndata, ee0);
        //    get_tim(ndata, tim);
        //
        //    // makes indices to triangle vertices
        //    trifac(nrows, ifp);
        //
        //    // areas and normals of the triangulated Gaussian image sphere
        //    areanorm(t, f, ndir, Numfac, ifp, at, af);
        //
        //    // Precompute some function values at each normal direction
        //    sphfunc(Numfac, at, af);
        //
        //    //get_dsph();
        //
        //    ellfit(cg_first, a, b, c_axis, Numfac, Ncoef, at, af);
        //
        //    // Act
        //    get_cg24(cg);
        //    get_ia24(ia);
        //    get_covar(25, _covar);
        //    get_aalpha(25, _aalpha);
        //    get_weight(122, Weight);
        //    get_inrel(10, Inrel);
        //    get_pleg();
        //    get_dsph2();
        //    get_sig(sig);
        //
        //    mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha);
        //
        //    // Assert
        //    Assert::AreEqual(_Chisq, Chisq);
        //
        //    for (int j = 0; j <= n; j++) {
        //        for (int i = 0; i <= n; i++)
        //        {
        //            Assert::IsTrue(_covar[i][j] - covar[i][j] < DBL_EPSILON);
        //        }
        //    }
        //
        //    for (int j = 0; j <= n; j++) {
        //        for (int i = 0; i <= n; i++)
        //        {
        //            Assert::IsTrue(_aalpha[i][j] - aalpha[i][j] < DBL_EPSILON);
        //        }
        //    }
        //}

        TEST_METHOD(TestValidator)
        {
            // Arrange
            bool period_match = true, rms_match = true, chisq_match = true;
            double tol_per = 0.1, tol_rms = 0.1, tol_chisq = 0.5;
            // From avx app : 17.20899774  0.167425  3.307656  1.6   96   59
            // From cpu app : 17.20899838  0.167423  3.307587  1.6   96   59
            const double per_1 = 17.20899774;
            const double per_2 = 17.20899838;
            const double rms_1 = 0.167425;
            const double rms_2 = 0.167423;
            const double chisq_1 = 3.307656;
            const double chisq_2 = 3.307587;

            // Act
            double period_result = fabs((per_1 - per_2) / (per_1 + per_2)) / 2;
            if (period_result > tol_per)
            {
                period_match = false;
            }

            double rms_result = fabs((rms_1 - rms_2) / (rms_1 + rms_2)) / 2;
            if (rms_result > tol_rms)
            {
                rms_match = false;
            }

            double chisq_result = fabs((chisq_1 - chisq_2) / (chisq_1 + chisq_2)) / 2;
            if (chisq_result > tol_chisq)
            {
                chisq_match = false;
            }

            // Assert
            Assert::IsTrue(period_match);
            Assert::IsTrue(rms_match);
            Assert::IsTrue(chisq_match);
        }
    };
}