#include "stdafx.h"
#include "CppUnitTest.h"
#include <memory>
#include <math.h>
#include <codecvt>
#include "arrangers.hpp"
#include "../period_search/declarations.h"
#include "../period_search/constants.h"
#include "../period_search/globals.h"
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
#include "../period_search/dot_product.c"
#include "../period_search/gauss_errc.c"
#include "../period_search/blmatrix.c"
#include "../period_search/curv.c"
#include "../period_search/matrix.c"
#include "../period_search/phasec.c"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

int Lmax, Mmax, Niter, Lastcall, Ncoef, Numfac, Lcurves, Nphpar, Deallocate, n_iter;
int *indx, *_indx, **ifp, **_ifp, *ia, *_ia;
int Lpoints[MAX_LC + 1], Inrel[MAX_LC + 1]; 
double Alamda, Alamda_incr, Alamda_start, Ochisq, Chisq, _Chisq, Phi_0, Scale; 
double Blmat[4][4], Dblm[3][4][4];
double *t, *f, *at, *af, *_at, *_af, *sig, *cg_first, *_cg_first, *d, *_d, **fitmat, **_fitmat, *fitvec, *_fitvec; 
double **ee, **_ee, **ee0, **_ee0, *tim, *_tim, *brightness, *_brightness, *cg, *_cg, **covar, **_covar, **aalpha, **_aalpha;
double Sclnw[MAX_LC + 1];
double Yout[MAX_N_OBS + 1];
double Weight[MAX_N_OBS + 1]; 

double Fc[MAX_N_FAC + 1][MAX_LM + 1];
double Fs[MAX_N_FAC + 1][MAX_LM + 1];
double Tc[MAX_N_FAC + 1][MAX_LM + 1];
double Ts[MAX_N_FAC + 1][MAX_LM + 1];
double Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
double _dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
double Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
double _pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];

__declspec(align(32)) double Nor[3][MAX_N_FAC + 4], Area[MAX_N_FAC + 4], Darea[MAX_N_FAC + 4], Dg[MAX_N_FAC + 8][MAX_N_PAR + 4];

namespace UnitTest_avx_win
{

    TEST_CLASS(UnitTest1)
    {
    public:

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
            covar = aligned_matrix_double(MAX_N_PAR, MAX_N_PAR);
            _covar = aligned_matrix_double(MAX_N_PAR, MAX_N_PAR);
            aalpha = aligned_matrix_double(MAX_N_PAR, MAX_N_PAR + 4);
            _aalpha = aligned_matrix_double(MAX_N_PAR, MAX_N_PAR + 4);


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
            aligned_deallocate_matrix_double(covar, MAX_N_PAR);
            aligned_deallocate_matrix_double(_covar, MAX_N_PAR);
            aligned_deallocate_matrix_double(aalpha, MAX_N_PAR);
            aligned_deallocate_matrix_double(_aalpha, MAX_N_PAR);

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

            ////std::ostringstream oss;
            //std::string str;
            //for (int k = 1; k <= 3; k++) {
            //    for (int i = 1; i <= ntri; i++) {
            //        {
            //            //oss << _ifp[k][l] << ", ";
            //            char c[20];
            //            sprintf_s(c, "%d, ", _ifp[i][k]);
            //            str = str + c;
            //        }
            //    }
            //    str = str + "\n";
            //}
            ////std::string str = oss.str();
            //char *msg = &str[0u];; // new char[str.length() + 1];
            //Logger::WriteMessage(msg);

            // Act
            //makes indices to triangle vertices
            trifac(nrows, ifp);

            //str = "";
            //for (int k = 1; k <= 3; k++) {
            //    for (int i = 1; i <= ntri; i++) {
            //        {
            //            //oss << _ifp[k][l] << ", ";
            //            char c[20];
            //            sprintf_s(c, "%d, ", ifp[i][k]);
            //            str = str + c;
            //        }
            //    }
            //    str = str + "\n";
            //}
            //msg = &str[0u];; // new char[str.length() + 1];
            //Logger::WriteMessage(msg);

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
                char c[500];
                sprintf_s(c, "Elements ifp[%d][%d] = %d and _ifp[%d][%d] = %d are NOT EQUAL!", o, p, ifp[o][p], o, p, _ifp[o][p]);
                char *msg = &c[0u];
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
                char c[500];
                sprintf_s(c, "Element _at[%d]<%.30f>; element at[%d]<%.30f>", i, _at[i], i, at[i]);
                std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
                std::wstring wide = converter.from_bytes(c);
                const wchar_t* wat = wide.c_str();
                sprintf_s(c, "Element _af[%d]<%.30f>; element af[%d]<%.30f>", i, _af[i], i, af[i]);
                wide = converter.from_bytes(c);
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

            get_dsph();

            // Assert
            for(int k = 1; k <= kMax; k++)
            {
                for(int i = 1; i < ndir; i++)
                {
                    char c[500];
                    sprintf_s(c, "Element _dsph[%d][%d]<%.30f>; element Dsph[%d][%d]<%.30f>", i,k, _dsph[i][k], i,k, Dsph[i][k]);
                    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
                    std::wstring wide = converter.from_bytes(c);
                    const wchar_t* wat = wide.c_str();
                    Assert::IsTrue(_dsph[i][k] - Dsph[i][k] < DBL_EPSILON, wat);
                }
            }
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

            for(i = 0; i <= ncoef; i++)
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
                char c[500];
                sprintf_s(c, "Element _cg_first[%d]<%.30f>; element cg_first[%d]<%.30f>", i, _cg_first[i], i, cg_first[i]);
                std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
                std::wstring wide = converter.from_bytes(c);
                const wchar_t* wat = wide.c_str();

                Assert::IsTrue(_cg_first[i] - cg_first[i] < DBL_EPSILON, wat);
            }
        }

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

        //    a = a0; b = b0; c_axis = c0;
        //    double ave = 0.382008772881355651573898057904;
        //    double al0 = 0.069555451868686835048549710336;
        //    int ial0 = 1;
        //    Lpoints[1] = 118;

        //    for (int j = 1; j <= Lpoints[1]; j++)
        //    {
        //        k2++;
        //        sig[k2] = ave;
        //    }

        //    /* Initial shape guess */
        //    double rfit = sqrt(2 * sig[ial0] / (0.5 * PI * (1 + cos(al0))));
        //    double escl = rfit / sqrt((a * b + b * c_axis + a * c_axis) / 3);
        //    if (onlyrel == 0)
        //        escl *= 0.8;
        //    a = a * escl;
        //    b = b * escl;
        //    c_axis = c_axis * escl;
        //    Niter = 0;

        //    get_t(t);
        //    get_f(f);

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

        //    get_ee(ndata, ee);
        //    get_ee0(ndata, ee0);
        //    get_tim(ndata, tim);
        //    
        //    // makes indices to triangle vertices
        //    trifac(nrows, ifp);

        //    // areas and normals of the triangulated Gaussian image sphere
        //    areanorm(t, f, ndir, Numfac, ifp, at, af);

        //    // Precompute some function values at each normal direction
        //    sphfunc(Numfac, at, af);

        //    //get_dsph();

        //    ellfit(cg_first, a, b, c_axis, Numfac, Ncoef, at, af);

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

        //    mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha);

        //    // Assert
        //    Assert::AreEqual(_Chisq, Chisq);

        //    for (int j = 0; j <= n; j++) {
        //        for (int i = 0; i <= n; i++)
        //        {
        //            Assert::IsTrue(_covar[i][j] - covar[i][j] < DBL_EPSILON);
        //        }
        //    }

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
            // From cpu app : 17.20794966  0.168685  3.357645  0.8  360   22
            const double per_1 = 17.20899774;
            const double per_2 = 17.20794966;
            const double rms_1 = 0.167425;
            const double rms_2 = 0.168685;
            const double chisq_1 = 3.307656;
            const double chisq_2 = 3.357645;

            // Act
            double period_result = fabs((per_1 - per_2) / (per_1 + per_2)) / 2;
            if ( period_result > tol_per)
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