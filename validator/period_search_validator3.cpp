#include <string>
#include <vector>
#include <math.h>
#include "error_numbers.h"
#include "boinc_db.h"
#include "sched_util.h"
#include "validate_util.h"
#include "validate_util2.h"
#include "validator.h"

using std::string;
using std::vector;

#define MAX_LINES 500000

struct DATA {
    int nlines;
    double *per;
    double *rms;
    double *chisq;
};
        
//extern int init_result(RESULT const & result, void*& data) {
int init_result(RESULT& result, void*& data) {
    FILE* f;
    OUTPUT_FILE_INFO fi;
    int n, retval, nlines;
    double per, rms, chisq, dark, lambda, beta;
                        
    retval = get_output_file_path(result, fi.path);
    if (retval) return retval;
    retval = try_fopen(fi.path.c_str(), f, "r");
    if (retval) return retval;

    DATA* dp = new DATA;
    dp->per=(double*)malloc(MAX_LINES*sizeof(double));
    dp->rms=(double*)malloc(MAX_LINES*sizeof(double));
    dp->chisq=(double*)malloc(MAX_LINES*sizeof(double));

    nlines = 0;
    while (feof(f) == 0)
    {
        n = fscanf(f, "%lf %lf %lf %lf %lf %lf", &per, &rms, &chisq, &dark, &lambda, &beta);
        if (n != 6 && n != -1) { fclose(f); return ERR_XML_PARSE; }
        dp->per[nlines] = per;
        dp->rms[nlines] = rms;
        dp->chisq[nlines] = chisq;
	nlines++;
//printf ("Výstup1: %lf %lf %lf %lf %lf %lf\n", per[nlines], rms[nlines], chisq[nlines], dark, lambda, beta);
//printf ("Počet řádků: %d\n", n);
//printf ("Aktuální řádek: %d\n", nlines);
    }
    dp->nlines = nlines;
    fclose(f);

    data = (void*) dp;
    return 0;
}
                                                                        
int compare_results(RESULT& r1, void* _data1, RESULT const& r2, void* _data2, bool& match) {
    
    int i;
    double tol_per = 0.1, tol_rms = 0.1, tol_chisq = 0.5;
    
    DATA* data1 = (DATA*)_data1;
    DATA* data2 = (DATA*)_data2;
    match = true;
    
    for (i = 0; i < data1->nlines; i++)
    {
//	if (fabs(data1->per[i] - data2->per[i]) > tol_per) match = false;
//        if (fabs(data1->rms[i] - data2->rms[i]) > tol_rms) match = false;
//        if (fabs(data1->chisq[i] - data2->chisq[i]) > tol_chisq) match = false;
        if (fabs((data1->per[i] - data2->per[i]) / (data1->per[i] + data2->per[i])) / 2 > tol_per) match = false;
        if (fabs((data1->rms[i] - data2->rms[i]) / (data1->rms[i] + data2->rms[i])) / 2 > tol_rms) match = false;
        if (fabs((data1->chisq[i] - data2->chisq[i]) / (data1->chisq[i] + data2->chisq[i])) / 2 > tol_chisq) match = false;
//printf ("Výstup: %lf %lf %lf \n", data1->per[i], data1->rms[i], data1->chisq[i]);
    }        
    return 0;
}
                                                                                                    
int cleanup_result(RESULT const& r, void* data) {
    if (data) 
    {
      DATA* dp = (DATA*)data;
      
      if (dp->per) free(dp->per);
      if (dp->rms) free(dp->rms);
      if (dp->chisq) free(dp->chisq);
      delete dp;
    }
    return 0;
}
                                                                                                            
