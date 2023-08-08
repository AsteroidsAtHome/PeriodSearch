#include "declarations.hpp"
#include "boinc_api.h"

int getData(const char* filename, DATA*& data)
{
	//#include "boinc_db.h"
	//#include "sched_util.h"

#define MAX_LINES 100000

	int n, retval, nlines, max_nlines;
	double per, rms, chisq, dark, lambda, beta;
	FILE* outfile;
	char filepath[512];

	//retval = try_fopen(fi.path.c_str(), f, "r");
	retval = boinc_resolve_filename(filename, filepath, sizeof(filepath));
	outfile = boinc_fopen(filepath, "r");
	if (!outfile) {
		fprintf(stderr,
			"Couldn't find input file, resolved name %s.\n",
			filepath
		);
		exit(-1);
	}

	//if (retval) return retval;
	DATA* dp = new DATA;
	max_nlines = MAX_LINES;
	dp->per = (double*)malloc(max_nlines * sizeof(double));
	dp->rms = (double*)malloc(max_nlines * sizeof(double));
	dp->chisq = (double*)malloc(max_nlines * sizeof(double));

	nlines = 0;
	while (feof(outfile) == 0)
	{
		if (nlines >= max_nlines)
		{
			max_nlines += MAX_LINES;
			dp->per = (double*)realloc(dp->per, max_nlines * sizeof(double));
			dp->rms = (double*)realloc(dp->rms, max_nlines * sizeof(double));
			dp->chisq = (double*)realloc(dp->chisq, max_nlines * sizeof(double));
		}
		n = fscanf(outfile, "%lf %lf %lf %lf %lf %lf", &per, &rms, &chisq, &dark, &lambda, &beta);
		if (n != 6 && n != -1) { fclose(outfile); return ERR_XML_PARSE; }
		if (isnan(per) || isnan(rms) || isnan(chisq)) { fclose(outfile); return ERR_XML_PARSE; }

		dp->per[nlines] = per;
		dp->rms[nlines] = rms;
		dp->chisq[nlines] = chisq;
		nlines++;
		//printf ("Výstup1: %lf %lf %lf %lf %lf %lf\n", per[nlines], rms[nlines], chisq[nlines], dark, lambda, beta);
		//printf ("Počet řádků: %d\n", n);
		//printf ("Aktuální řádek: %d\n", nlines);
	}

	dp->nlines = nlines;
	fclose(outfile);

	data = (DATA*)dp;
	return 0;
}

int compare_results(DATA* _data1, DATA* _data2, bool& match) {

	int i;
	double tol_per = 0.1, tol_rms = 0.1, tol_chisq = 0.5;

	DATA* data1 = _data1;
	DATA* data2 = _data2;
	match = true;

	if ((data1->nlines == 0) || (data2->nlines == 0) || (data1->nlines != data2->nlines))
	{
		match = false;
		return 0;
	}

	for (i = 0; i < data1->nlines; i++)
	{
		//	if (fabs(data1->per[i] - data2->per[i]) > tol_per) match = false;
		//        if (fabs(data1->rms[i] - data2->rms[i]) > tol_rms) match = false;
		//        if (fabs(data1->chisq[i] - data2->chisq[i]) > tol_chisq) match = false;
		if (fabs((data1->per[i] - data2->per[i]) / (data1->per[i] + data2->per[i])) / 2 > tol_per)
		{
			match = false;
			break;
		}
		if (fabs((data1->rms[i] - data2->rms[i]) / (data1->rms[i] + data2->rms[i])) / 2 > tol_rms)
		{
			match = false;
			break;
		}
		if (fabs((data1->chisq[i] - data2->chisq[i]) / (data1->chisq[i] + data2->chisq[i])) / 2 > tol_chisq)
		{
			match = false;
			break;
		}
		//printf ("Výstup: %lf %lf %lf \n", data1->per[i], data1->rms[i], data1->chisq[i]);
	}
	return 0;
}

int cleanup_result(DATA* data)
{
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

void CompareResult(const char* output_filename)
{
	DATA* result;
	DATA* comparrer;
	const char* cmpFilename = "period_search_out_cuda_correct";
	bool match = false;

	getData(output_filename, result);
	getData(cmpFilename, comparrer);
	compare_results(result, comparrer, match);
	if (match)
	{
		std::cerr << std::endl << "Result file matches the comparrer file! All good!" << std::endl << std::endl;
	}
	else
	{
		std::cerr << std::endl << "Result file does not matches the comparrer file! Check the differences!" << std::endl << std::endl;
	}

	cleanup_result(result);
	cleanup_result(comparrer);
}