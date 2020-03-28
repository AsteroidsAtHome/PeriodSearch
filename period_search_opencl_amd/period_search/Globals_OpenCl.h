#pragma once

struct freq_result
{
	int isReported;
	double dark_best, per_best, dev_best, la_best, be_best;
};

extern freq_result* CUDA_FR;