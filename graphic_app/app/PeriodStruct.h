#pragma once
using namespace std;

namespace ps {
	struct PeriodStruct
	{
		vector<double> period;
		vector<double> rms;
		vector<double> chi;
		vector<float> dark;
		vector<int> alpha;
		vector<int> beta;
	};

	inline void ResizeVectors(const int size, PeriodStruct& periodStruct)
	{
		periodStruct.period.resize(size);
		periodStruct.rms.resize(size);
		periodStruct.chi.resize(size);
		periodStruct.dark.resize(size);
		periodStruct.alpha.resize(size);
		periodStruct.beta.resize(size);
	}}
