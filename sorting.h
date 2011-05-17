#ifndef _SORTING_H
#define _SORTING_H

#include <vector>
#include "FeatureVector.h"

// object to help us compare two feature vectors
class idx_cmp {
	public:
		idx_cmp(unsigned int idx): m_idx(idx) {};
		bool operator() ( FeatureVector *a, FeatureVector *b);
	private:
		unsigned int m_idx;
};

#endif
