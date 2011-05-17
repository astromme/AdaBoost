#include <iostream>
#include <algorithm>
#include "sorting.h"

using namespace std;
	
/****************************************
 * Function: operator()
 * --------------------
 * Comparison operator for two feature vectors. Used in create_feature_views()
 * in AdaBooster.
 */
bool idx_cmp::operator() (FeatureVector *a, FeatureVector *b) {
	return ( a->at(m_idx) < b->at(m_idx) );
}
