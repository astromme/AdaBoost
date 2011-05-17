#ifndef _FEATURE_VECTOR_H
#define _FEATURE_VECTOR_H

#include <vector>

static const int POS = 1;
static const int NEG = -1;

/**********************************
 * Class: FeatureVector
 * --------------------
 * Feature vectors have a value, weight, and vector of features (floats) that
 * can be compared with other feature vectors
 */
class FeatureVector {
	public:
		FeatureVector(const std::vector<float> &in_vec, 
				int in_val); // constructor

		FeatureVector(const FeatureVector &other);

		unsigned int size() const;
		int val() const;
		float weight() const;
		float at(unsigned int i) const;

		void setWeight(float weight);

		void printFeature() const;

	private:
		std::vector <float> fvec; // feature vector
		int value; // value (POS or NEG)
		float wt; // weight
};

#endif
