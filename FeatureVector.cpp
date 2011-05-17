#include <cstdio>
#include "FeatureVector.h"

using namespace std;

//constructor
FeatureVector::FeatureVector(const vector<float> &in_vec, 
		int in_val) : value(in_val), wt(-1){
	// copy vector of floats
	for (unsigned int i=0; i<in_vec.size(); i++)
		fvec.push_back(in_vec[i]);
}

FeatureVector::FeatureVector(const FeatureVector &other) {
  this->fvec = other.fvec;
  this->value = other.value;
  this->wt = other.wt;
}

// returns size of feature vector (useful in addFeature in TrainingData)
unsigned int FeatureVector::size() const { return fvec.size(); }

// more getter methods
int FeatureVector::val() const { return value; }
float FeatureVector::weight() const { return wt; }
float FeatureVector::at(unsigned int i) const { return fvec[i]; }

// sets weight to given value 
void FeatureVector::setWeight(float weight) { wt = weight; }

// prints out all instance fields of a feature
void FeatureVector::printFeature() const {
	for (unsigned int i=0; i<fvec.size(); i++)
		printf("[%d]: %.3f ",i,fvec[i]);
	printf("\n\tval: %d, weight: %f\n",value,wt);
}
