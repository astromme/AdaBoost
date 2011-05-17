#ifndef _TRAINING_DATA_H
#define _TRAINING_DATA_H

#include <vector>
#include <string>
#include "FeatureVector.h"

class TrainingData {
	public:
		TrainingData() : 
			num_dimensions(0) {}; // constructor

		TrainingData(const TrainingData &other);
		TrainingData shuffled() const;

		bool addFeature(FeatureVector in_ftr);
		FeatureVector removeFeatureAt(int location);
		unsigned int size() const;
		unsigned int dimensions() const;

		int val(unsigned int i) const;
		float weight(unsigned int i) const;

		float at(unsigned int idx, unsigned int dimension) const;
		FeatureVector* feature(unsigned int idx) const;

		void setWeight(unsigned int idx, float weight);

		// Output
		void printData();
		void writeData(std::string filename);

	private:
		std::vector <FeatureVector> data; // set of feature vectors
		unsigned int num_dimensions; // size of each feature vector
};

#endif
