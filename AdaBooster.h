#ifndef _ADA_BOOSTER_H
#define _ADA_BOOSTER_H

#include <vector>
#include "FeatureVector.h"
#include "TrainingData.h"
#include "WeakClassifier.h"
#include "StrongClassifier.h"

/*************************
 * Class: AdaBooster
 * -----------------
 * This class uses a set of training data to generate a strong classifier. 
 */
class AdaBooster {
	public:
		AdaBooster(); // constructor
		StrongClassifier getStrongClassifier(const TrainingData &td, unsigned int num_classifiers);
		std::vector< std::vector<double> > getStrongError(TrainingData &td,
				const WeakClassifierList &strong);
		std::vector<int> getFalseIndices();
		void printStrongStats(std::vector< std::vector<double> > strong_err);

	private:
		const float err_bound; // upper error bound
		std::vector<FeatureVector *> *sorted;
		unsigned int dimensions;
		unsigned int num_features;

		// contains list of indices of features that we incorrectly guessed
		// (either false_pos or false_neg). Useful for validation stats.
		std::vector<int> false_indices;

		// threshold used in getStrongError to determine whether a weak
		// classifier classified a point as POS or NEG
		float strong_err_threshold; 

		WeakClassifier* get_best_classifier();
		double weight_classifier(double err);
		bool update_feature_weight(TrainingData &td, WeakClassifier &wc);
		bool is_classifier_correct(WeakClassifier &wc, FeatureVector &fv);
		void init_feature_weight(TrainingData &td);
		void create_feature_views(TrainingData &td);
};

#endif
