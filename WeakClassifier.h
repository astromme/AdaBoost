#ifndef _WEAK_CLASSIFIER_H
#define _WEAK_CLASSIFIER_H

#include <string>
#include <vector>

const int ZERO = 0;

/******************************
 * Class: WeakClassifier
 * ---------------------
 * Weak classifiers have a threshold and a flipped boolean which together tell
 * us if the WeakClassifier correctly classified a feature. It also has a
 * weight.
 */
class WeakClassifier {
	public:
		WeakClassifier(unsigned int in_dim, float in_thresh,bool in_flip, float in_wt) :
			dim(in_dim), thresh(in_thresh), flipped(in_flip), wt(in_wt) {}; // constructor

		int dimension() const;
		float weight() const;
		float threshold() const;
		bool isFlipped() const;

		// Output
		void printClassifier() const;
		void writeClassifier(std::string fname) const;

	private:
		unsigned int dim; // dimension
		float thresh; // threshold
		bool flipped;
		float wt; // weight

};

#endif
