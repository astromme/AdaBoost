#include <fstream>
#include <iostream>
#include "WeakClassifier.h"

using namespace std;

// getter methods
int WeakClassifier::dimension() const { return dim; }
float WeakClassifier::weight() const { return wt; }
float WeakClassifier::threshold() const { return thresh; }
bool WeakClassifier::isFlipped() const { return flipped; }

// prints all instance fields of a weak classifier
void WeakClassifier::printClassifier() const {
	printf("dimension: %d\t threshold: %.10f\t flipped: %s\t weight: %f\n",
			dim, thresh, (flipped)?"true ":"false",wt);
}

// writes classifier to file
void WeakClassifier::writeClassifier(string fname) const {
	ofstream outFile(fname.c_str(), ios::app);
	outFile << dim << "\t\t" << thresh << "\t\t" << flipped << "\t\t" << wt << endl;
	outFile.close();
}
