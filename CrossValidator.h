#ifndef _CROSS_VALIDATOR_H
#define _CROSS_VALIDATOR_H

#include <vector>
#include "AdaBooster.h"

using std::vector;

/***************************
 * Class: VictimValidator
 * This class is used to run cross validation on victim data from the
 * CS63-final project
 */
class CrossValidator {
    public:
        ErrorStruct validate(vector<FeatureVector> *shuffled, unsigned int partitions, unsigned int classifiers);
        vector<FeatureVector> *shuffleTrainingData(const TrainingData &td, unsigned int partitions);

    private:
        double** get_strong_err_avg(vector< vector<	vector<double> > > &strong);
        void recompute_prcsn_recall(double **avg, unsigned int rows);
        void print_strong_err_avg(double **avg, unsigned int row);
};

#endif
