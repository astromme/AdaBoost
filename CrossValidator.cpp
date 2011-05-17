#include "CrossValidator.h"
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <stdlib.h>

using namespace std;

/**************************************
 * Function: validate
 * -------------------------
 * runs cross-validation on a set of victims -- prints results to stdout
 */
ErrorStruct CrossValidator::validate(vector<FeatureVector> *shuffled, unsigned int partitions,
        unsigned int classifiers){
    AdaBooster ada;
    StrongClassifier strong; // strong classifier
    ErrorStruct strong_error;
	vector< vector< vector<double> > > strong_err_vec;

    vector<FeatureVector> victim_list;
    int training_count; // counts training sets so far
    for (unsigned int test=0; test < partitions; test++){
        training_count = 0;
        for (unsigned int train=0; train < partitions; train++){
            // make sure we're not training on the testing partition
            if (test != train){
                // give informative output
                training_count++;
                printf("Creating Training Set %d\r", training_count);
                fflush(stdout);

                // add current partition of victims to victim_list
                for (unsigned int i=0; i<shuffled[train].size(); i++)
                    victim_list.push_back(shuffled[train][i]);
            }
        }

        TrainingData train_data;
        for (unsigned int i=0; i<victim_list.size(); i++)
            train_data.addFeature(victim_list[i]);

        printf("\rCreating Strong Classifier");
        fflush(stdout);

        strong = ada.getStrongClassifier(train_data,classifiers);

        TrainingData test_data;
        for (unsigned int i=0; i<shuffled[test].size(); i++) {
            test_data.addFeature(shuffled[test][i]);
        }

        // get strong error
        //vector< vector<double> > strong_err = ada.getStrongError(test_data, strong.weakClassifiers());
        //strong_err_vec.push_back(strong_err);

        victim_list.clear();
        
        printf("\rTesting set %d completed       \n", test+1);

        strong_error = strong_error + (strong.errorForFeatures(test_data));
    }

    strong_error = strong_error / partitions;
    // clean up
    delete [] shuffled;
    
    //double **strong_err_avg = get_strong_err_avg(strong_err_vec);
    //print_strong_err_avg(strong_err_avg, strong_err_vec[0].size());

    std::cout << strong_error.true_pos << " true positives" << std::endl;
    std::cout << strong_error.false_pos << " false positives" << std::endl;
    std::cout << strong_error.true_neg << " true negatives" << std::endl;
    std::cout << strong_error.false_neg << " false negatives" << std::endl;
    std::cout << strong_error.error * 100 << "% error" << std::endl;
    std::cout << std::endl;

    // finish cleaning up
    //for (unsigned int i=0; i<strong_err_vec[0].size(); i++)
    //	delete [] strong_err_avg[i];
    //delete [] strong_err_avg;

    return strong_error;
}


/*****************************
 * Function: shufflePeople
 * "Shuffles" training data into partitions
 * 
 * td: training data that we're going to extract
 * partitions: # of partitions to shuffle into
 *
 * Returns: array of vectors of FeatureVectors. Each entry in array is a
 *	partition of "people" == vector<FeatureVector>
 */
vector<FeatureVector> *CrossValidator::shuffleTrainingData(const TrainingData &td, unsigned int partitions){
    TrainingData copy = td;
    vector<FeatureVector> *shuffled = new vector<FeatureVector>[partitions];
    // traverse people
    srand(time(NULL));
    int count = 0;
    while (copy.size() > 0) {
      int randomElementLocation = rand() % copy.size();
      shuffled[count % partitions].push_back(copy.removeFeatureAt(randomElementLocation));
      count++;
    }

    /*
    for (unsigned int i=0; i<td.size(); i++){
        // split people into k categories
        shuffled[i%partitions].push_back(*td.feature(i));
    }
    */
    return shuffled;
}

/***********************************
 * Function: get_strong_err_avg
 * ----------------------------
 * given a vector of strong error vectors, we want to average them all
 * together. Note that we return a double array of doubles, so we have to
 * delete that allocation at some point (right now, in runValidate)
 */
double** CrossValidator::get_strong_err_avg(
		vector< vector<	vector<double> > > &strong){
	double **avg;

	// allocate memory for average
	avg = new double*[strong[0].size()];
	for(unsigned int i=0; i<strong[0].size(); i++){
		avg[i] = new double[strong[0][0].size()];
	}

	// traverse grids and add values together
	for (unsigned int j=0; j<strong.size(); j++){
		for (unsigned int k=0; k<strong[j].size(); k++){
			for (unsigned int m=0; m<strong[j][k].size(); m++){
				if (j == 0)
					avg[k][m] = strong[j][k][m];
				else
					avg[k][m] += strong[j][k][m];
			}
		}
	}

	// now divide by number of partitions (size of outer-most vector)
	// (but we don't want to average true/false positive/negative
	// values, so don't divide by those)
	for (unsigned int k=0; k<strong[0].size(); k++){
		for (unsigned int m=0; m<strong[0][0].size()-4; m++){
			avg[k][m] /= strong.size();
		}
	}

	if ( isnan(avg[strong[0].size()-1][1]) || 	// check if precision is nan
		isnan(avg[strong[0].size()-1][2]) ) { // check if recall is nan
		// recompute values
		recompute_prcsn_recall(avg, strong[0].size());
	}

	return avg;
}

/*****************************************
 * Function: recompute_prcsn_recall
 * ---------------------------------
 * Given avg from get_strong_err_avg and number or rows in the double array, we
 * go through and recompute the precision and recall. This function only gets
 * called when the precision or recall is NaN. This can be caused by, somewhere
 * along the line, the precision or recall is NaN for one of the sets, and
 * therefore when we try to average them all together, we get NaN for the whole
 * thing. So, if that's the case, we just recalculate the stats and go on with
 * our day.
 *
 * Note: this function is void but does alter the pointer it's given.
 *
 * Precision = (true pos / (true pos + false pos) )
 * Recall = (true pos / (true pos + false neg) )
 */
void CrossValidator::recompute_prcsn_recall(double **avg, unsigned int rows){
	int true_pos, true_neg, false_pos, false_neg;
	double prcsn, recall;

	for (unsigned int r=0; r<rows; r++){
		// get true/false pos/neg values
		true_pos = avg[r][3];
		true_neg = avg[r][4];
		false_pos = avg[r][5];
		false_neg = avg[r][6];

		// calculate new precision and recall
		prcsn = (double) true_pos / (true_pos + false_pos);
		recall = (double) true_pos / (true_pos + false_neg);

		// set precision and recall
		avg[r][1] = prcsn;
		avg[r][2] = recall;
	}
}


/**************************************
 * Function: print_strong_err_avg
 * ------------------------------
 * given output from get_strong_err_avg() as input and the number of rows to
 * print out, we print out all the good stuff
 */
void CrossValidator::print_strong_err_avg(double **avg, unsigned int row){
	printf("                   \r\tStrong Error Average\n");
	printf("\t--------------------\n");
	printf(" idx \terror\t prcsn \trecall\t tp \t tn \t fp \t fn\n");
	printf("-----\t------\t------\t------\t----\t----\t----\t----\n");
	
	for (unsigned int r=0; r<row; r++){
                printf("[%03d]\t%6.2f\t%6.2f\t%6.2f\t%4d\t%4d\t%4d\t%4d\n",r,
			avg[r][0]*100,avg[r][1]*100,avg[r][2]*100,
			(int)avg[r][3],(int)avg[r][4],(int)avg[r][5],(int)avg[r][6]);
	}
	
	printf("-----\t------\t------\t------\t----\t----\t----\t----\n");
	printf(" idx \terror\t prcsn \trecall\t tp \t tn \t fp \t fn\n");
}
