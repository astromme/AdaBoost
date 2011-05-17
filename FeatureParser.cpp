#include <fstream>
#include "FeatureParser.h"

using namespace std;

/****************************************
 * Function: parseFtrVectors
 * ---------------------------
 * Parse the data given from victim_parser.py into TrainingData
 * 
 * td: TrainingData object to add to
 * ftr_file: file produced by victim_parser.py
 *
 * Effects:
 *      adds feature vectors from ftr_file to td
 */
void FeatureParser::parseFtrVectors(TrainingData &td, string ftr_file){
    int count = 0;

    ifstream infile;
    infile.open(ftr_file.c_str());

    string pos_neg;
    vector<float> ftrs;
    float tmp_ftr;

    while (! infile.eof()){

        // clear the vector
        ftrs.clear();

        // is this a positive or negative training example?
        infile >> pos_neg;

        // grab all the features
        for (int i=0; i<FTR_DIMENSION; i++){
            infile >> tmp_ftr;
            ftrs.push_back(tmp_ftr);
        }

        // set as a POSITIVE or NEGATIVE feature
        if (pos_neg == "POS"){
            FeatureVector fv(ftrs,POS);
            if (! td.addFeature(fv))
                printf("ERROR: feature vector incorrect size!\n");
        }
        else if (pos_neg == "NEG"){
            FeatureVector fv(ftrs,NEG);
            if (! td.addFeature(fv))
                printf("ERROR: feature vector incorrect size!\n");
        }

        printf("Training Data Counted: %d\r", count);
        count++;
    }

    infile.close();
}
