#ifndef _FEATURE_PARSER_H
#define _FEATURE_PARSER_H

#include <string>
#include "TrainingData.h"

using std::string;

// simple class that parses feature vectors from file created by
// victim_parser.py
class FeatureParser {
    public:
        void parseFtrVectors(TrainingData &td, string ftr_file);

    private:
        static const int FTR_DIMENSION = 16;
};

#endif
