#ifndef STRONGCLASSIFIER_H
#define STRONGCLASSIFIER_H

#include <vector>
#include "WeakClassifier.h"
#include "TrainingData.h"

typedef std::vector<WeakClassifier> WeakClassifierList;

class ErrorStruct {
public:
  ErrorStruct() :
    error(0),
    true_pos(0),
    false_pos(0),
    true_neg(0),
    false_neg(0)
  {}

  double error;
  int true_pos;
  int false_pos;
  int true_neg;
  int false_neg;
};

ErrorStruct operator+(const ErrorStruct &s1, const ErrorStruct &s2);
ErrorStruct operator/(const ErrorStruct &s1, int divisor);

class StrongClassifier
{
public:
    StrongClassifier();
    StrongClassifier(const std::vector<WeakClassifier> &weakList);
    StrongClassifier(const StrongClassifier &other);

    WeakClassifierList weakClassifiers() const;

    float evaluate(const std::vector<float> &features) const;
    bool decide(const std::vector<float> &features) const;
    bool decide(const FeatureVector &features) const;

    ErrorStruct errorForFeatures(const TrainingData &features, bool printStats=false) const;

private:
    WeakClassifierList m_weakClassifiers;
};

#endif // STRONGCLASSIFIER_H
