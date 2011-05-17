#include "StrongClassifier.h"
#include <iostream>

ErrorStruct operator+(const ErrorStruct &s1, const ErrorStruct &s2) {
  ErrorStruct added;
  added.error = s1.error + s2.error;
  added.true_pos = s1.true_pos + s2.true_pos;
  added.false_pos = s1.false_pos + s2.false_pos;
  added.true_neg = s1.true_neg + s2.true_neg;
  added.false_neg = s1.false_neg + s2.false_neg;

  return added;
}

ErrorStruct operator/(const ErrorStruct &s1, int divisor) {
  ErrorStruct divided;
  divided.error = s1.error/divisor;
  divided.true_pos = s1.true_pos/divisor;
  divided.false_pos = s1.false_pos/divisor;
  divided.true_neg = s1.true_neg/divisor;
  divided.false_neg = s1.false_neg/divisor;

  return divided;
}

StrongClassifier::StrongClassifier()
{
}

StrongClassifier::StrongClassifier(const std::vector<WeakClassifier> &weakList) {
  m_weakClassifiers = weakList;
}

StrongClassifier::StrongClassifier(const StrongClassifier &other){
  m_weakClassifiers = other.m_weakClassifiers;
}

WeakClassifierList StrongClassifier::weakClassifiers() const {
  return m_weakClassifiers;
}

float StrongClassifier::evaluate(const std::vector<float> &features) const {
  float decision = 0;
  for (int i=0; i<m_weakClassifiers.size(); i++) {
    WeakClassifier weak = m_weakClassifiers.at(i);

    int sign;
    if ( (weak.threshold() > features[weak.dimension()] && !weak.isFlipped()) ||
            (weak.threshold() < features[weak.dimension()] && weak.isFlipped()) )
        sign = 1;
    else
        sign = -1;
    decision += weak.weight() * sign;
  }
  return decision;
}

bool StrongClassifier::decide(const std::vector<float> &features) const {
  return (evaluate(features) > 0);
}

bool StrongClassifier::decide(const FeatureVector &featureVector) const {
  std::vector<float> features(featureVector.size());
  for (int i=0; i<featureVector.size(); i++) {
    features[i] = featureVector.at(i);
  }

  return decide(features);
}

ErrorStruct StrongClassifier::errorForFeatures(const TrainingData &features, bool printStats) const {

  ErrorStruct e;

  for (int i=0; i<features.size(); i++) {
    FeatureVector feature = *(features.feature(i));
    if (decide(feature)) {
      feature.val() == POS ? e.true_pos++ : e.false_pos++;
    } else {
      feature.val() == NEG ? e.true_neg++ : e.false_neg++;
    }
  }

  e.error = (e.false_pos + e.false_neg) / ((float)features.size());

  if (printStats) {
    std::cout << e.true_pos << " true positives" << std::endl;
    std::cout << e.false_pos << " false positives" << std::endl;
    std::cout << e.true_neg << " true negatives" << std::endl;
    std::cout << e.false_neg << " false negatives" << std::endl;
    std::cout << e.error * 100 << "% error" << std::endl;
    std::cout << std::endl;
  }

  return e;
}

