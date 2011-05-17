cmake_minimum_required(VERSION 2.6)

ADD_LIBRARY(adaboost 
  sorting.cpp 
  FeatureVector.cpp WeakClassifier.cpp 
  TrainingData.cpp AdaBooster.cpp
  CrossValidator.cpp FeatureParser.cpp
  StrongClassifier.cpp
)
