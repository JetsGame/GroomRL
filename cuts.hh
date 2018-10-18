#ifndef __cuts_HH__
#define __cuts_HH__

#include "fastjet/Selector.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/tools/Recluster.hh"

using namespace fastjet;
using namespace std;

/// this is the common jet definition that we
/// will all use
inline JetDefinition common_jet_definition() {
  double R = 1.0;
  //return JetDefinition(antikt_algorithm, R);
  return JetDefinition(aachen_algorithm, R);
}

/// this is the jet selector that is to be used as common between all the
/// runs
inline Selector common_jet_selector(double ptmin, double ptmax = 1e100) {
  const double rapmax = 2.5;
  return  SelectorPtRange(ptmin,ptmax) * SelectorNHardest(2) * SelectorAbsRapMax(rapmax);
}

inline JetDefinition common_jet_recluster_def() {
  return JetDefinition(aachen_algorithm,100.0);
}

inline Recluster common_jet_recluster() {
    return Recluster(common_jet_recluster_def());
}

#endif
