/// python friendly version, which produces json files with jet
/// constituents and declusterings
/// 
/// simple program to get a quick estimate of cross sections and a
/// handful of basic distributions from MC run and flat/HepMC/UW
/// files, etc.
///
#include "AnalysisFramework.hh"
#include "boost/foreach.hpp"
#include "fastjet/tools/Recluster.hh"
#include "fastjet/contrib/SoftDrop.hh"
#include "json.hpp"
#include "cuts.hh"
#include <queue>
#define foreach BOOST_FOREACH


using namespace std;
using namespace fastjet;
using nlohmann::json;

double log10(double x) {return log(x)/log(10.0);}


struct Declustering {
  // the (sub)jet, and its two declustering parts, for subsequent use
  PseudoJet jj, j1, j2;
  // variables of the (sub)jet about to be declustered
  double pt, m;
  // properties of the declustering; NB kt is the _relative_ kt
  // = pt2 * delta_R (ordering is pt2 < pt1)
  double pt1, pt2, delta_R, z, kt, varphi;
};

struct Particle {
  double px, py, pz, E;
};

void to_json(json& j, const Declustering& d) {
  j = json{{"pt", float(d.pt)}, {"m", float(d.m)},
           {"pt1", float(d.pt1)}, {"pt2", float(d.pt2)},
           {"delta_R", float(d.delta_R)}, {"z",float(d.z)},
	   {"kt",float(d.kt)}, {"varphi", float(d.varphi)}};
}

void to_json(json& j, const Particle& p) {
  j = json{{"px", float(p.px)}, {"py", float(p.py)}, {"pz", float(p.pz)},
	   {"E", float(p.E)}};
}

// definitions needed for comparison of subjets
struct CompareJetsWithDeltaRsqr {
  // return the squared Delta R value between the two subjets
  // is there a better way of doing this by saving the cluster sequence in Recluster??
  double jet_deltaRsqr(const PseudoJet& jet) const {
    PseudoJet piece1, piece2;
    if (jet.has_parents(piece1,piece2))
      return piece1.squared_distance(piece2);
    return 0.0;
  }
    
  bool operator ()(const PseudoJet& j1, const PseudoJet& j2) const {
    return jet_deltaRsqr(j1) < jet_deltaRsqr(j2);
  }
};

/// Example class derived from AnalysisFramework that will help in evaluation of
/// some basic cross sections. 
class XSctAnalysis : public AnalysisFramework {
  unique_ptr<CleverOFStream> jsonfile, jsonfile_constit;
  Selector jet_sel;
  Recluster jet_rec;

  int number_of_imgs;
  int current_img_number;

  contrib::SoftDrop sd;
  bool do_SD, find_W;
  
public:
  XSctAnalysis(CmdLine & cmdline) : AnalysisFramework(cmdline),
				    sd(0.0, cmdline.value<double>("-zcut",0.1),
				       cmdline.value<double>("-R",1.0)),
				    do_SD(cmdline.present("-SD")),
				    find_W(cmdline.present("-find-W")){}
  ~XSctAnalysis() {}
  void user_startup() {
    // extra user parameters
    param["missing.ptmin"] = 30.0;
    param["jet.ptmin"] = cmdline.value<double>("-ptmin",800.0);
    //param["jet.rapmax"] = 5.0;
    param["zmin"] = 0.2;
    param["zmax"] = 0.3;

    number_of_imgs=cmdline.value<double>("-nimg",10000);
    current_img_number=0;
    //jet_def = JetDefinition(antikt_algorithm, R);
    //jet_sel = SelectorPtMin(param["jet.ptmin"]) && SelectorAbsRapMax(param["jet.rapmax"]);

    jet_def = common_jet_definition();
    jet_sel = common_jet_selector(param["jet.ptmin"]);
    jet_rec = common_jet_recluster();

    jsonfile.reset(new CleverOFStream(cmdline.value<string>("-out-lund")+".json.gz"));
    if (cmdline.present("-out-jet")) 
      jsonfile_constit.reset(new CleverOFStream(cmdline.value<string>("-out-jet")+".json.gz"));
    else
      jsonfile_constit = 0;
    
    averages["selected-jet-pt"].ref_xsection = "selected-jet-cross-section";
    
    DefaultHist::set_defaults(0.0, 4.4, cmdline.value("-hist.binwidth",0.2));

    hists_2d["lund-zrel"].declare(0.0, 9.0, 0.2, -12.0, 0.0, 0.4);
    hists_2d["lund-zabs"].declare(0.0, 9.0, 0.2, -12.0, 0.0, 0.4);
    hists_2d["lund-lnpt"].declare(0.0, 9.0, 0.2,  -5.0, 8.0, 0.4);
  }

  void user_output(std::ostream &) {
    cout << "Outputting jet and lund images "<<current_img_number << " out of "
	 << number_of_imgs << endl;
  }

  void create_queue(const PseudoJet& j,
		    priority_queue<PseudoJet, vector<PseudoJet>,
		                        CompareJetsWithDeltaRsqr>& pq) {
    PseudoJet j1,j2;
    if (j.has_parents(j1,j2)) {
      pq.push(j);
      create_queue(j1,pq);
      create_queue(j2,pq);
    }
  }
  
  /// returns a vector of (primary) C/A declusterings for the jet,
  /// in declustering order, i.e. decreasing delta_R
  vector<Declustering> jet_declusterings(const PseudoJet & jet_in) {

    PseudoJet j = jet_rec(jet_in);
    
    vector<Declustering> result;
    PseudoJet jj, j1, j2;
    jj = j;
    priority_queue<PseudoJet, vector<PseudoJet>, CompareJetsWithDeltaRsqr> pq;
    create_queue(j, pq);
    while(!pq.empty()) {
      PseudoJet jj, j1, j2;
      jj = pq.top();
      pq.pop();
      if (jj.has_parents(j1,j2)) {
	Declustering declust;
        // make sure j1 is always harder branch
        if (j1.pt2() < j2.pt2()) swap(j1,j2);

        // store the subjets themselves
        declust.jj   = jj;
        declust.j1   = j1;
        declust.j2   = j2;
        
        // get info about the jet 
        declust.pt   = jj.pt();
        declust.m    = jj.m();

        // collect info about the declustering
        declust.pt1     = j1.pt();
        declust.pt2     = j2.pt();
        declust.delta_R = j1.delta_R(j2);
        declust.z       = declust.pt2 / (declust.pt1 + declust.pt2);
        declust.kt      = j2.pt() * declust.delta_R;

	// this is now phi along the jet axis, defined in a
	// long. boost. inv. way
        declust.varphi = atan2(j1.rap()-j2.rap(), j1.delta_phi_to(j2));

	// add it to our result
        result.push_back(declust);
      }
    }
    return result;
    if (result.size() == 0) {
      cerr << "HEY THERE, empty declustering. Jet p_t and n_const = " << jet_in.pt() << " " << jet_in.constituents().size() << endl;
    }
  }

  vector<Particle> constituents(const PseudoJet& j) {
    vector<Particle> res;
    Particle pjet;
    pjet.px = j.px();
    pjet.py = j.py();
    pjet.pz = j.pz();
    pjet.E = j.E();
    res.push_back(pjet);
    for(PseudoJet jj : j.constituents()) {
      Particle c;
      c.px = jj.px();
      c.py = jj.py();
      c.pz = jj.pz();
      c.E  = jj.E();
      res.push_back(c);
    }
    return res;
  }
  
  void analyse_event() {
    // cout << "Cross section pointer: " 
    // 	 << driver->generator->gen_event()->cross_section()  << " "
    // 	 << driver->generator->gen_event()->weights().size()
    // 	 << endl;

    double evwgt = driver->generator->hadron_level().weight();
    xsections["total cross section"] += evwgt;

    // // temporary for Ken Lane: muon rapidity
    // auto muons = (SelectorNHardest(2)*SelectorAbsPDGId(13))(driver->generator->hadron_level().particles());
    // hists["mupair.rap"].set_lims_add_entry(-10.0, 10.0, 1.0, (muons[0]+muons[1]).rap(),evwgt);
    // hists["mupair.eta"].set_lims_add_entry(-10.0, 10.0, 1.0, (muons[0]+muons[1]).eta(),evwgt);

    auto particles  = driver->generator->hadron_level().particles();
    
    averages["event multipliticity"] += evwgt * driver->generator->hadron_level().particles().size();

    auto jets = SelectorNHardest(2)(jet_sel(jet_def(particles)));

    for (const auto & j: jets) {
      
      PseudoJet jet = j;
      if (do_SD) jet = sd(jet);
      PseudoJet jj, j1, j2;
      jj = jet_rec(jet);
      bool found_W = false;
      while (jj.has_parents(j1,j2)) {
        // make sure j1 is always harder branch
        if (j1.pt2() < j2.pt2()) swap(j1,j2);

        // collect info and fill in the histogram
        double delta_R = j1.delta_R(j2);
        double delta_R_norm = delta_R / jet_def.R();
        double z = j2.pt()/(j1.pt() + j2.pt());
        double y = log(1.0 / delta_R_norm);
	//std::cout << j.pt() << ": " << j1.pt() << ", " << j2.pt()
	//          << "; " << delta_R<<" <> "<<j2.pt()*delta_R<< std::endl;

        // there is an ambiguity here: can use z or j2.pt() / j.pt()
        double lnpt_rel = log(z * delta_R_norm);
        double lnpt_abs = log(j2.pt()/jet.pt() * delta_R_norm);
	
        hists_2d["lund-zrel"].add_entry(y, lnpt_rel, evwgt);
        hists_2d["lund-zabs"].add_entry(y, lnpt_abs, evwgt);

        double lnpt = log(j2.pt() * delta_R);
        hists_2d["lund-lnpt"].add_entry(y, lnpt, evwgt);

	// check whether we found a W
        if (    70.0 < jj.m() && jj.m() < 90.0
            &&  param["zmin"] < z && z < param["zmax"]) 
	  found_W = true;
	
        // follow harder branch
        jj = j1;
      }
      if (find_W && ! found_W) continue;
      
      auto declusterings = jet_declusterings(j);
      //record_declusterings(declusterings, j, jet_def.R(), "primary");
      
      if (current_img_number < number_of_imgs) {
	++current_img_number;
	json J = declusterings;
	*jsonfile << J << endl;
	if (jsonfile_constit) {
	  json J_constits = constituents(j);
	  *jsonfile_constit << J_constits << endl;
	}
      }
      
    }
  }
};

//----------------------------------------------------------------------
int main (int argc, char ** argv) {
  
  CmdLine cmdline(argc,argv);
  XSctAnalysis analysis(cmdline);
  analysis.run();
}
