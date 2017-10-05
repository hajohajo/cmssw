#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <limits>

#include "getBestVertex.h"

#include "TFile.h"

namespace {
  
template<bool PROMPT>
struct mva {
  mva(const edm::ParameterSet &cfg):
    forestLabel_    ( cfg.getParameter<std::string>("GBRForestLabel") ),
    dbFileName_     ( cfg.getParameter<std::string>("GBRForestFileName") ),
    useForestFromDB_( (!forestLabel_.empty()) & dbFileName_.empty())
  {}

  void beginStream() {
    if(!dbFileName_.empty()){
      TFile gbrfile(dbFileName_.c_str());
      forestFromFile_.reset((GBRForest*)gbrfile.Get(forestLabel_.c_str()));
    }
  }

  void initEvent(const edm::EventSetup& es) {
    forest_ = forestFromFile_.get();
    if(useForestFromDB_){
      edm::ESHandle<GBRForest> forestHandle;
      es.get<GBRWrapperRcd>().get(forestLabel_,forestHandle);
      forest_ = forestHandle.product();
    }
  }

  void initDNN(const edm::EventSetup& es) {
    std::cout<<"INIT SESSION"<<std::endl;
/*    std::string GraphPath_="/afs/cern.ch/work/j/jhavukai/private/LWTNNinCMSSW/CMSSW_9_4_X_2017-10-01-0000/src/Tensorflow_graph";
    tf::Graph graph_(GraphPath_);
    session_ = tf::Session(&graph_);
//    session_->initVariables();
*/
    tf::Shape xShape_[] = {1,22};
    x_ = new tf::Tensor(2,xShape_);
    y_ = new tf::Tensor();
  }

  float operator()(reco::Track const & trk,
		   reco::BeamSpot const & beamSpot,
		   reco::VertexCollection const & vertices) const {

    auto tmva_pt_ = trk.pt();
    auto tmva_ndof_ = trk.ndof();
    auto tmva_nlayers_ = trk.hitPattern().trackerLayersWithMeasurement();
    auto tmva_nlayers3D_ = trk.hitPattern().pixelLayersWithMeasurement()
        + trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
    auto tmva_nlayerslost_ = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
    float chi2n =  trk.normalizedChi2();
    float chi2n_no1Dmod = chi2n;
    
    int count1dhits = 0;
    for (auto ith =trk.recHitsBegin(); ith!=trk.recHitsEnd(); ++ith) {
      const auto & hit = *(*ith);
      if (hit.dimension()==1) ++count1dhits;
    }
    
    if (count1dhits > 0) {
      float chi2 = trk.chi2();
      float ndof = trk.ndof();
      chi2n = (chi2+count1dhits)/float(ndof+count1dhits);
    }
    auto tmva_chi2n_ = chi2n;
    auto tmva_chi2n_no1dmod_ = chi2n_no1Dmod;
    auto tmva_eta_ = trk.eta();
    auto tmva_relpterr_ = float(trk.ptError())/std::max(float(trk.pt()),0.000001f);
    auto tmva_nhits_ = trk.numberOfValidHits();
    int lostIn = trk.hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS);
    int lostOut = trk.hitPattern().numberOfHits(reco::HitPattern::MISSING_OUTER_HITS);
    auto tmva_minlost_ = std::min(lostIn,lostOut);
    auto tmva_lostmidfrac_ = static_cast<float>(trk.numberOfLostHits()) / static_cast<float>(trk.numberOfValidHits() + trk.numberOfLostHits());
   
    float gbrVals_[PROMPT ? 16 : 12];
    gbrVals_[0] = tmva_pt_;
    gbrVals_[1] = tmva_lostmidfrac_;
    gbrVals_[2] = tmva_minlost_;
    gbrVals_[3] = tmva_nhits_;
    gbrVals_[4] = tmva_relpterr_;
    gbrVals_[5] = tmva_eta_;
    gbrVals_[6] = tmva_chi2n_no1dmod_;
    gbrVals_[7] = tmva_chi2n_;
    gbrVals_[8] = tmva_nlayerslost_;
    gbrVals_[9] = tmva_nlayers3D_;
    gbrVals_[10] = tmva_nlayers_;
    gbrVals_[11] = tmva_ndof_;

    if (PROMPT) {
      auto tmva_absd0_ = std::abs(trk.dxy(beamSpot.position()));
      auto tmva_absdz_ = std::abs(trk.dz(beamSpot.position()));
      Point bestVertex = getBestVertex(trk,vertices);
      auto tmva_absd0PV_ = std::abs(trk.dxy(bestVertex));
      auto tmva_absdzPV_ = std::abs(trk.dz(bestVertex));
      
      gbrVals_[12] = tmva_absd0PV_;
      gbrVals_[13] = tmva_absdzPV_;
      gbrVals_[14] = tmva_absdz_;
      gbrVals_[15] = tmva_absd0_;
    }

 

    return forest_->GetClassifier(gbrVals_);
    
  }

  float operator()(reco::Track const & trk,
                   reco::BeamSpot const & beamSpot,
                   reco::VertexCollection const & vertices,
		   tf::Session * session_,
		   tf::Tensor * x_,
		   tf::Tensor * y_) const {

    auto & session = *session_;

    auto tmva_pt_ = trk.pt();
    auto tmva_eta_ = trk.eta();
    auto tmva_lambda_ = trk.lambda();
    auto tmva_absd0_ = std::abs(trk.dxy(beamSpot.position()));
    auto tmva_absdz_ = std::abs(trk.dz(beamSpot.position()));
    Point bestVertex = getBestVertex(trk,vertices);
    auto tmva_absd0PV_ = std::abs(trk.dxy(bestVertex));
    auto tmva_absdzPV_ = std::abs(trk.dz(bestVertex));
    auto tmva_ptErr_ = trk.ptError();
    auto tmva_etaErr_ = trk.etaError();
    auto tmva_lambdaErr_ = trk.lambdaError();
    auto tmva_dxyErr_ = trk.dxyError();
    auto tmva_dzErr_ = trk.dzError();
    auto tmva_nChi2_ = trk.normalizedChi2();
    auto tmva_ndof_ = trk.ndof();
    auto tmva_nInvalid_ = trk.hitPattern().numberOfLostHits(reco::HitPattern::TRACK_HITS);
    auto tmva_nPixel_ = trk.hitPattern().numberOfValidPixelHits();
    auto tmva_nStrip_ = trk.hitPattern().numberOfValidStripHits();
    auto tmva_nPixelLay_ = trk.hitPattern().pixelLayersWithMeasurement();
    auto tmva_nStripLay_ = trk.hitPattern().stripLayersWithMeasurement();
    auto tmva_n3DLay_ = (trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo()+trk.hitPattern().pixelLayersWithMeasurement());
    auto tmva_nLostLay_ = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
    auto tmva_algo_ = trk.algo();

    std::vector<float> gbrVals_; 
    gbrVals_.push_back(tmva_pt_);
    gbrVals_.push_back(tmva_eta_);
    gbrVals_.push_back(tmva_lambda_);
    gbrVals_.push_back(tmva_absd0_);
    gbrVals_.push_back(tmva_absdz_);
    gbrVals_.push_back(tmva_absd0PV_);
    gbrVals_.push_back(tmva_absdzPV_);
    gbrVals_.push_back(tmva_ptErr_);
    gbrVals_.push_back(tmva_etaErr_);
    gbrVals_.push_back(tmva_lambdaErr_);
    gbrVals_.push_back(tmva_dxyErr_);
    gbrVals_.push_back(tmva_dzErr_);
    gbrVals_.push_back(tmva_nChi2_);
    gbrVals_.push_back(tmva_ndof_);
    gbrVals_.push_back(tmva_nInvalid_);
    gbrVals_.push_back(tmva_nPixel_);
    gbrVals_.push_back(tmva_nStrip_);
    gbrVals_.push_back(tmva_nPixelLay_);
    gbrVals_.push_back(tmva_nStripLay_);
    gbrVals_.push_back(tmva_n3DLay_);
    gbrVals_.push_back(tmva_nLostLay_);
    gbrVals_.push_back(tmva_algo_);


    x_->setVector<float>(1,0,gbrVals_);
//    std::string GraphPath_="/afs/cern.ch/work/j/jhavukai/private/LWTNNinCMSSW/CMSSW_9_4_X_2017-10-01-0000/src/Tensorflow_graph";
//    tf::Graph graph_(GraphPath_);
//    tf::Session session_(&graph_);
    tf::IOs inputs = { session.createIO(x_, "ins") };
    tf::IOs outputs = { session.createIO(y_, "outs/Sigmoid") };
    session.run(inputs, outputs);

    float output_ = *y_->getPtr<float>(0, 0);

    //Rescale to match with the BDT range traditionally used in Iterative Tracking
    output_=2.0*output_-1.0;

    return output_;
  }

  static const char * name();

  static void fillDescriptions(edm::ParameterSetDescription & desc) {
    desc.add<std::string>("GBRForestLabel",std::string());
    desc.add<std::string>("GBRForestFileName",std::string());
  }
  
  std::unique_ptr<GBRForest> forestFromFile_;
  const GBRForest *forest_ = nullptr; // owned by somebody else
  const std::string forestLabel_;
  const std::string dbFileName_;
  const bool useForestFromDB_;

//  tf::Graph graph_;
  tf::Session session_;
  tf::Tensor* x_ = nullptr;
  tf::Tensor* y_ = nullptr;



};

  using TrackMVAClassifierDetached = TrackMVAClassifier<mva<false>>;
  using TrackMVAClassifierPrompt = TrackMVAClassifier<mva<true>>;
  template<>
  const char * mva<false>::name() { return "TrackMVAClassifierDetached";}
  template<>
  const char * mva<true>::name() { return "TrackMVAClassifierPrompt";}
  
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackMVAClassifierDetached);
DEFINE_FWK_MODULE(TrackMVAClassifierPrompt);
