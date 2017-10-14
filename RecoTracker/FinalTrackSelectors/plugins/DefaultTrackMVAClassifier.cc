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
struct dnn {
  dnn(const edm::ParameterSet &cfg)
//    std::string path_; //= "/afs/cern.ch/work/j/jhavukai/private/LWTNNinCMSSW/CMSSW_9_3_X_2017-09-25-1100/src/Tensorflow_graph";
//    tf::MetaGraphDef* metaGraph_;
//    tf::Session session_; //(&graph_);
////    xShape_[], // = {1,22},
//    tf::Tensor* x_;
//    tf::Tensor* y_;
//    tf::Session* session_;

  {}

  void beginStream(){
    path_ = "/afs/cern.ch/work/j/jhavukai/private/LWTNNinCMSSW/NewTest/CMSSW_9_4_X_2017-10-01-0000/src/Tensorflow_graph";
    input_names_.push_back("ins");
    output_names_.push_back("outs/Sigmoid");
    metaGraph_ = tf::loadMetaGraph(path_);
    session_ = tf::createSession(metaGraph_,path_);
  }

  void initEvent(const edm::EventSetup& es) {}

  float operator()(reco::Track const & trk,
		   reco::BeamSpot const & beamSpot,
		   reco::VertexCollection const & vertices) const {

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

    std::vector<tf::Tensor> inputs_;
    tf::Tensor input(tf::DT_FLOAT,{1,22});
    input.matrix<float>()(0,0)=tmva_pt_;
    input.matrix<float>()(0,1)=tmva_eta_;
    input.matrix<float>()(0,2)=tmva_lambda_;
    input.matrix<float>()(0,3)=tmva_absd0_;
    input.matrix<float>()(0,4)=tmva_absdz_;
    input.matrix<float>()(0,5)=tmva_absd0PV_;
    input.matrix<float>()(0,6)=tmva_absdzPV_;
    input.matrix<float>()(0,7)=tmva_ptErr_;
    input.matrix<float>()(0,8)=tmva_etaErr_;
    input.matrix<float>()(0,9)=tmva_lambdaErr_;
    input.matrix<float>()(0,10)=tmva_dxyErr_;
    input.matrix<float>()(0,11)=tmva_dzErr_;
    input.matrix<float>()(0,12)=tmva_nChi2_;
    input.matrix<float>()(0,13)=tmva_ndof_;
    input.matrix<float>()(0,14)=tmva_nInvalid_;
    input.matrix<float>()(0,15)=tmva_nPixel_;
    input.matrix<float>()(0,16)=tmva_nStrip_;
    input.matrix<float>()(0,17)=tmva_nPixelLay_;
    input.matrix<float>()(0,18)=tmva_nStripLay_;
    input.matrix<float>()(0,19)=tmva_n3DLay_;
    input.matrix<float>()(0,20)=tmva_nLostLay_;
    input.matrix<float>()(0,21)=tmva_algo_;

    inputs_.push_back(input);

    std::vector<tf::Tensor> outputs_;

    tf::run(session_,input_names_,inputs_,output_names_, &outputs_);

    float dnn_value = outputs_[0].matrix<float>()(0,0);

    inputs_.clear();
    outputs_.clear();

    return dnn_value;
  }

  static const char * name();

  static void fillDescriptions(edm::ParameterSetDescription & desc) {}

  std::string path_;
  tf::Session* session_ = nullptr; //*
  tf::MetaGraphDef* metaGraph_ = nullptr; //*
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;


};

  using TrackDNNClassifierVar22 = TrackDNNClassifier<dnn<true>>;
  template<>
  const char * dnn<true>::name() { return "TrackDNNClassifier";}

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackDNNClassifierVar22);

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
