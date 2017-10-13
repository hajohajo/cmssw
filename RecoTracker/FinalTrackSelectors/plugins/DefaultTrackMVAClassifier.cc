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

  void beginStream(){}

  void initEvent(const edm::EventSetup& es) {
    path_ = "/afs/cern.ch/work/j/jhavukai/private/LWTNNinCMSSW/CMSSW_9_3_X_2017-09-25-1100/src/Tensorflow_graph";
    metaGraph_ = tf::loadMetaGraph(path_);
    session_ = tf::createSession(metaGraph_,path_);
    x_ = tf::Tensor(tf::DT_FLOAT,{1,22});
//    inputs_.push_back(x_);
    input_names_.push_back("ins");
    output_names_.push_back("outs/Sigmoid");
  }

  float operator()(reco::Track const & trk,
		   reco::BeamSpot const & beamSpot,
		   reco::VertexCollection const & vertices) const {

    std::cout<<"Start of operator"<<std::endl;
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

    std::cout<<"Start of tensorflow magic"<<std::endl;
    //This is silly, fixing it soon
    tf::Tensor input(tf::DT_FLOAT,{1,22});
//    for (size_t i = 0; i<22;i++) input.matrix<float>()(0,i) = float(i);
    std::vector<tf::Tensor> innes_;
    std::cout<<"Loading x_"<<std::endl;
    for (size_t i = 0; i<22;i++) input.matrix<float>()(0,i) = gbrVals_[i];
    innes_.push_back(input);

    std::vector<tf::Tensor> outputs_;

    std::cout<<"Running..."<<std::endl;
    tf::run(session_,input_names_,inputs_,output_names_, &outputs_);

    std::cout<<"Reading output"<<std::endl;
    float dnn_value = outputs_[0].matrix<float>()(0,0);
//    float dnn_value = 0.0;
    std::cout<<dnn_value<<std::endl;

    return dnn_value;
  }

  static const char * name();

  static void fillDescriptions(edm::ParameterSetDescription & desc) {}

  std::string path_;
  tf::Session* session_ = nullptr; //*
  tf::MetaGraphDef* metaGraph_ = nullptr; //*
  tf::Tensor x_; // = nullptr;
  tf::Tensor* y_ = nullptr;
  std::vector<tf::Tensor> inputs_;
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
