#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include<cassert>


void TrackMVAClassifierBase::fill( edm::ParameterSetDescription& desc) {
  desc.add<edm::InputTag>("src",edm::InputTag());
  desc.add<edm::InputTag>("beamspot",edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("vertices",edm::InputTag("firstStepPrimaryVertices"));
  desc.add<bool>("ignoreVertices",false);
  desc.add<std::string>("GBRForestLabel",std::string());
  desc.add<std::string>("GBRForestFileName",std::string());
  desc.add<bool>("OverrideWithDNN",false);
  // default cuts for "cut based classification"
  std::vector<double> cuts = {-.7, 0.1, .7};
  desc.add<std::vector<double>>("qualityCuts", cuts);
}


TrackMVAClassifierBase::~TrackMVAClassifierBase(){}

TrackMVAClassifierBase::TrackMVAClassifierBase( const edm::ParameterSet & cfg ) :
  src_     ( consumes<reco::TrackCollection>   (cfg.getParameter<edm::InputTag>( "src" ))      ),
  beamspot_( consumes<reco::BeamSpot>          (cfg.getParameter<edm::InputTag>( "beamspot" )) ),
  vertices_( mayConsume<reco::VertexCollection>(cfg.getParameter<edm::InputTag>( "vertices" )) ),
  ignoreVertices_( cfg.getParameter<bool>( "ignoreVertices" ) ),
  forestLabel_   ( cfg.getParameter<std::string>("GBRForestLabel") ),
  dbFileName_    ( cfg.getParameter<std::string>("GBRForestFileName") ),
  useForestFromDB_( (!forestLabel_.empty()) & dbFileName_.empty()),
  overrideWithDNN_( cfg.getParameter<bool>( "OverrideWithDNN") ){

  auto const & qv  = cfg.getParameter<std::vector<double>>("qualityCuts");
  assert(qv.size()==3);
  std::copy(std::begin(qv),std::end(qv),std::begin(qualityCuts));

  produces<MVACollection>("MVAValues");
  produces<QualityMaskCollection>("QualityMasks");

}

void TrackMVAClassifierBase::produce(edm::Event& evt, const edm::EventSetup& es ) {

  // Get tracks 
  edm::Handle<reco::TrackCollection> hSrcTrack;
  evt.getByToken(src_, hSrcTrack );
  auto const & tracks(*hSrcTrack);

  // looking for the beam spot
  edm::Handle<reco::BeamSpot> hBsp;
  evt.getByToken(beamspot_, hBsp);

  // Select good primary vertices for use in subsequent track selection
  edm::Handle<reco::VertexCollection> hVtx;
  evt.getByToken(vertices_, hVtx);

  GBRForest const * forest = forest_.get();
  if(useForestFromDB_){
    edm::ESHandle<GBRForest> forestHandle;
    es.get<GBRWrapperRcd>().get(forestLabel_,forestHandle);
    forest = forestHandle.product();
  }

  // products
  auto mvas  = std::make_unique<MVACollection>(tracks.size(),-99.f);
  auto quals = std::make_unique<QualityMaskCollection>(tracks.size(),0);

  if ( hVtx.isValid() && !ignoreVertices_ && !overrideWithDNN_ ) {
    computeMVA(tracks,*hBsp,*hVtx,forest,*mvas);
  } else if ( !overrideWithDNN_ ){
    if ( !ignoreVertices_ ) 
      edm::LogWarning("TrackMVAClassifierBase") << "ignoreVertices is set to False in the configuration, but the vertex collection is not valid"; 
    std::vector<reco::Vertex> vertices;
    computeMVA(tracks,*hBsp,vertices,forest,*mvas);
  } else if ( hVtx.isValid() && !ignoreVertices_ && overrideWithDNN_ ) {
    std::string path = "/afs/cern.ch/work/j/jhavukai/private/LWTNNinCMSSW/CMSSW_9_3_X_2017-09-25-1100/src/Tensorflow_graph";
    tf::Graph graph(path);
    tf::Session session(&graph);
    session.initVariables();
    tf::Shape xShape[] = {1,22};
    tf::Tensor *x = new tf::Tensor(2,xShape);
    tf::Tensor *y = new tf::Tensor();
    computeMVA(tracks,*hBsp,*hVtx,&session,x,y,*mvas);
  } else {
    std::string path = "/afs/cern.ch/work/j/jhavukai/private/LWTNNinCMSSW/CMSSW_9_3_X_2017-09-25-1100/src/Tensorflow_graph";
    tf::Graph graph(path);
    tf::Session	session(&graph);
    session.initVariables();
    tf::Shape xShape[] = {1,12};
    tf::Tensor * x = new tf::Tensor(2,xShape);
    tf::Tensor *y = new	tf::Tensor();
    if ( !ignoreVertices_ )
      edm::LogWarning("TrackMVAClassifierBase") << "ignoreVertices is set to False in the configuration, but the vertex collection is not valid";
    std::vector<reco::Vertex> vertices;
    computeMVA(tracks,*hBsp,vertices,&session,x,y,*mvas);
  }

  assert((*mvas).size()==tracks.size());

  unsigned int k=0;
  for (auto mva : *mvas) {
    (*quals)[k++]
      =  (mva>qualityCuts[0]) << reco::TrackBase::loose
      |  (mva>qualityCuts[1]) << reco::TrackBase::tight
      |  (mva>qualityCuts[2]) << reco::TrackBase::highPurity
     ;

  }
  

  evt.put(std::move(mvas),"MVAValues");
  evt.put(std::move(quals),"QualityMasks");
  
}


#include <TFile.h>
void TrackMVAClassifierBase::beginStream(edm::StreamID) {
  if(!dbFileName_.empty()){
     TFile gbrfile(dbFileName_.c_str());
     forest_.reset((GBRForest*)gbrfile.Get(forestLabel_.c_str()));
  }
}

