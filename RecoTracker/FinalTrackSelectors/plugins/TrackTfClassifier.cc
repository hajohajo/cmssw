#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "getBestVertex.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

namespace {
	struct tfDnn {
		tfDnn(const edm::ParameterSet& cfg):
			tfDnnLabel_(cfg.getParameter<std::string>("tfDnnLabel"))
		{}

		static const char *name() { return "TrackTfClassifier"; }

		static void fillDescriptions(edm::ParameterSetDescription& desc) {
			desc.add<std::string>("tfDnnLabel", "trackSelectionTf");
		}

		//Used to clamp each input value within a range that was seen in the training process.
    const std::vector<float> upperThresholds = std::vector<float>{2.056, 2.469, 0.76, 7.599, 0.761, 0.314, 0.083, 0.018, 0.167, 0.558, 2.943, 41.0, 1.0, 6.0, 19.0};
    const std::vector<float> lowerThresholds = std::vector<float>{0.201, -2.47, -0.771, -7.757, -0.77, -0.313, 0.004, 0.001, 0.005, 0.008, 0.144, 1.0, 0.0, 0.0, 0.0};
    float clampValue(float value, int index) const {
      return (float)std::min(std::max(value, lowerThresholds.at(index)), upperThresholds.at(index));
    }

		void beginStream() {}
		void initEvent(const edm::EventSetup& es) {
			edm::ESHandle<tensorflow::Session> tfDnnHandle;
			es.get<TrackingComponentsRecord>().get(tfDnnLabel_, tfDnnHandle);
      session_ = const_cast<tensorflow::Session*>(tfDnnHandle.product());
		}

		float operator()(reco::Track const & trk,
                     reco::BeamSpot const & beamSpot,
                     reco::VertexCollection const & vertices) const { 

			Point bestVertex = getBestVertex(trk, vertices);

			tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, 16});

      input.matrix<float>()(0, 0) = clampValue(trk.pt(), 0);
      input.matrix<float>()(0, 1) = clampValue(trk.eta(), 1);
      input.matrix<float>()(0, 2) = clampValue(trk.dxy(beamSpot.position()), 2); // Training done without taking absolute value
      input.matrix<float>()(0, 3) = clampValue(trk.dz(beamSpot.position()), 3); // Training done without taking absolute value
      input.matrix<float>()(0, 4) = clampValue(trk.dxy(bestVertex), 4); // Training done without taking absolute value
      input.matrix<float>()(0, 5) = clampValue(trk.dz(bestVertex), 5); // Training done without taking absolute value
      input.matrix<float>()(0, 6) = clampValue(trk.ptError(), 6);
      input.matrix<float>()(0, 7) = clampValue(trk.etaError(), 7);
      input.matrix<float>()(0, 8) = clampValue(trk.dxyError(), 8);
      input.matrix<float>()(0, 9) = clampValue(trk.dzError(), 9);
      input.matrix<float>()(0, 10) = clampValue(trk.normalizedChi2(), 10);
      input.matrix<float>()(0, 11) = clampValue(trk.ndof(), 11);
      input.matrix<float>()(0, 12) = clampValue(trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS), 12);
      input.matrix<float>()(0, 13) = clampValue(trk.hitPattern().numberOfValidPixelHits(), 13);
      input.matrix<float>()(0, 14) = clampValue(trk.hitPattern().numberOfValidStripHits(), 14);
      input.matrix<float>()(0, 15) = trk.algo();

      std::vector<tensorflow::Tensor> outputs;
      tensorflow::run(session_, { {"x", input} }, { "Identity" }, {}, &outputs);

      float output = 2.0*outputs[0].matrix<float>()(0, 0)-1.0;

			return output;
		}

		std::string tfDnnLabel_;
		tensorflow::Session *session_;
	};

	using TrackTfClassifier = TrackMVAClassifier<tfDnn>;
}
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackTfClassifier);
