#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class TfESProducer: public edm::ESProducer {
public:
	TfESProducer(const edm::ParameterSet& iConfig);
	~TfESProducer() override = default;

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<tensorflow::Session> produce(const TrackingComponentsRecord& iRecord);

private:
	edm::FileInPath fileName_;
};

TfESProducer::TfESProducer(const edm::ParameterSet& iConfig):
	fileName_(iConfig.getParameter<edm::FileInPath>("fileName"))
{
	auto componentName = iConfig.getParameter<std::string>("ComponentName");
	setWhatProduced(this, componentName);
}

void TfESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	edm::ParameterSetDescription desc;
	desc.add<std::string>("ComponentName", "tfESProducer");
	desc.add<edm::FileInPath>("fileName", edm::FileInPath());
	descriptions.add("tfESProducer", desc);
}

std::unique_ptr<tensorflow::Session> TfESProducer::produce(const TrackingComponentsRecord& iRecord) {
	std::atomic<tensorflow::GraphDef*> graphDef = tensorflow::loadGraphDef(fileName_.fullPath().c_str());
	return (std::unique_ptr<tensorflow::Session>)tensorflow::createSession(graphDef);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(TfESProducer);
