from RecoTracker.FinalTrackSelectors.lwtnnESProducer_cfi import lwtnnESProducer as _lwtnnESProducer
trackSelectionLwtnn = _lwtnnESProducer.clone(
    ComponentName = "trackSelectionLwtnn",
    fileName = "RecoTracker/FinalTrackSelectors/data/LWTNN_network_v1.json",
)
