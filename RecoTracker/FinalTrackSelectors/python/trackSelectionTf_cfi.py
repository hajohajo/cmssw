from RecoTracker.FinalTrackSelectors.tfESProducer_cfi import tfESProducer as _tfESProducer
trackSelectionTf = _tfESProducer.clone(
    ComponentName = "trackSelectionTf",
    fileName = "RecoTracker/FinalTrackSelectors/data/frozen_graph_v2.pb",
)
