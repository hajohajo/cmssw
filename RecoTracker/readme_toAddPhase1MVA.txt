#add the following code when running track reconstruction in order to
#use the phase1 retrained weights

process.load("CondCore.CondDB.CondDB_cfi")
# input database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:./GBRWrapper_13TeV_900pre2.db'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_InitialStep'),
        tag = cms.string('TMVAWeights_InitialStep')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_LowPtQuadStep'),
        tag = cms.string('TMVAWeights_LowPtQuadStep')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_HighPtTripletStep'),
        tag = cms.string('TMVAWeights_HighPtTripletStep')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_LowPtTripletStep'),
        tag = cms.string('TMVAWeights_LowPtTripletStep')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_DetachedQuadStep'),
        tag = cms.string('TMVAWeights_DetachedQuadStep')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_DetachedTripletStep'),
        tag = cms.string('TMVAWeights_DetachedTripletStep')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_MixedTripletStep'),
        tag = cms.string('TMVAWeights_MixedTripletStep')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_PixelLessStep'),
        tag = cms.string('TMVAWeights_PixelLessStep')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_TobTecStep'),
        tag = cms.string('TMVAWeights_TobTecStep')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        label = cms.untracked.string('TMVAWeights_JetCoreRegionalStep'),
        tag = cms.string('TMVAWeights_JetCoreRegionalStep')
      ),
    )
)

