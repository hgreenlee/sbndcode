art_make_library(
                 SOURCE       CRTEventDisplayAlg.cc
                 LIBRARIES
                        larsim::Utils
                        ROOT::Graf3d
                        sbnobj::SBND_CRT
                        sbndcode::CRT_CRTBackTracker
                        lardata::DetectorPropertiesService
)

cet_build_plugin(CRTEventDisplay art::EDAnalyzer
                 LIBRARIES PRIVATE
                 sbndcode::CRT_CRTEventDisplay
)

cet_build_plugin(CRTChannelMappingEventDisplay art::EDAnalyzer
                 LIBRARIES PRIVATE
                 sbndcode::CRT_CRTEventDisplay
)

install_fhicl()
