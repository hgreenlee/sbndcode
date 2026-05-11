install_fhicl()

cet_build_plugin(CRTAnalysis art::EDAnalyzer
                 LIBRARIES PRIVATE
                 sbndcode::CRT_CRTBackTracker
                 sbndcode::CRT_CRTReco
                 sbnobj::SBND_CRT
                 sbnobj::SBND_Timing
                 larsim::Utils
                 lardata::DetectorClocksService
                 larcore::Geometry_Geometry_service
                 lardataobj::AnalysisBase
                 nug4::ParticleNavigation
                 art_root_io::TFileService_service
                 ROOT::Tree
                 )

cet_build_plugin(CRTRateAnalysis art::EDAnalyzer
                 LIBRARIES PRIVATE
                 sbndcode::Geometry_GeometryWrappers_CRTGeoService_service
                 sbndcode::ChannelMaps_CRT_CRTChannelMapService_service
                 sbnobj::SBND_CRT
                 larcore::Geometry_Geometry_service
                 larcoreobj::geo_vectors
                 lardataobj::Simulation
                 nusimdata::SimulationBase
                 artdaq_core::artdaq-core_Data
                 art_root_io::TFileService_service
                 ROOT::Tree
                 )

cet_build_plugin(CRTTimingAnalysis art::EDAnalyzer
                 LIBRARIES PRIVATE
                 sbndcode::CRT_CRTReco
                 sbnobj::SBND_CRT
                 sbnobj::SBND_Timing
                 larcore::Geometry_Geometry_service
                 lardataobj::RecoBase
                 nusimdata::SimulationBase
                 art_root_io::TFileService_service
                 ROOT::Tree
                 )

cet_build_plugin(ADRIFT art::EDAnalyzer
                 LIBRARIES PRIVATE
                 sbndcode::Geometry_GeometryWrappers_CRTGeoService_service
                 sbndcode::ChannelMaps_CRT_CRTChannelMapService_service
                 sbnobj::SBND_CRT
                 larcore::Geometry_Geometry_service
                 larcoreobj::geo_vectors
                 lardataobj::Simulation
                 nusimdata::SimulationBase
                 artdaq_core::artdaq-core_Data
                 art_root_io::TFileService_service
                 ROOT::Tree
                 ROOT::Graf
                 ROOT::Gpad
                 )

