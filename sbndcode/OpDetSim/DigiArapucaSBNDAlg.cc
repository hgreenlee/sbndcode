#include "sbndcode/OpDetSim/DigiArapucaSBNDAlg.hh"

//------------------------------------------------------------------------------
//--- opdet::simarapucasbndAlg implementation
//------------------------------------------------------------------------------

namespace opdet {

  DigiArapucaSBNDAlg::DigiArapucaSBNDAlg(ConfigurationParameters_t const& config)
    : fParams{config}
    , fSampling(fParams.frequency/ 1000.) //in GHz to cancel with ns
    , fSampling_Daphne(fParams.frequency_Daphne/ 1000.) //in GHz to cancel with ns
    , fXArapucaVUVEff(fParams.XArapucaVUVEff / fParams.larProp->ScintPreScale())
    , fXArapucaVISEff(fParams.XArapucaVISEff / fParams.larProp->ScintPreScale())
    , fADCSaturation(fParams.Baseline + fParams.Saturation * fParams.ADC * fParams.MeanAmplitude)
    , fEngine(fParams.engine)
  {

    if(fXArapucaVUVEff > 1.0001 || fXArapucaVISEff > 1.0001)
      mf::LogWarning("DigiArapucaSBNDAlg")
        << "Quantum efficiency set in fhicl file "
        << fParams.XArapucaVUVEff << " or " << fParams.XArapucaVISEff
        << " seems to be too large!\n"
        << "Final QE must be equal to or smaller than the scintillation "
        << "pre scale applied at simulation time.\n"
        << "Please check this number (ScintPreScale): "
        << fParams.larProp->ScintPreScale();

    std::string fname;
    cet::search_path sp("FW_SEARCH_PATH");
    sp.find_file(fParams.ArapucaDataFile, fname);
    TFile* file = TFile::Open(fname.c_str(), "READ");
    //Note: TPB time now implemented at digitization module for both coated pmts and (x)arapucas
    //OpDetSim/digi_arapuca_sbnd.root updated in sbnd_data (now including the TPB times vector)

    // TPB emission time histogram for visible (x)arapucas
    std::vector<double>* timeTPB_p;
    file->GetObject("timeTPB", timeTPB_p);
    fTimeTPB = std::make_unique<CLHEP::RandGeneral>
      (*fEngine, timeTPB_p->data(), timeTPB_p->size());

    std::vector<double>* TimeXArapucaVUV_p;
    file->GetObject("TimeXArapucaVUV", TimeXArapucaVUV_p);
    fTimeXArapucaVUV = std::make_unique<CLHEP::RandGeneral>
      (*fEngine, TimeXArapucaVUV_p->data(), TimeXArapucaVUV_p->size());

    size_t pulseSize = fParams.PulseLength * fSampling;
    fWaveformSP.resize(pulseSize);

    size_t pulseSize_Daphne = fParams.PulseLength * fSampling_Daphne;
	  fWaveformSP_Daphne.resize(pulseSize_Daphne);
	
    if(fParams.ArapucaSinglePEmodel) {
      mf::LogDebug("DigiArapucaSBNDAlg") << " using testbench pe response";
      TFile* file =  TFile::Open(fname.c_str(), "READ");
      std::vector<double>* SinglePEVec_40ftCable_Daphne;
      std::vector<double>* SinglePEVec_40ftCable_Apsaia;
      file->GetObject("SinglePEVec_40ftCable_Apsaia", SinglePEVec_40ftCable_Apsaia);
      file->GetObject("SinglePEVec_40ftCable_Daphne", SinglePEVec_40ftCable_Daphne);
      fWaveformSP = *SinglePEVec_40ftCable_Apsaia;
      fWaveformSP_Daphne = *SinglePEVec_40ftCable_Daphne;
    }	
  	else{
      mf::LogDebug("DigiArapucaSBNDAlg") << " using ideal pe response";
      Pulse1PE(fWaveformSP,fSampling);
      Pulse1PE(fWaveformSP_Daphne,fSampling_Daphne);
		}

    file->Close();
  } // end constructor

  DigiArapucaSBNDAlg::~DigiArapucaSBNDAlg() {}


  void DigiArapucaSBNDAlg::ConstructWaveform(
    int ch,
    sim::SimPhotons const& simphotons,
    std::vector<short unsigned int>& waveform,
    std::string pdtype,
    bool is_daphne,
    double start_time,
    unsigned n_samples)
  {
    std::vector<double> waves(n_samples, fParams.Baseline);
    CreatePDWaveform(simphotons, start_time, waves, pdtype,is_daphne);
    waveform = std::vector<short unsigned int> (waves.begin(), waves.end());
  }


  void DigiArapucaSBNDAlg::ConstructWaveformLite(
    int ch,
    sim::SimPhotonsLite const& litesimphotons,
    std::vector<short unsigned int>& waveform,
    std::string pdtype,
    bool is_daphne,
    double start_time,
    unsigned n_samples)
  {
    std::vector<double> waves(n_samples, fParams.Baseline);
    std::map<int, int> const& photonMap = litesimphotons.DetectedPhotons;
    CreatePDWaveformLite(photonMap, start_time, waves, pdtype,is_daphne);
    waveform = std::vector<short unsigned int> (waves.begin(), waves.end());
  }


  void DigiArapucaSBNDAlg::CreatePDWaveform(
    sim::SimPhotons const& simphotons,
    double t_min,
    std::vector<double>& wave,
    std::string pdtype,
    bool is_daphne)
  {
    int nCT = 1;
    if(pdtype == "xarapuca_vuv") {
      for(size_t i = 0; i < simphotons.size(); i++) {
        if((CLHEP::RandFlat::shoot(fEngine, 1.0)) < fXArapucaVUVEff) {
          double tphoton = (fTimeXArapucaVUV->fire()) + simphotons[i].Time - t_min;
          if(tphoton < 0.) continue; // discard if it didn't made it to the acquisition
          if(fParams.CrossTalk > 0.0 && (CLHEP::RandFlat::shoot(fEngine, 1.0)) < fParams.CrossTalk) nCT = 2;
          else nCT = 1;
          size_t timeBin = (is_daphne) ? std::floor(tphoton * fSampling_Daphne) : std::floor(tphoton * fSampling);
          if(timeBin < wave.size()) {
						if (!is_daphne) {AddSPE(timeBin, wave, fWaveformSP, nCT);
						}
            else{ AddSPE(timeBin, wave, fWaveformSP_Daphne, nCT);}
          }
        }
      }
    }
    else if(pdtype == "xarapuca_vis") {
      for(size_t i = 0; i < simphotons.size(); i++) {
        if((CLHEP::RandFlat::shoot(fEngine, 1.0)) < fXArapucaVISEff) {
          double tphoton = (CLHEP::RandExponential::shoot(fEngine, fParams.DecayTXArapucaVIS)) + simphotons[i].Time - t_min + fTimeTPB->fire();
          if(tphoton < 0.) continue; // discard if it didn't made it to the acquisition
          if(fParams.CrossTalk > 0.0 && (CLHEP::RandFlat::shoot(fEngine, 1.0)) < fParams.CrossTalk) nCT = 2;
          else nCT = 1;
          size_t timeBin = (is_daphne) ? std::floor(tphoton * fSampling_Daphne) : std::floor(tphoton * fSampling);
          if(timeBin < wave.size()) {
            if (!is_daphne) {AddSPE(timeBin, wave, fWaveformSP, nCT);
            }
            else{ AddSPE(timeBin, wave, fWaveformSP_Daphne, nCT);
            }
          }
        }
      }
    }
    else{
      throw cet::exception("DigiARAPUCASBNDAlg") << "Wrong pdtype: " << pdtype << std::endl;
    }
    if(fParams.BaselineRMS > 0.0) AddLineNoise(wave);
    if(fParams.DarkNoiseRate > 0.0) AddDarkNoise(wave);
    CreateSaturation(wave);
  }


  void DigiArapucaSBNDAlg::CreatePDWaveformLite(
    std::map<int, int> const& photonMap,
    double t_min,
    std::vector<double>& wave,
    std::string pdtype,
    bool is_daphne)
  {
    if(pdtype == "xarapuca_vuv"){
      SinglePDWaveformCreatorLite(fXArapucaVUVEff, fTimeXArapucaVUV, wave, photonMap, t_min,is_daphne);
    }
    else if(pdtype == "xarapuca_vis"){
      // creating the waveforms for xarapuca_vis is different than the rest
      // so there's an overload for that which lacks the timeHisto
      SinglePDWaveformCreatorLite(fXArapucaVISEff, wave, photonMap, t_min,is_daphne);
    }
    else{
      throw cet::exception("DigiARAPUCASBNDAlg") << "Wrong pdtype: " << pdtype << std::endl;
    }
    if(fParams.BaselineRMS > 0.0) AddLineNoise(wave);
    if(fParams.DarkNoiseRate > 0.0) AddDarkNoise(wave);
    CreateSaturation(wave);
  }


  void DigiArapucaSBNDAlg::SinglePDWaveformCreatorLite(
    double effT,
    std::unique_ptr<CLHEP::RandGeneral>& timeHisto,
    std::vector<double>& wave,
    std::map<int, int> const& photonMap,
    double const& t_min,
    bool is_daphne
    )
  {
    // TODO: check that this new approach of not using the last
    // (1-accepted_photons) doesn't introduce some bias
    double meanPhotons;
    size_t acceptedPhotons;
    double tphoton;
    for (auto const& photonMember : photonMap) {
      // TODO: check that this new approach of not using the last
      // (1-accepted_photons) doesn't introduce some bias
      meanPhotons = photonMember.second*effT;
      acceptedPhotons = CLHEP::RandPoissonQ::shoot(fEngine, meanPhotons);
      for(size_t i = 0; i < acceptedPhotons; i++) {
        tphoton = timeHisto->fire();
        tphoton += photonMember.first - t_min;
        if(tphoton < 0.) continue; // discard if it didn't made it to the acquisition
        int nCT=1;
        if(fParams.CrossTalk > 0.0 &&
           (CLHEP::RandFlat::shoot(fEngine, 1.0)) < fParams.CrossTalk) nCT = 2;
          size_t timeBin = (is_daphne) ? std::floor(tphoton * fSampling_Daphne) : std::floor(tphoton * fSampling);
          if(timeBin < wave.size()) {
						if (!is_daphne) {AddSPE(timeBin, wave, fWaveformSP, nCT);
						}
            else{ AddSPE(timeBin, wave, fWaveformSP_Daphne, nCT);}
          }
      }
    }
  }


  void DigiArapucaSBNDAlg::SinglePDWaveformCreatorLite(
    double effT,
    std::vector<double>& wave,
    std::map<int, int> const& photonMap,
    double const& t_min,
    bool is_daphne
    )
  {
    double meanPhotons;
    size_t acceptedPhotons;
    double tphoton;
    int nCT;
    for (auto const& photonMember : photonMap) {
      // TODO: check that this new approach of not using the last
      // (1-accepted_photons) doesn't introduce some bias
      meanPhotons = photonMember.second*effT;
      acceptedPhotons = CLHEP::RandPoissonQ::shoot(fEngine, meanPhotons);
      for(size_t i = 0; i < acceptedPhotons; i++) {
        tphoton = (CLHEP::RandExponential::shoot(fEngine, fParams.DecayTXArapucaVIS));
        tphoton += photonMember.first - t_min;
        if(tphoton < 0.) continue; // discard if it didn't made it to the acquisition
        if(fParams.CrossTalk > 0.0 && (CLHEP::RandFlat::shoot(fEngine, 1.0)) < fParams.CrossTalk) nCT = 2;
        else nCT = 1;
          size_t timeBin = (is_daphne) ? std::floor(tphoton * fSampling_Daphne) : std::floor(tphoton * fSampling);
          if(timeBin < wave.size()) {
						if (!is_daphne) {AddSPE(timeBin, wave, fWaveformSP, nCT);
						}
            else{ AddSPE(timeBin, wave, fWaveformSP_Daphne, nCT);}
          }
      }
    }
  }

  //Ideal single pulse waveform, same shape for both electronics (not realistic: different capacitances, ...) 
  void DigiArapucaSBNDAlg::Pulse1PE(std::vector<double>& fWaveformSP, const double sampling)//TODO: use only one Pulse1PE functions for both electronics ~rodrigoa
  {
    double constT1 = fParams.ADC * fParams.MeanAmplitude;
    for(size_t i = 0; i<fWaveformSP.size(); i++) {
      double time = static_cast<double>(i) / sampling;
      if (time < fParams.PeakTime) fWaveformSP[i] = constT1 * std::exp((time - fParams.PeakTime) / fParams.RiseTime);
      else fWaveformSP[i] = constT1 * std::exp(-(time - fParams.PeakTime) / fParams.FallTime);
    }
  }
  
  void DigiArapucaSBNDAlg::AddSPE(
    const size_t time_bin,
    std::vector<double>& wave,
    const std::vector<double>& fWaveformSP,
    const int nphotons) //adding single pulse //TODO: use only one function, use pulsize and fWaveformSP as arguments instead ~rodrigoa
  {
    // if(time_bin > wave.size()) return;
    size_t max = time_bin + fWaveformSP.size() < wave.size() ? time_bin + fWaveformSP.size() : wave.size();
    auto min_it = std::next(wave.begin(), time_bin);
    auto max_it = std::next(wave.begin(), max);
    std::transform(min_it, max_it,
                   fWaveformSP.begin(), min_it,
                   [nphotons](double w, double ws) -> double {
                     return w + ws*nphotons  ; });
  }

  void DigiArapucaSBNDAlg::CreateSaturation(std::vector<double>& wave)
  {
    std::replace_if(wave.begin(), wave.end(),
                    [&](auto w){return w > fADCSaturation;}, fADCSaturation);
  }


  void DigiArapucaSBNDAlg::AddLineNoise(std::vector< double >& wave)
  {
    // TODO: after running the profiler I can see that this is where
    // most cycles are being used.  Potentially some improvement could
    // be achieved with the below code but the array dynamic allocation
    // is not helping. The function would need to have it passed.
    // ~icaza
    //
    // double *array = new double[wave.size()]();
    // CLHEP::RandGaussQ::shootArray(fEngine, wave.size(), array, 0, fParams.BaselineRMS);
    // for(size_t i = 0; i<wave.size(); i++) {
    //   wave[i] += array[i];
    // }
    // delete array;
    //
    std::transform(wave.begin(), wave.end(), wave.begin(),
                   [this](double w) -> double {
                     return w + CLHEP::RandGaussQ::shoot(fEngine, 0, fParams.BaselineRMS) ; });
  }


  void DigiArapucaSBNDAlg::AddDarkNoise(std::vector<double>& wave)
  {
    int nCT;
    // Multiply by 10^9 since fDarkNoiseRate is in Hz (conversion from s to ns)
    double mean = 1000000000.0 / fParams.DarkNoiseRate;
    double darkNoiseTime = CLHEP::RandExponential::shoot(fEngine, mean);
    while(darkNoiseTime < wave.size()) {
      size_t timeBin = std::round(darkNoiseTime);
      if(fParams.CrossTalk > 0.0 && (CLHEP::RandFlat::shoot(fEngine, 1.0)) < fParams.CrossTalk) nCT = 2;
      else nCT = 1;
      if(timeBin < wave.size()) AddSPE(timeBin, wave, fWaveformSP, nCT);
      // Find next time to add dark noise
      darkNoiseTime += CLHEP::RandExponential::shoot(fEngine, mean);
    }//while
  }


  double DigiArapucaSBNDAlg::FindMinimumTime(sim::SimPhotons const& simphotons)
  {
    double t_min = 1e15;
    for(size_t i = 0; i < simphotons.size(); i++) {
      if(simphotons[i].Time < t_min) t_min = simphotons[i].Time;
    }
    return t_min;
  }


  double DigiArapucaSBNDAlg::FindMinimumTimeLite(std::map<int, int> const& photonMap)
  {
    for (auto const& mapMember : photonMap) {
      if(mapMember.second != 0) return (double)mapMember.first;
    }
    return 1e5;
  }


  // -----------------------------------------------------------------------------
  // // ---  opdet::DigiArapucaSBNDAlgMaker
  // // -----------------------------------------------------------------------------
  DigiArapucaSBNDAlgMaker::DigiArapucaSBNDAlgMaker
  (Config const& config)
  {
    // settings
    fBaseConfig.ADC                   = config.voltageToADC();
    fBaseConfig.Baseline              = config.baseline();
    fBaseConfig.Saturation            = config.saturation();
    fBaseConfig.XArapucaVUVEff        = config.xArapucaVUVEff();
    fBaseConfig.XArapucaVISEff        = config.xArapucaVISEff();
    fBaseConfig.RiseTime              = config.riseTime();
    fBaseConfig.FallTime              = config.fallTime();
    fBaseConfig.MeanAmplitude         = config.meanAmplitude();
    fBaseConfig.DarkNoiseRate         = config.darkNoiseRate();
    fBaseConfig.BaselineRMS           = config.baselineRMS();
    fBaseConfig.CrossTalk             = config.crossTalk();
    fBaseConfig.PulseLength           = config.pulseLength();
    fBaseConfig.PeakTime              = config.peakTime();
    fBaseConfig.DecayTXArapucaVIS     = config.decayTXArapucaVIS();
    fBaseConfig.ArapucaDataFile       = config.arapucaDataFile();
    fBaseConfig.ArapucaSinglePEmodel  = config.ArapucasinglePEmodel();
    fBaseConfig.frequency_Daphne      = config.DaphneFrequency();
  }

  std::unique_ptr<DigiArapucaSBNDAlg> DigiArapucaSBNDAlgMaker::operator()(
    detinfo::LArProperties const& larProp,
    detinfo::DetectorClocksData const& clockData,
    CLHEP::HepRandomEngine* engine
    ) const
  {
    // set the configuration
    auto params = fBaseConfig;

    // set up parameters
    params.larProp = &larProp;
    params.frequency = clockData.OpticalClock().Frequency();
    params.frequency_Daphne = fBaseConfig.frequency_Daphne; //Mhz
    params.engine = engine;

    return std::make_unique<DigiArapucaSBNDAlg>(params);
  } // DigiArapucaSBNDAlgMaker::create()

}
