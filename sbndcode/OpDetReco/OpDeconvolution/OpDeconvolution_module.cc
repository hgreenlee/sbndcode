////////////////////////////////////////////////////////////////////////
// Class:       OpDeconvolution
// Plugin Type: producer (art v3_06_03)
// File:        OpDeconvolution_module.cc
//
// Generated at Fri Jun 11 05:27:39 2021 by Francisco Nicolas-Arnaldos using cetskelgen
// from cetlib version v3_11_01.
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"
#include "art_root_io/TFileService.h"

#include "lardata/DetectorInfoServices/DetectorClocksService.h"
#include "lardata/DetectorInfoServices/LArPropertiesService.h"
#include "lardataobj/Simulation/sim.h"

#include <memory>

#include "CLHEP/Random/RandGaussQ.h"

#include "lardataobj/RawData/OpDetWaveform.h"
#include "lardata/Utilities/LArFFT.h"
#include "TFile.h"

//Baseline estimation
#include "larana/OpticalDetector/OpHitFinder/OpticalRecoTypes.h"
#include "larana/OpticalDetector/OpHitFinder/PMTPedestalBase.h"
#include "larana/OpticalDetector/OpHitFinder/PedAlgoEdges.h"
#include "larana/OpticalDetector/OpHitFinder/PedAlgoRollingMean.h"
#include "larana/OpticalDetector/OpHitFinder/PedAlgoUB.h"

//PDS map
#include "sbndcode/OpDetSim/sbndPDMapAlg.hh"
#include <cmath>

#include "TCanvas.h"
#include "TH1.h"
#include "TComplex.h"



namespace opreco {
  class OpDeconvolution;
}


class opreco::OpDeconvolution : public art::EDProducer {
public:
  explicit OpDeconvolution(fhicl::ParameterSet const& p);
  // The compiler-generated destructor is fine for non-base
  // classes without bare pointers or other resource use.

  // Plugins should not be copied or assigned.
  OpDeconvolution(OpDeconvolution const&) = delete;
  OpDeconvolution(OpDeconvolution&&) = delete;
  OpDeconvolution& operator=(OpDeconvolution const&) = delete;
  OpDeconvolution& operator=(OpDeconvolution&&) = delete;

  // Required functions.
  void produce(art::Event& e) override;

  // Selected optional functions.
  void beginJob() override;
  void endJob() override;

private:
  std::string fOpDaqModuleLabel;
  bool fDebug;
  bool fSaveDecoWaveforms;
  int fFFTSizePow;
  std::vector<double> fSinglePEWave;
  bool fApplyExpoAvSmooth;
  bool fApplyUnAvSmooth;
  float fExpoAvSmoothPar;
  short unsigned int fUnAvNeighbours;
  double fHypoSignalTauFast;
  bool fCustomHypoSignalTauFast;
  double fBaselineRMS;
  double fHypoSignalScale;
  double fPMTChargeToADC;
  double fDecoWaveformPrecision;
  bool fUseBaselineMode;
  double fBaselineMinADC, fBaselineMaxADC;


  double fNormUnAvSmooth;
  double fSamplingFreq;
  size_t nbinsFFT;

  //PDS mapping
  opdet::sbndPDMapAlg pdsmap;
  unsigned int NDecoWf;

  //Random Numbers
  std::unique_ptr<CLHEP::RandGauss>  fCLHEPEGauss;

  std::vector<double> fSignalHypothesis;
  std::vector<double> fNoiseHypothesis;
  std::vector<TComplex> fSERfft;
  std::vector<double> fNSR, fSERfft_abs2;//fHypothesisfft_abs2, fNoisefft_abs2;

  // Declare member data here.

  // Declare member functions
  void ApplyExpoAvSmoothing(std::vector<double>& wf);
  void ApplyUnAvSmoothing(std::vector<double>& wf);
  size_t WfSizeFFT(size_t n);
  std::vector<double> CreateWienerHypothesis(size_t n, detinfo::LArProperties const& lar_prop);
  std::vector<double> CreateNoiseHypothesis(size_t n);
  void SubstractBaseline(const std::vector<raw::ADC_Count_t>& wf, std::vector<double>& wf_s);

  //Baseline estimation algorithm
  pmtana::PMTPedestalBase* fBaselineAlg;

  //Load TFileService serrvice
  art::ServiceHandle<art::TFileService> tfs;
  //Load FFT serrvice
  art::ServiceHandle<util::LArFFT> fft_service;
};


opreco::OpDeconvolution::OpDeconvolution(fhicl::ParameterSet const& p)
  : EDProducer{p}  // ,
  // More initializers here.
{
  // Call appropriate produces<>() functions here.
  // Call appropriate consumes<>() for any products to be retrieved by this module.

  produces< std::vector< raw::OpDetWaveform > >();

  std::string fname;
  cet::search_path sp("FW_SEARCH_PATH");
  sp.find_file("OpDetSim/digi_pmt_sbnd.root", fname);
  TFile* file = TFile::Open(fname.c_str(), "READ");


  //read fhicl paramters
  fOpDaqModuleLabel = p.get< std::string >("OpDaqModuleLabel");
  fDebug = p.get< bool >("Debug");
  fSaveDecoWaveforms = p.get< bool >("SaveDecoWaveforms");
  fFFTSizePow = p.get< int >("FFTSizePow");
  fApplyExpoAvSmooth   = p.get< bool >("ApplyExpoAvSmooth");
  fApplyUnAvSmooth   = p.get< bool >("ApplyUnAvSmooth");
  fExpoAvSmoothPar = p.get< float >("ExpoAvSmoothPar");
  fUnAvNeighbours = p.get< short unsigned int >("UnAvNeighbours");
  fCustomHypoSignalTauFast = p.get< bool >("CustomHypoSignalTauFast");
  fHypoSignalTauFast = p.get< double >("HypoSignalTauFast");
  fBaselineRMS = p.get< double >("BaselineRMS");
  fHypoSignalScale = p.get< double >("HypoSignalScale");
  fPMTChargeToADC = p.get< double >("PMTChargeToADC");
  fDecoWaveformPrecision = p.get< double >("DecoWaveformPrecision");
  fUseBaselineMode = p.get< bool >("UseBaselineMode");


  fNormUnAvSmooth=1./(2*fUnAvNeighbours+1);

  //Initizalize random numbers
  auto& engine = createEngine(sim::GetRandomNumberSeed());
  fCLHEPEGauss = std::make_unique<CLHEP::RandGauss>(engine);

  NDecoWf=0;

  auto const clockData = art::ServiceHandle<detinfo::DetectorClocksService const>()->DataForJob();
  fSamplingFreq=clockData.OpticalClock().Frequency()/1000.;
  auto const* lar_prop = lar::providerFrom<detinfo::LArPropertiesService>();

  if(fUseBaselineMode){
    auto const baselinemode_pset = p.get< fhicl::ParameterSet >("BaselineModePset");
    fBaselineMinADC = baselinemode_pset.get< double >("MinADC");
    fBaselineMaxADC = baselinemode_pset.get< double >("MaxADC");
  }
  else{
    //Initizalize baseline estimator (stolen from "sbndcode/OpDetReco/OpHit/SBNDOpHitFinder_module.cc)
    auto const baselinealg_pset = p.get< fhicl::ParameterSet >("DecoBaselinePset");
    std::string baselinealg_pset_name = baselinealg_pset.get< std::string >("Name");
    if(baselinealg_pset_name == "Edges")
      fBaselineAlg = new pmtana::PedAlgoEdges(baselinealg_pset);
    else if (baselinealg_pset_name == "RollingMean")
      fBaselineAlg = new pmtana::PedAlgoRollingMean(baselinealg_pset);
    else if (baselinealg_pset_name == "UB"   )
      fBaselineAlg = new pmtana::PedAlgoUB(baselinealg_pset);
    else throw art::Exception(art::errors::UnimplementedFeature)
      << "Cannot find implementation for "<< baselinealg_pset_name << " algorithm.\n";
  }



  nbinsFFT=std::pow(2, fFFTSizePow);
  fSignalHypothesis = CreateWienerHypothesis(nbinsFFT, *lar_prop);
  std::cout<<"Creating light signal hypothesis size"<<fSignalHypothesis.size()<<std::endl;

  fNoiseHypothesis = CreateNoiseHypothesis(nbinsFFT);
  std::cout<<"Creating noise size"<<fNoiseHypothesis.size()<<std::endl;
  std::vector<double>* SinglePEVec_p;

  file->GetObject("SinglePEVec", SinglePEVec_p);
  fSinglePEWave = *SinglePEVec_p;
  while(fSinglePEWave.size()<nbinsFFT)
    fSinglePEWave.push_back(0);
  std::cout<<"Creating SER size:"<<fSinglePEWave.size()<<std::endl;

  std::vector<TComplex> noise_fft, hypothesis_fft;
  fft_service->ReinitializeFFT (nbinsFFT, "", 20);
  noise_fft.resize(nbinsFFT);
  fSERfft.resize(nbinsFFT);
  hypothesis_fft.resize(nbinsFFT);
  fft_service->DoFFT(fSinglePEWave, fSERfft);
  fft_service->DoFFT(fNoiseHypothesis, noise_fft);
  fft_service->DoFFT(fSignalHypothesis, hypothesis_fft);
  for(size_t k=0; k<nbinsFFT; k++){
    //fHypothesisfft_abs2.push_back( std::pow(TComplex::Abs(hypothesis_fft[k]),2) );
    //fNoisefft_abs2.push_back( std::pow(TComplex::Abs(noise_fft[k]),2) );
    fNSR.push_back( 1.*std::pow(TComplex::Abs(noise_fft[k]),2)/std::pow(TComplex::Abs(hypothesis_fft[k]),2) );
    fSERfft_abs2.push_back( std::pow(TComplex::Abs(fSERfft[k]),2) );
  }
}




void opreco::OpDeconvolution::produce(art::Event& e)
{
  int eventID = e.id().event();
  //Load the waveforms
  art::Handle< std::vector< raw::OpDetWaveform > > wfHandle;
  e.getByLabel(fOpDaqModuleLabel, wfHandle);
  if (!wfHandle.isValid()) {
    std::cout<<"Non valid waveform handle\n";
    return;
  }

  //Createing deconvolved waveforms
  std::unique_ptr< std::vector< raw::OpDetWaveform > > DecoWf_VecPtr(std::make_unique< std::vector< raw::OpDetWaveform > > ());

  for(auto const& wf : *wfHandle)
  {
    // Deconvolve only PMTs
    if (pdsmap.pdType(wf.ChannelNumber())!="pmt_coated" && pdsmap.pdType(wf.ChannelNumber())!="pmt_uncoated") continue;
    //Check this
    if(wf.Waveform().size()>nbinsFFT) {
      std::cout<<" Wf size: "<<wf.Waveform().size()<<"   event="<<eventID<<" OpCh="<<wf.ChannelNumber()<<".. skipping\n";
      continue;
    }

    //Substract baseline
    std::vector<double> wave;
    SubstractBaseline( wf.Waveform(), wave);
    double minADC=*min_element(wave.begin(), wave.end());
    double scaling_factor=fHypoSignalScale*(-minADC)/fPMTChargeToADC;

    //Apply smoothing
    if(fApplyExpoAvSmooth)
      ApplyExpoAvSmoothing(wave);
    if(fApplyUnAvSmooth)
      ApplyUnAvSmoothing(wave);


    //Note: only nbinsFFT/2 bins used in deconvolution, optimize memory?
    TComplex kerinit(0,0,false);
    double scaling_factor2=1./(scaling_factor*scaling_factor);
    std::vector<TComplex> kernel(nbinsFFT,kerinit);
    for(size_t k=0; k<nbinsFFT/2; k++){
      double den = fSERfft_abs2[k] + fNSR[k] * scaling_factor2;
      kernel[k]= TComplex::Conjugate( fSERfft[k] ) / den;
      //std::cout<< k<<":"<<ser_fft[k]<<":"<<hypothesis_fft[k]<<":"<<noise_fft[k]<<":"<<num/den << " ";
    }

    //Deconvolve raw signal (covolve with kernel)
    fft_service->ReinitializeFFT (nbinsFFT, "", 20);
    //Padding with 0's
    wave.resize(nbinsFFT,0.);
    //fft_service->Convolute(rawsignal, kernel);
    fft_service->Convolute(wave, kernel);


    //Prepare deconvolved waveform before saving
    double fDecoWfScaleFactor=1./fDecoWaveformPrecision;
    //std::transform(rawsignal.begin(), rawsignal.end(), rawsignal.begin(), [fDecoWfScaleFactor](double &dec){ return fDecoWfScaleFactor*dec; } );
    std::transform(wave.begin(), wave.end(), wave.begin(), [fDecoWfScaleFactor](double &dec){ return fDecoWfScaleFactor*dec; } );

    //Debbuging and save wf in hist file
    if(fDebug){
      std::cout<<"\n.....Debbuging.....\n";
      auto minADC_ix=min_element(wave.begin(), wave.end());
      std::cout<<std::endl<<"Stamp="<<wf.TimeStamp()<<" OpCh"<<wf.ChannelNumber()<<" MinADC="<<minADC<<" (";
      std::cout<<minADC_ix-wave.begin()<<") Size="<<wf.Waveform().size()<<" ScFactor="<<scaling_factor<<std::endl;
      if(wave.size()>6000){
        std::string name="h_raw"+std::to_string(eventID)+"_"+std::to_string(wf.ChannelNumber())+"_"+std::to_string(wf.TimeStamp());
        TH1F * h_raw = tfs->make< TH1F >(name.c_str(),";Bin;ADC", nbinsFFT, 0, nbinsFFT);
        name="h_deco"+std::to_string(eventID)+"_"+std::to_string(wf.ChannelNumber())+"_"+std::to_string(wf.TimeStamp());
        TH1F * h_deco = tfs->make< TH1F >(name.c_str(),";Bin;#PE", nbinsFFT, 0, nbinsFFT);

        for(size_t k=0; k<wave.size(); k++){
          //if(fDebug) std::cout<<k<<":"<<wave[k]<<":"<<rawsignal[k]<<"  ";
          h_deco->Fill(k, wave[k]);
        }
        for(size_t k=0; k<wf.Waveform().size(); k++){
          //if(fDebug) std::cout<<k<<":"<<wave[k]<<":"<<rawsignal[k]<<"  ";
          h_raw->Fill(k, wf.Waveform()[k]);
        }
      }
    }

    //Saving deconvolved waveform
    if(fSaveDecoWaveforms){
      //raw::OpDetWaveform decowf(wf.TimeStamp(), wf.ChannelNumber(), std::vector<short unsigned int> (rawsignal.begin(), std::next(rawsignal.begin(), wf.Waveform().size())) );
      raw::OpDetWaveform decowf(wf.TimeStamp(), wf.ChannelNumber(), std::vector<short unsigned int> (wave.begin(),  std::next(wave.begin(), wf.Waveform().size()) ) );
      DecoWf_VecPtr->push_back(decowf);
    }
    NDecoWf++;
  }

  //save object in event
  e.put( std::move(DecoWf_VecPtr) );
}

void opreco::OpDeconvolution::beginJob()
{
  // Implementation of optional member function here.
}

void opreco::OpDeconvolution::endJob()
{
  // Implementation of optional member function here.
}

void opreco::OpDeconvolution::ApplyExpoAvSmoothing(std::vector<double>& wf){
  std::transform (std::next(wf.begin(), 1), wf.end(), wf.begin(), std::next(wf.begin(), 1),
    [&](double _x, double _y) { return  fExpoAvSmoothPar*_x+ (1. - fExpoAvSmoothPar)*_y; }  );
}

void opreco::OpDeconvolution::ApplyUnAvSmoothing(std::vector<double>& wf){
  std::vector<double> wf_aux(wf.begin(), wf.end());
  for(size_t bin=fUnAvNeighbours; bin<wf.size()-fUnAvNeighbours; bin++){
    double sum=0.;
    for(size_t nbin=bin-fUnAvNeighbours; nbin<=bin+fUnAvNeighbours; nbin++)
      sum+=wf_aux[nbin];
    //std::cout<<bin<<" "<<sum<<" "<<sum*fNormUnAvSmooth<<std::endl;
    wf[bin]=sum*fNormUnAvSmooth;
  }
}

size_t opreco::OpDeconvolution::WfSizeFFT(size_t n){
  size_t cont=0;
  while(n>0){
    cont++;
    n=(n>>1);
    //std::cout<<"JE "<<n<<std::endl;
  }
  //std::cout<<"Cont="<<cont<<std::endl;
  return pow(2, cont);
}

std::vector<double> opreco::OpDeconvolution::CreateWienerHypothesis(size_t n, detinfo::LArProperties const& lar_prop){
  double SlowFast_Fraction = lar_prop.ScintYieldRatio();
  double TSlow = lar_prop.ScintSlowTimeConst();
  double TFast;
  if(fCustomHypoSignalTauFast)
    TFast=fHypoSignalTauFast;
  else
    TFast=lar_prop.ScintFastTimeConst();
  std::cout<<"HypoPar: "<<SlowFast_Fraction<<" "<<TSlow<<" "<<TFast<<" "<<fSamplingFreq<<std::endl;
  std::vector<double> v;
  double t;
  for(size_t k=0; k<n; k++){
    t = (double)(k) / fSamplingFreq; //in ns
    v.push_back( SlowFast_Fraction*std::exp(-1.*t/TFast) + (1.-SlowFast_Fraction)*std::exp(-1.*t/TSlow) );
    //std::cout<< "Hypo: "<<v[k]<<std::endl;
  }
  return v;
}

/*std::vector<short int> opreco::OpDeconvolution::CreateNoiseHypothesis(size_t n){
  std::vector<short int> v;
  for(size_t k=0; k<n; k++){
    std::cout<<k<<" "<<(short int) fCLHEPEGauss->fire(0, 1)<<std::endl;
    v.push_back( (short int)fCLHEPEGauss->fire(0, 1) );
  }
  return v;
}*/

std::vector<double> opreco::OpDeconvolution::CreateNoiseHypothesis(size_t n){
  std::vector<double> v;
  for(size_t k=0; k<n; k++){
    //std::cout<<"Noise "<<k<<" "<<fCLHEPEGauss->fire(0, fBaselineRMS)<<std::endl;
    v.push_back( fCLHEPEGauss->fire(0, fBaselineRMS) );
  }
  return v;
}

void opreco::OpDeconvolution::SubstractBaseline(const std::vector<raw::ADC_Count_t>& wf, std::vector<double>& wf_s){
  wf_s.resize(wf.size());
  if(fUseBaselineMode){
    TH1F h_ba = TH1F("",";;", (int)(fBaselineMaxADC-fBaselineMinADC), fBaselineMinADC-0.5, fBaselineMaxADC-0.5);
    for(auto &adc:wf)
      h_ba.Fill(adc);
    double _baseline=h_ba.GetXaxis()->GetBinCenter(h_ba.GetMaximumBin());
    for(size_t k=0; k<wf.size(); k++)
      wf_s[k]=(double)wf[k]-_baseline;
    if(fDebug){
      std::string name="h_baseline"+std::to_string((int)fCLHEPEGauss->fire(0, 10000))+"_"+std::to_string(NDecoWf);
      TH1F * hs_ba = tfs->make< TH1F >(name.c_str(),"Baseline;ADC;# entries",(int)(fBaselineMaxADC-fBaselineMinADC), fBaselineMinADC-0.5, fBaselineMaxADC-0.5);
      for(int k=1; k<=h_ba.GetNbinsX(); k++)
        hs_ba->SetBinContent(k, h_ba.GetBinContent(k));
      std::cout<<"   -- Estimating baseline (mode algorithm): "<<_baseline<<std::endl;
    }
  }
  else{
    if(fDebug)
      std::cout<<"   -- Estimating baseline (larana algorithm)"<<std::endl;
    fBaselineAlg->Evaluate(wf);
    for(size_t k=0; k<wf.size(); k++)
      wf_s[k] = (double)wf[k]-fBaselineAlg->Mean()[k];
  }

  return;
}




DEFINE_ART_MODULE(opreco::OpDeconvolution)


/*std::vector<double> foo;
for (int i=0; i<5; i++){
  foo.push_back (i*10.);
}
std::cout<<"\n";
for(auto & a: foo)
  std::cout<<" A="<<a<<" ";

ApplyExpoAvSmoothing(foo);

std::cout<<"\n";
for(auto & a: foo)
  std::cout<<" B="<<a<<" ";

ApplyUnAvSmoothing(foo);

std::cout<<"\n";
for(auto & a: foo)
  std::cout<<" C="<<a<<" ";*/






/*int wf_size=2048;
std::cout<<"IN 1 \n";
art::ServiceHandle<util::LArFFT> fft_service;
int fftSize = fft_service->FFTSize();
std::cout<<fftSize<<std::endl;
fft_service->ReinitializeFFT (wf_size, "", 20);
fftSize = fft_service->FFTSize();
std::cout<<fftSize<<std::endl;
// Implementation of required member function here.
std::vector<double> wave;
for(int k=0; k<fftSize; k++){
  wave.push_back(17);
}

std::vector<double> ser, signal;
for(int k=0; k<fftSize; k++){
  if( ( (size_t)k )<fSinglePEWave.size() )
    ser.push_back(fSinglePEWave[k]);
  else
    ser.push_back(0);
  signal.push_back(0);
}
//ser[4]=1; ser[5]=2; ser[6]=1;
//signal[8]=1;
signal[49]=10;
signal[48]=6;
signal[50]=3;




for(size_t k=0; k<signal.size(); k++) {
  std::cout<<k<<" TrSig="<<signal[k]<<" "<<ser[k]<<std::endl;
}




fft_service->Convolute(signal, ser);

std::vector<double> deco;
for(size_t k=0; k<signal.size(); k++) {
  deco.push_back( signal[k] );
  std::cout<<k<<" Conv="<<signal[k]<<" "<<ser[k]<<std::endl;
}


fft_service->Deconvolute(deco, ser);

for(size_t k=0; k<deco.size(); k++) {
  std::cout<<k<<" Deco="<<deco[k]<<" "<<ser[k]<<std::endl;
}*/


/*double sum=0, sum2=0, nbins=0;
std::cout<<"\n PRE:\n";
if(wave.size()>2001){
  for(size_t k=4000; k<4050; k++){
    std::cout<<k<<":"<<wave[k]<<"  ";
    sum+=wave[k];
    sum2+=wave[k]*wave[k];
    nbins++;
  }
  std::cout<<"\nBias="<<sum/nbins<<" StdDev="<<sqrt( sum2/nbins - pow(sum/nbins,2) );
}*/

/*for(size_t j=0; j<wave.size(); j++){
  wave[j]=wave[j]-fBaselineAlg->Mean()[j];
  //std::cout<<wave[j]<<" "<<fBaselineAlg->Mean()[j]<<std::endl;
}
//std::transform(wave.begin(), wave.end(), fBaselineAlg->Mean().begin(), std::back_inserter(wave),[](double rawadc, double baseline) {return rawadc-baseline; });


minADC=*min_element(wave.begin(), wave.end());
minADC_ix=min_element(wave.begin(), wave.end());
size_t minADCindex = minADC_ix - wave.begin();
std::cout<<"New minADC="<<minADC<<" Baseline="<<fBaselineAlg->Mean()[minADCindex]<<std::endl;*/


/*std::vector<double> rawsignal;
for(size_t k=0; k<nbinsFFT; k++){
  if( ( (size_t)k )<wave.size() )
    rawsignal.push_back(wave[k]);
  else
    rawsignal.push_back(0);
}*/
