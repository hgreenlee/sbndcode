#ifndef SBNDOPDECONVOLUTION_H
#define SBNDOPDECONVOLUTION_H

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

#include "CLHEP/Random/JamesRandom.h"
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

#include <cmath>
#include "TCanvas.h"
#include "TH1.h"
#include "TComplex.h"

#include "nurandom/RandomUtils/NuRandomService.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"


namespace opdet {
  class OpDeconvolution;
}


class opdet::OpDeconvolution {
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
  std::vector<raw::OpDetWaveform> RunDeconvolution(std::vector<raw::OpDetWaveform> wfHandle);


private:
  bool fDebug;
  int fFFTSizePow;
  std::vector<double> fSinglePEWave;
  bool fApplyExpoAvSmooth;
  bool fApplyUnAvSmooth;
  float fExpoAvSmoothPar;
  short unsigned int fUnAvNeighbours;
  double fHypoSignalTauFast;
  bool fCustomHypoSignalTauFast;
  double fHypoSignalTimeWindow;
  double fBaselineRMS;
  double fHypoSignalScale;
  double fPMTChargeToADC;
  double fDecoWaveformPrecision;
  double fBaselineMinADC, fBaselineMaxADC;
  std::string fOpDetDataFile;
  std::string fFilter;
  bool fScaleHypoSignal;

  double fNormUnAvSmooth;
  double fSamplingFreq;
  size_t nbinsFFT;

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
  void SubstractBaseline(std::vector<double>& wf);

  //Load TFileService serrvice
  art::ServiceHandle<art::TFileService> tfs;
  //Load FFT serrvice
  art::ServiceHandle<util::LArFFT> fft_service;
};


opdet::OpDeconvolution::OpDeconvolution(fhicl::ParameterSet const& p)
{
  //read fhicl paramters
  fDebug = p.get< bool >("Debug");
  fFFTSizePow = p.get< int >("FFTSizePow");
  fApplyExpoAvSmooth   = p.get< bool >("ApplyExpoAvSmooth");
  fApplyUnAvSmooth   = p.get< bool >("ApplyUnAvSmooth");
  fExpoAvSmoothPar = p.get< float >("ExpoAvSmoothPar");
  fUnAvNeighbours = p.get< short unsigned int >("UnAvNeighbours");
  fCustomHypoSignalTauFast = p.get< bool >("CustomHypoSignalTauFast");
  fHypoSignalTauFast = p.get< double >("HypoSignalTauFast");
  fBaselineRMS = p.get< double >("BaselineRMS");
  fHypoSignalTimeWindow = p.get< double >("HypoSignalTimeWindow");
  fHypoSignalScale = p.get< double >("HypoSignalScale");
  fPMTChargeToADC = p.get< double >("PMTChargeToADC");
  fDecoWaveformPrecision = p.get< double >("DecoWaveformPrecision");
  fBaselineMinADC = p.get< double >("BaselineMinADC");
  fBaselineMaxADC = p.get< double >("BaselineMaxADC");
  fOpDetDataFile = p.get< std::string >("OpDetDataFile");
  fFilter = p.get< std::string >("Filter");
  fScaleHypoSignal = p.get< bool >("ScaleHypoSignal");

  std::string fname;
  cet::search_path sp("FW_SEARCH_PATH");
  sp.find_file(fOpDetDataFile, fname);
  TFile* file = TFile::Open(fname.c_str(), "READ");


  fNormUnAvSmooth=1./(2*fUnAvNeighbours+1);
  NDecoWf=0;

  auto const clockData = art::ServiceHandle<detinfo::DetectorClocksService const>()->DataForJob();
  fSamplingFreq=clockData.OpticalClock().Frequency()/1000.;
  auto const* lar_prop = lar::providerFrom<detinfo::LArPropertiesService>();

  //Initizalize random numbers
  //auto& engine = art::ServiceHandle<rndm::NuRandomService>{}->createEngine(1);//sim::GetRandomNumberSeed());
  //fCLHEPEGauss = std::make_unique<CLHEP::RandGauss>(engine);
  //CLHEP::HepRandomEngine& fEngine; // random engine
  art::ServiceHandle<rndm::NuRandomService> seedSvc;
  CLHEP::HepJamesRandom *engine = new CLHEP::HepJamesRandom;
  seedSvc->registerEngine(rndm::NuRandomService::CLHEPengineSeeder(engine), "opdeconvolutionSBND");
  CLHEP::RandGauss GaussGen(engine);
  fCLHEPEGauss = std::make_unique<CLHEP::RandGauss>(engine);


  nbinsFFT=std::pow(2, fFFTSizePow);
  fSignalHypothesis = CreateWienerHypothesis(nbinsFFT, *lar_prop);
  std::cout<<"Creating light signal hypothesis... size"<<fSignalHypothesis.size()<<std::endl;

  fNoiseHypothesis = CreateNoiseHypothesis(nbinsFFT);
  std::cout<<"Creating noise... size"<<fNoiseHypothesis.size()<<std::endl;
  std::vector<double>* SinglePEVec_p;

  file->GetObject("SinglePEVec", SinglePEVec_p);
  fSinglePEWave = *SinglePEVec_p;
  while(fSinglePEWave.size()<nbinsFFT)
    fSinglePEWave.push_back(0);
  std::cout<<"Creating SER... size:"<<fSinglePEWave.size()<<std::endl;

  std::cout<<"Selected Filter: "<<fFilter<<std::endl;

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
    fSERfft_abs2.push_back( std::pow(TComplex::Abs(fSERfft[k]),2) );

    //OLD WIENER FILTER:
    fNSR.push_back( 1.*std::pow(TComplex::Abs(noise_fft[k]),2)/std::pow(TComplex::Abs(hypothesis_fft[k]),2) );
    //NEW WIENER FILTER: fNSR.push_back( 1.*std::pow( fBaselineRMS ,2)/std::pow(TComplex::Abs(hypothesis_fft[k]),2) );
    //fNSR.push_back( 1.*std::pow( fBaselineRMS ,2)/std::pow(TComplex::Abs(hypothesis_fft[k]),2) );
  }
}




std::vector<raw::OpDetWaveform> opdet::OpDeconvolution::RunDeconvolution(std::vector<raw::OpDetWaveform> wfVector)
{
  std::vector<raw::OpDetWaveform> wfDeco;

  for(auto const& wf : wfVector)
  {
    //Read waveform and substract baseline
    //std::cout<<" Wf size: "<<wf.Waveform().size()<<" OpCh="<<wf.ChannelNumber()<<"\n";
    size_t wfsize=wf.Waveform().size();
    std::vector<double> wave(wf.Waveform().begin(), wf.Waveform().end());
    SubstractBaseline(wave);



    double minADC=*min_element(wave.begin(), wave.end());
    double scaling_factor=fHypoSignalScale*(-minADC)/fPMTChargeToADC;

    //Apply smoothing
    if(fApplyExpoAvSmooth)
      ApplyExpoAvSmoothing(wave);
    if(fApplyUnAvSmooth)
      ApplyUnAvSmoothing(wave);

    size_t wfsizefft=WfSizeFFT(wfsize);

    fft_service->ReinitializeFFT (wfsizefft, "", 20);
    std::vector<double> ser(fSinglePEWave.begin(), std::next(fSinglePEWave.begin(), wfsizefft));
    std::vector<double> hypo(fSignalHypothesis.begin(), std::next(fSignalHypothesis.begin(), wfsizefft));
    std::vector<TComplex> serfft, hypofft;
    serfft.resize(wfsizefft); hypofft.resize(wfsizefft);
    fft_service->DoFFT(ser, serfft);
    fft_service->DoFFT(hypo, hypofft);


    TComplex kerinit(0,0,false);
    std::vector<TComplex> kernel(wfsizefft, kerinit);
    double noise_power=wfsize*fBaselineRMS*fBaselineRMS;
    if(fScaleHypoSignal){
      noise_power/=pow(scaling_factor, 2);
    }


    for(size_t k=0; k<wfsizefft/2; k++){
      double den=1.;
      if(fFilter=="Wiener")
        den = TComplex::Abs(serfft[k])*TComplex::Abs(serfft[k]) + noise_power / (TComplex::Abs(hypofft[k])*TComplex::Abs(hypofft[k]) );
      else if(fFilter=="Wiener1PE")
        den = TComplex::Abs(serfft[k])*TComplex::Abs(serfft[k]) + noise_power ;

      kernel[k]= TComplex::Conjugate( serfft[k] ) / den;
      //std::cout<< k<<":"<<serfft[k]<<":"<<hypofft[k]<<":"<<den << " ";
    }
    //Deconvolve raw signal (covolve with kernel)
    fft_service->ReinitializeFFT(wfsizefft, "", 20);
    wave.resize(wfsizefft, 0);
    fft_service->Convolute(wave, kernel);


    //Note: only nbinsFFT/2 bins used in deconvolution, optimize memory?
    /*TComplex kerinit(0,0,false);
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
    fft_service->Convolute(wave, kernel);*/


    //Prepare deconvolved waveform before saving
    double fDecoWfScaleFactor=1./fDecoWaveformPrecision;
    std::transform(wave.begin(), wave.end(), wave.begin(), [fDecoWfScaleFactor](double &dec){ return fDecoWfScaleFactor*dec; } );

    //Debbuging and save wf in hist file
    if(fDebug){
      std::cout<<"\n.....Debbuging.....\n";
      auto minADC_ix=min_element(wave.begin(), wave.end());
      std::cout<<std::endl<<"Stamp="<<wf.TimeStamp()<<" OpCh"<<wf.ChannelNumber()<<" MinADC="<<minADC<<" (";
      std::cout<<minADC_ix-wave.begin()<<") Size="<<wf.Waveform().size()<<" ScFactor="<<scaling_factor<<std::endl;
      if(wave.size()>6000){
        std::string name="h_raw"+std::to_string(NDecoWf)+"_"+std::to_string(wf.ChannelNumber())+"_"+std::to_string(wf.TimeStamp());
        TH1F * h_raw = tfs->make< TH1F >(name.c_str(),";Bin;ADC", nbinsFFT, 0, nbinsFFT);
        name="h_deco"+std::to_string(NDecoWf)+"_"+std::to_string(wf.ChannelNumber())+"_"+std::to_string(wf.TimeStamp());
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

    raw::OpDetWaveform decowf(wf.TimeStamp(), wf.ChannelNumber(), std::vector<short unsigned int> (wave.begin(),  std::next(wave.begin(), wf.Waveform().size()) ) );
    wfDeco.push_back(decowf);
    NDecoWf++;
  }

  return wfDeco;
}


void opdet::OpDeconvolution::ApplyExpoAvSmoothing(std::vector<double>& wf){
  std::transform (std::next(wf.begin(), 1), wf.end(), wf.begin(), std::next(wf.begin(), 1),
    [&](double _x, double _y) { return  fExpoAvSmoothPar*_x+ (1. - fExpoAvSmoothPar)*_y; }  );
}

void opdet::OpDeconvolution::ApplyUnAvSmoothing(std::vector<double>& wf){
  std::vector<double> wf_aux(wf.begin(), wf.end());
  for(size_t bin=fUnAvNeighbours; bin<wf.size()-fUnAvNeighbours; bin++){
    double sum=0.;
    for(size_t nbin=bin-fUnAvNeighbours; nbin<=bin+fUnAvNeighbours; nbin++)
      sum+=wf_aux[nbin];
    //std::cout<<bin<<" "<<sum<<" "<<sum*fNormUnAvSmooth<<std::endl;
    wf[bin]=sum*fNormUnAvSmooth;
  }
}

size_t opdet::OpDeconvolution::WfSizeFFT(size_t n){
  size_t cont=0;
  while(n>0){
    cont++;
    n=(n>>1);
  }
  return pow(2, cont);
}

std::vector<double> opdet::OpDeconvolution::CreateWienerHypothesis(size_t n, detinfo::LArProperties const& lar_prop){
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
    if(t<fHypoSignalTimeWindow)
      v.push_back( SlowFast_Fraction*std::exp(-1.*t/TFast) + (1.-SlowFast_Fraction)*std::exp(-1.*t/TSlow) );
    else
      v.push_back(0);
  }
  return v;
}


std::vector<double> opdet::OpDeconvolution::CreateNoiseHypothesis(size_t n){
  std::vector<double> v;
  for(size_t k=0; k<n; k++){
    //std::cout<<"Noise "<<k<<" "<<fCLHEPEGauss->fire(0, fBaselineRMS)<<std::endl;
    v.push_back( fCLHEPEGauss->fire(0, fBaselineRMS) );
  }
  return v;
}

void opdet::OpDeconvolution::SubstractBaseline(std::vector<double> &wf){
  double minADC=*min_element(wf.begin(), wf.end());
  double maxADC=*max_element(wf.begin(), wf.end());
  TH1F h_ba = TH1F("",";;", (int)(maxADC-minADC), minADC-0.5, maxADC-0.5);
  for(auto &adc:wf)
    h_ba.Fill(adc);

  double _baseline=h_ba.GetXaxis()->GetBinCenter(h_ba.GetMaximumBin());

  std::transform (wf.begin(), wf.end(), wf.begin(), [&](double _x) { return _x-_baseline; }  );

  if(fDebug){
    std::cout<<"   -- Estimating baseline (mode algorithm): "<<_baseline<<std::endl;
    std::string name="h_baseline_"+std::to_string(NDecoWf);
    TH1F * hs_ba = tfs->make< TH1F >(name.c_str(),"Baseline;ADC;# entries",(int)(maxADC-minADC), minADC-0.5, maxADC-0.5);
    for(int k=1; k<=h_ba.GetNbinsX(); k++)
      hs_ba->SetBinContent(k, h_ba.GetBinContent(k));
  }

  return;
}



#endif


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


/*
//BORRAR:
fBaselineRMS=.5; double nfac=1.;
std::vector<TComplex> noise_fft_aux;
std::vector<double> noiseabs_fft_aux;

fNoiseHypothesis = CreateNoiseHypothesis((size_t)pow(2, 11));
fft_service->ReinitializeFFT ((size_t)pow(2, 11), "", 20); noise_fft_aux.resize((size_t)pow(2, 11));
fft_service->DoFFT(fNoiseHypothesis, noise_fft_aux);
noiseabs_fft_aux.clear(); //nfac=noise_fft_aux.size();
for(size_t k=0; k<noise_fft_aux.size(); k++) noiseabs_fft_aux.push_back( std::pow(TComplex::Abs(noise_fft_aux[k]),2)/nfac);
noiseabs_fft_aux.resize(noiseabs_fft_aux.size()/2);
std::cout<<"++++++ NOISE 1:"<<noise_fft_aux.size()<<" "<<EstimateBaseline(noiseabs_fft_aux)<<std::endl;

fNoiseHypothesis = CreateNoiseHypothesis((size_t)pow(2, 13));
fft_service->ReinitializeFFT ((size_t)pow(2, 13), "", 20); noise_fft_aux.resize((size_t)pow(2, 13));
fft_service->DoFFT(fNoiseHypothesis, noise_fft_aux);
noiseabs_fft_aux.clear(); //nfac=noise_fft_aux.size();
for(size_t k=0; k<noise_fft_aux.size(); k++) noiseabs_fft_aux.push_back( std::pow(TComplex::Abs(noise_fft_aux[k]),2)/nfac);
noiseabs_fft_aux.resize(noiseabs_fft_aux.size()/2);
std::cout<<"++++++ NOISE 2:"<<noise_fft_aux.size()<<" "<<EstimateBaseline(noiseabs_fft_aux)<<std::endl;

fNoiseHypothesis = CreateNoiseHypothesis((size_t)pow(2, 15));
fft_service->ReinitializeFFT ((size_t)pow(2, 15), "", 20); noise_fft_aux.resize((size_t)pow(2, 15));
fft_service->DoFFT(fNoiseHypothesis, noise_fft_aux);
noiseabs_fft_aux.clear(); //nfac=noise_fft_aux.size();
for(size_t k=0; k<noise_fft_aux.size(); k++) noiseabs_fft_aux.push_back( std::pow(TComplex::Abs(noise_fft_aux[k]),2)/nfac );
noiseabs_fft_aux.resize(noiseabs_fft_aux.size()/2);
std::cout<<"++++++ NOISE 3:"<<noise_fft_aux.size()<<" "<<EstimateBaseline(noiseabs_fft_aux)<<std::endl;
*/

/*void opdet::OpDeconvolution::SubstractBaseline2(std::vector<double> &waveform){
  std::cout<<"In estimate Baseline 2\n";
  waveform.reserve(waveform.size());
  std::cout<<"JJ\n";
  TH1F h_ba = TH1F("",";;", (int)(fBaselineMaxADC-fBaselineMinADC), fBaselineMinADC-0.5, fBaselineMaxADC-0.5);
  for(auto &adc:waveform){
    h_ba.Fill(adc);
    std::cout<<adc<<"  ";
  }

  std::cout<<h_ba.GetEntries()<<" Jj\n";
  double _baseline=h_ba.GetXaxis()->GetBinCenter(h_ba.GetMaximumBin());
  std::cout<<"Jk "<<waveform.size()<<" "<<_baseline<<std::endl;

  for(auto &adc:waveform){
    std::cout<<adc<<"  ";
  }

  for(size_t k=0; k<waveform.size(); k++){
    std::cout<<k<<":"<<waveform.at(k)<<" ";
    waveform.at(k)=waveform.at(k)-_baseline;
  }


  return;
}*/
