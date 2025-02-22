////////////////////////////////////////////////////////////////////////
// Class:       pmtTriggerAna
// Module Type: analyzer
// File:        pmtTriggerAna_module.cc
//
// Analyzer to determine pmt hardware trigger
//
// Authors: Erin Yandel
////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <vector>
#include <cmath>
#include <memory>
#include <string>

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "canvas/Utilities/Exception.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "larcore/Geometry/Geometry.h"

#include "lardata/DetectorInfoServices/DetectorClocksService.h"
#include "lardata/DetectorInfoServices/DetectorClocksServiceStandard.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesService.h"
#include "lardata/DetectorInfoServices/LArPropertiesService.h"
#include "lardataobj/RawData/OpDetWaveform.h"
#include "sbndcode/Utilities/SignalShapingServiceSBND.h"
#include "lardataobj/Simulation/sim.h"
#include "lardataobj/Simulation/SimChannel.h"
#include "lardataobj/Simulation/SimPhotons.h"

#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"

#include "sbndcode/OpDetSim/sbndPDMapAlg.hh"

namespace opdet {

  class pmtTriggerAna;

  class pmtTriggerAna : public art::EDAnalyzer {
  public:
    explicit pmtTriggerAna(fhicl::ParameterSet const & p);
    // The destructor generated by the compiler is fine for classes
    // without bare pointers or other resource use.

    // Plugins should not be copied or assigned.
    pmtTriggerAna(pmtTriggerAna const &) = delete;
    pmtTriggerAna(pmtTriggerAna &&) = delete;
    pmtTriggerAna & operator = (pmtTriggerAna const &) = delete;
    pmtTriggerAna & operator = (pmtTriggerAna &&) = delete;

    // Required functions.
    void analyze(art::Event const & e) override;

    //Selected optional functions
    void beginJob() override;
    void endJob() override;

    opdet::sbndPDMapAlg pdMap; //map for photon detector types
  private:

    int run;
    int subrun;
    int event;
    size_t fEvNumber;
    size_t fChNumber; //channel number
    double fSampling; //sampling rate
    double fStartTime; //start time (in us) of raw waveform
    double fEndTime; //end time (in us) of raw waeform
    double fWindowStart; //start time (in us) of trigger window (set in fcl, 0 for beam spill)
    double fWindowEnd; //end time (in us) of trigger window (set in fcl, 1.6 for beam spill)
    TTree *fTree;

    //double fBaseline = 8000.0; //baseline ADC (set in simulation)
    double fThreshold; //individual pmt threshold in ADC (set in fcl, passes if ADC is LESS THAN threshold)
    int fOVTHRWidth;//over-threshold width, page 40 of manual (set in fcl)
    std::vector<int> fPair1; //channel numbers for first set of paired pmts (set in fcl)
    std::vector<int> fPair2; //channel numbers for second set of paired pmts (set in fcl)
    std::vector<int> fUnpaired;//channel numbers for unpired pmts (set in fcl)

    std::string fInputModuleName; //opdet waveform module name (set in fcl)
    std::vector<std::string> fOpDetsToPlot; //types of optical detetcors (e.g. "pmt_coated", "xarapuca_vuv", etc.), should only be pmt_coated and pmt_uncoated (set in fcl)
    std::stringstream histname; //raw waveform hist name
    std::stringstream histname2; //other hists names
    std::string opdetType; //opdet wavform's opdet type (required to be pmt_coated or pmt_uncoated)

    std::vector<int> passed_trigger; //index =time (us, triger window only), content = number of pmt pairs passed threshold
    int max_passed; //maximum number of pmt pairs passing threshold at the same time within trigger window
  };


  pmtTriggerAna::pmtTriggerAna(fhicl::ParameterSet const & p)
    :
    EDAnalyzer(p),
    fTree(nullptr)
    // More initializers here.
  {
    art::ServiceHandle<art::TFileService> tfs;
    fTree = tfs->make<TTree>("pmttriggertree","analysis tree");
    fTree->Branch("run", &run);
    fTree->Branch("subrun", &subrun);
    fTree->Branch("event", &event);
    fTree->Branch("passed_trigger", &passed_trigger);
    fTree->Branch("max_passed",&max_passed);

    fInputModuleName = p.get< std::string >("InputModule" );
    fOpDetsToPlot    = p.get<std::vector<std::string> >("OpDetsToPlot");
    fThreshold       = p.get<float>("Threshold");
    fOVTHRWidth        = p.get<int>("OVTHRWidth");
    fPair1    = p.get<std::vector<int> >("Pair1");
    fPair2    = p.get<std::vector<int> >("Pair2");
    fUnpaired    = p.get<std::vector<int> >("Unpaired");
    fWindowStart = p.get<float>("WindowStart");
    fWindowEnd = p.get<float>("WindowEnd");

    auto const clockData = art::ServiceHandle<detinfo::DetectorClocksService const>()->DataForJob();
    fSampling = clockData.OpticalClock().Frequency(); // MHz

  }

  void pmtTriggerAna::beginJob()
  {

  }

  void pmtTriggerAna::analyze(art::Event const & e)
  {
    // Implementation of required member function here.
    std::cout << "My module on event #" << e.id().event() << std::endl;

    art::ServiceHandle<art::TFileService> tfs;
    fEvNumber = e.id().event();
    run = e.run();
    subrun = e.subRun();
    event = e.id().event();

    art::Handle< std::vector< raw::OpDetWaveform > > waveHandle;
    e.getByLabel(fInputModuleName, waveHandle);

    if(!waveHandle.isValid()) {
      std::cout << Form("Did not find any G4 photons from a producer: %s", "largeant") << std::endl;
    }

    // // example of usage for pdMap.getCollectionWithProperty()
    // //
    // // define a container
    // auto inBoxTwo = pdMap.getCollectionWithProperty("pds_box", 2);
    // // you can cout the whole json object
    // std::cout << "inBoxTwo:\t" << inBoxTwo << "\n";
    // // traverse its components in a loop
    // for (auto const &e: inBoxTwo) {
    //   std::cout << e["pd_type"] << " " << e["channel"] << ' ' << "\n";
    // }

    // // example of usage for pdMap.getCollectionFromCondition()
    // // define a lambda function with the conditions
    // auto subsetCondition = [](auto const& e)->bool
    //   // modify conditions as you want in the curly braces below
    //   {return e["pd_type"] == "pmt_uncoated" && e["tpc"] == 0;};
    // // get the container that satisfies the conditions
    // auto uncoatedsInTPC0 = pdMap.getCollectionFromCondition(subsetCondition);
    // std::cout << "uncoatedsInTPC0.size():\t" << uncoatedsInTPC0.size() << "\n";
    // for(auto const& e:uncoatedsInTPC0){
    //   std::cout << "e:\t" << e << "\n";
    // }

    //fOpDetsToPlot = ["pmt_coated", "pmt_uncoated"];

    max_passed = 0;

    //std::cout << "Number of PMT waveforms: " << waveHandle->size() << std::endl;
    int num_pmt_wvf = 0;
    int num_pmt_ch = 0;

    std::cout << "fOpDetsToPlot:\t";
    for (auto const& opdet : fOpDetsToPlot){std::cout << opdet << " ";}
    std::cout << std::endl;

    //size_t previous_channel = -1;
    //std::vector<int> previous_waveform;
    //std::vector<int> previous_waveform_down;

    std::vector<int> channel_numbers = {6,7,8,9,10,11,12,13,14,15,16,17,36,37,38,39,40,41,60,61,62,63,64,65,66,67,68,69,70,71,
      84,85,86,87,88,89,90,91,92,93,94,95,114,115,116,117,118,119,138,139,140,141,142,143,144,145,146,147,148,149,
      162,163,164,165,166,167,168,169,170,171,172,173,192,193,194,195,196,197,216,217,218,219,220,221,222,223,224,225,226,227,
      240,241,242,243,244,245,246,247,248,249,250,251,270,271,272,273,274,275,294,295,296,297,298,299,300,301,302,303,304,305};
    std::vector<std::vector<int>> channel_bin_wvfs;

    std::vector<int> paired;
    std::vector<std::vector<int>> unpaired_wvfs;

    //std::vector<int> passed_trigger;


    std::vector<int> wvf_bin_0;
    for (double i = -1500.0; i<1500.0; i+=(1./fSampling)){
      wvf_bin_0.emplace_back(0);
    }
    for (size_t i = 0; i<120; i++){
      channel_bin_wvfs.emplace_back(wvf_bin_0);
    }

    for (size_t i = 0; i<fPair1.size(); i++){
      paired.emplace_back(0);
    }

    for (size_t i = 0; i<fPair1.size(); i++){
      unpaired_wvfs.emplace_back(paired);
    }

    for (double i = fWindowStart; i<fWindowEnd; i+=(4./fSampling)){
      passed_trigger.emplace_back(0);
    }


    if (fPair2.size()!=paired.size()){std::cout<<"Pair lists mismatched sizes!"<<std::endl;}

    size_t wvf_id = -1;
    int hist_id = -1;
    for(auto const& wvf : (*waveHandle)) {
      wvf_id++;
      hist_id++;
      fChNumber = wvf.ChannelNumber();
      opdetType = pdMap.pdType(fChNumber);
      if (std::find(fOpDetsToPlot.begin(), fOpDetsToPlot.end(), opdetType) == fOpDetsToPlot.end()) {continue;}
      num_pmt_wvf++;
        histname.str(std::string());
        histname << "event_" << fEvNumber
                 << "_opchannel_" << fChNumber
                 << "_" << opdetType
                 << "_" << hist_id
                 << "_raw";


        fStartTime = wvf.TimeStamp(); //in us
        fEndTime = double(wvf.size()) / fSampling + fStartTime; //in us

        //double orig_size = double(wvf.size());

        //baseline
        //double baseline = -1.;

        //std::vector<double> wvf_full;

        //create binary waveform
        std::vector<int> wvf_bin;
        //std::vector<int> wvf_bin_down;

        //pmt above Threshold
        //bool above_thres = false;

        //Create a new histogram for binary waveform
        TH1D *wvfHist = tfs->make< TH1D >(histname.str().c_str(), "Raw Waveform"/*TString::Format(";t - %f (#mus);", fStartTime)*/, wvf.size(), fStartTime, fEndTime);
        wvfHist->GetXaxis()->SetTitle("t (#mus)");
        for(unsigned int i = 0; i < wvf.size(); i++) {
          wvfHist->SetBinContent(i + 1, (double)wvf[i]);
        }

      /*  wvfHist->Fit("pol0", "Q");
        //std::cout << "Line 190" << std::endl;
        TF1 *bline = (TF1*)wvfHist->GetListOfFunctions()->FindObject("pol0");
        //std::cout << "Line 192" << std::endl;
        baseline = bline->GetParameter(0); */
        //std::cout << "Line 194" << std::endl;


        /*for(unsigned int i = 0; i < wvf.size(); i++) {
          wvf_full.emplace_back((double)wvf[i]);
        }*/

      /*  histname2.str(std::string());
        histname2 << "event_" << fEvNumber
                 << "_opchannel_" << fChNumber
                 << "_" << opdetType
                 << "_" << hist_id
                 << "_binary"; */

        if (fStartTime < -1500.0/*-1488.94*/){std::cout<<"Start Time is "<<fEndTime<<std::endl;}

         if (fStartTime > -1500.0){
           for (double i = fStartTime+1500.0; i>0.; i-=(1./fSampling)){
             wvf_bin.emplace_back(0);
           }
         }

        for(unsigned int i = 0; i < wvf.size(); i++) {
          if((double)wvf[i]<fThreshold){wvf_bin.emplace_back(1);}else{wvf_bin.emplace_back(0);}
        }

        if (fEndTime > 1500.0/*1486.98 1473.08*/){std::cout<<"End Time is "<<fEndTime<<std::endl;}

        if (fEndTime < 1500.0){
          for (double i = 1500.0-fEndTime; i>0.; i-=(1./fSampling)){
            wvf_bin.emplace_back(0);
          }
        }

      //  fStartTime = -1500.0;
      //  fEndTime = 1500.0;//1473.08;

        //combine wavform with any other waveforms from same channel
        int i_ch = -1.;
        auto ich = std::find(channel_numbers.begin(), channel_numbers.end(), fChNumber);
        if (ich != channel_numbers.end()){
          i_ch = ich - channel_numbers.begin();
        }
        for(unsigned int i = 0; i < wvf_bin.size(); i++) {
          if(channel_bin_wvfs.at(i_ch)[i]==1 || wvf_bin[i]==1){channel_bin_wvfs.at(i_ch)[i]=1;}else{channel_bin_wvfs.at(i_ch)[i]=0;}
        }

      }//wave handle loop

        //combine all waveforms for a single PMT into one, only works if OoDetWaveForms are saved in order of channel number (i.e. all waveforms for
        //a channel are next to each other)
      /*  if (fChNumber==previous_channel){

          //if (wvf_bin.size()!=previous_waveform.size()){std::cout << "Mismatch binary waveform sizes" << std::endl;}
          //unsigned int wvf_size = wvf_bin.size();
          //if (wvf_bin.size() < previous_waveform.size()){

          for(unsigned int i = 0; i < wvf_bin.size(); i++) {
            //if (i==previous_waveform.size()){previous_waveform.emplace_back(0);}
            if(previous_waveform[i]==1 || wvf_bin[i]==1){wvf_bin[i]=1;}else{wvf_bin[i]=0;}
          }

          //if (wvf_bin.size()!=previous_waveform.size()){std::cout << "Mismatch binary waveform sizes" << std::endl;}

        } */

        //if (wvf_bin.size()!=wvf.size()){std::cout << "Mismatch analog and binary waveform size" << std::endl;}

      int wvf_num = -1;

      for (auto wvf_bin : channel_bin_wvfs){
        wvf_num++;
        fChNumber = channel_numbers.at(wvf_num);
        fStartTime = -1500.0;
        fEndTime = 1500.0;

        //downscale binary waveform by 4
        std::vector<int> wvf_bin_down;
        for(unsigned int i = 0; i < wvf_bin.size(); i++) {
          if(i%4==0){wvf_bin_down.emplace_back(wvf_bin[i]);}
        }

      //if (wvf_id==waveHandle->size() || waveHandle->at(wvf_id+1).ChannelNumber()!=fChNumber){

        num_pmt_ch++;

        histname2.str(std::string());
        histname2 << "event_" << fEvNumber
                 << "_opchannel_" << fChNumber
                 << "_binary";
        TH1D *wvfbHist = tfs->make< TH1D >(histname2.str().c_str(), "Binary Waveform"/*TString::Format(";t - %f (#mus);", fStartTime)*/, wvf_bin.size(), fStartTime, fEndTime);
        wvfbHist->GetXaxis()->SetTitle("t (#mus)");
        for(unsigned int i = 0; i < wvf_bin.size(); i++) {
          wvfbHist->SetBinContent(i + 1, wvf_bin[i]);
        }

        histname2.str(std::string());
        histname2 << "event_" << fEvNumber
                 << "_opchannel_" << fChNumber
                 << "_binary_down";

        TH1D *wvfbdHist = tfs->make< TH1D >(histname2.str().c_str(), "Downsampled Binary Waveform"/*TString::Format(";t - %f (#mus);", fStartTime)*/, wvf_bin_down.size(), fStartTime, fEndTime);
        wvfbdHist->GetXaxis()->SetTitle("t (#mus)");
        for(unsigned int i = 0; i < wvf_bin_down.size(); i++) {
          wvfbdHist->SetBinContent(i + 1, wvf_bin_down[i]);
        }

        bool combine = false;
        bool found = false;
        bool unpaired = false;
        size_t pair_num = -1;


          /*  std::cout<<"Pair1 size:"<<fPair1.size()<<std::endl;
            std::cout<<"Pair2 size:"<<fPair2.size()<<std::endl;
            std::cout<<"paired size:"<<paired.size()<<std::endl;
            std::cout<<"unpaired wvfs size:"<<unpaired_wvfs.size()<<std::endl;*/

        for (size_t i = 0; i < fUnpaired.size(); i++){
          if (fUnpaired.at(i) == (int)fChNumber){found=true; unpaired=true;}
        }

        if (!found){
          for (size_t i = 0; i < fPair1.size(); i++){
            if (fPair1.at(i) == (int)fChNumber && paired.at(i)==1){found=true; pair_num=i; combine=true; break;}
            else if (fPair1.at(i) == (int)fChNumber && paired.at(i)==0){found=true; unpaired_wvfs.at(i)=wvf_bin_down; paired.at(i)=1; break;}
          }
          if (!found){
            for (size_t i = 0; i < fPair2.size(); i++){
              if (fPair2.at(i) == (int)fChNumber && paired.at(i)==1){found=true; pair_num=i; combine=true; break;}
              else if (fPair2.at(i) == (int)fChNumber && paired.at(i)==0){found=true; unpaired_wvfs.at(i)=wvf_bin_down; paired.at(i)=1; break;}
            }
          }
        }

        if (combine || unpaired){
          std::vector<int> wvf_combine;
          if (combine){
            if (unpaired_wvfs.at(pair_num).size()!=wvf_bin_down.size()){std::cout<<"Mismatched paired waveform size"<<std::endl;}
            for(unsigned int i = 0; i < wvf_bin_down.size(); i++) {
              //if (i==unpaired_wvfs.at(pair_num).size()){unpaired_wvfs.at(pair_num).emplace_back(0);}
              if(unpaired_wvfs.at(pair_num)[i]==1 || wvf_bin_down[i]==1){wvf_combine.emplace_back(1);}else{wvf_combine.emplace_back(0);}
            }
          }else if(unpaired){
            wvf_combine = wvf_bin_down;
          }


        histname2.str(std::string());
        if (unpaired){
          histname2 << "event_" << fEvNumber
                   << "_opchannels_" << fChNumber
                   << "_unpaired"
                   << "_combined";
        }else{
          histname2 << "event_" << fEvNumber
                   << "_opchannels_" << fPair1.at(pair_num)
                   << "_" << fPair2.at(pair_num)
                   << "_combined";
        }

        TH1D *wvfcHist = tfs->make< TH1D >(histname2.str().c_str(), "Paired Waveform"/*TString::Format(";t - %f (#mus);", fStartTime)*/, wvf_combine.size(), fStartTime, fEndTime);
        wvfcHist->GetXaxis()->SetTitle("t (#mus)");
        for(unsigned int i = 0; i < wvf_combine.size(); i++) {
          wvfcHist->SetBinContent(i + 1, wvf_combine[i]);
        }

        //std::cout<<"Hist "<<histname2.str().c_str()<<" created"<<std::endl;

        //implement over threshold trigger signal width
        //(Every time the combined waveform transitions from 0 to 1, change the next fOVTHRWidth values to 1 (ex: fOVTHRWidth=11 -> 12 high -> 12*8=96 ns true) )
        for(unsigned int i = 1; i < wvf_combine.size()-fOVTHRWidth; i++) {
          if(wvf_combine[i]==1 && wvf_combine[i-1]==0){
            for(unsigned int j = i+1; j < i+fOVTHRWidth+1; j++){
              wvf_combine[j] = 1;
            }
            }
        }

        histname2.str(std::string());
        if (unpaired){
          histname2 << "event_" << fEvNumber
                   << "_opchannels_" << fChNumber
                   << "_unpaired"
                   << "_combined_width";
        }else{
          histname2 << "event_" << fEvNumber
                   << "_opchannels_" << fPair1.at(pair_num)
                   << "_" << fPair2.at(pair_num)
                   << "_combined_width";
        }

        TH1D *wvfcwHist = tfs->make< TH1D >(histname2.str().c_str(), "Over Threshold Paired Waveform"/*TString::Format(";t - %f (#mus);", fStartTime)*/, wvf_combine.size(), fStartTime, fEndTime);
        wvfcwHist->GetXaxis()->SetTitle("t (#mus)");
        for(unsigned int i = 0; i < wvf_combine.size(); i++) {
          wvfcwHist->SetBinContent(i + 1, wvf_combine[i]);
        }

        //Combine the waveforms to get a 1D array of integers where the value corresponds to the number of pairs ON and the
        //index corresponds to the tick in the waveform
        double binspermus = wvf_combine.size()/(fEndTime-fStartTime);
        unsigned int startbin = std::floor(binspermus*(fWindowStart - fStartTime));
        unsigned int endbin = std::ceil(binspermus*(fWindowEnd - fStartTime));
        unsigned int i_p = 0;
        for(unsigned int i = startbin; i<endbin; i++){
          if (wvf_combine[i]==1){passed_trigger[i_p]++;}
          i_p++;
        }

      }



        //previous_channel = fChNumber;
        //previous_waveform = wvf_bin;
        //previous_waveform_down = wvf_bin_down;
        //hist_id++;
    }

    histname.str(std::string());
    histname << "event_" << fEvNumber
             << "_passed_trigger";

    TH1D *passedHist = tfs->make< TH1D >(histname.str().c_str(), "Number of PMTs Passing Trigger During Beam"/*TString::Format(";t - %f (#mus);", fStartTime)*/, passed_trigger.size(), fWindowStart, fWindowEnd);
    passedHist->GetXaxis()->SetTitle("t (#mus)");
    for(unsigned int i = 0; i < passed_trigger.size(); i++) {
      passedHist->SetBinContent(i + 1, passed_trigger[i]);
    }

    for (int pmts: passed_trigger){
      if (pmts > max_passed) max_passed = pmts;
    }

    std::cout << "Number of PMT waveforms: " << num_pmt_wvf << std::endl;
    std::cout << "Number of PMT channels: " << num_pmt_ch << std::endl;
    fTree->Fill();

    passed_trigger.clear();
    max_passed= 0;

  }

  void pmtTriggerAna::endJob()
  {
  }

  DEFINE_ART_MODULE(opdet::pmtTriggerAna)

}
