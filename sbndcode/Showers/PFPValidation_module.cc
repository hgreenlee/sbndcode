////////////////////////////////////////////////////////////////////////
// Class:       PFPValidation
// Plugin Type: analyzer (art v3_02_06)
// File:        PFPValidation_module.cc
//
// Generated at Wed Oct  2 03:27:09 2019 by Edward Tyley using cetskelgen
// from cetlib version v3_07_02.
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"
#include "art_root_io/TFileService.h"
#include "canvas/Persistency/Common/FindOneP.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "canvas/Persistency/Common/PtrVector.h"

//LArSoft Includes
#include "nusimdata/SimulationBase/MCTruth.h"
#include "larsim/MCCheater/ParticleInventoryService.h"
#include "larsim/MCCheater/BackTrackerService.h"
#include "lardata/DetectorInfoServices/DetectorClocksService.h"
#include "lardataobj/RecoBase/Cluster.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/Vertex.h"
#include "lardataobj/Simulation/SimChannel.h"
#include "lardataobj/RecoBase/PFParticleMetadata.h"
#include "sbndcode/RecoUtils/RecoUtils.h"
#include "sbndcode/RecoUtils/ShowerUtils.h"

//Root Includes
#include "TMath.h"
#include "TTree.h"
#include <vector>
#include <iostream>

namespace ana {
  class PFPValidation;
}

class ana::PFPValidation : public art::EDAnalyzer {
  public:
    explicit PFPValidation(fhicl::ParameterSet const& pset);
    // The compiler-generated destructor is fine for non-base
    // classes without bare pointers or other resource use.

    // Plugins should not be copied or assigned.
    PFPValidation(PFPValidation const&) = delete;
    PFPValidation(PFPValidation&&) = delete;
    PFPValidation& operator=(PFPValidation const&) = delete;
    PFPValidation& operator=(PFPValidation&&) = delete;

    // Required functions.
    void analyze(art::Event const& evt) override;
    void beginJob();

    void clearPFPTree(std::string PFParticleLabel);
    void clearTrueTree();
    void clearEventTree();

    std::map<int, int> GetTruePrimaryHits(
        const detinfo::DetectorClocksData& clockData,
        std::map<int,const simb::MCParticle*>& trueParticles,
        std::map<int,std::vector<int> >& truePrimaries,
        std::vector< art::Ptr< recob::Hit> >& allHits);

    std::map<int, float> GetTruePrimaryEnergies(
        std::map<int,const simb::MCParticle*>& trueParticles,
        std::map<int,std::vector<int> >& truePrimaries,
        std::vector<art::Ptr<sim::SimChannel> > simchannels);


    std::map<int, float> GetTruePrimaryHitEnergies(
        const detinfo::DetectorClocksData& clockData,
        std::map<int,const simb::MCParticle*>& trueParticles,
        std::map<int,std::vector<int> >& truePrimaries,
        std::vector< art::Ptr< recob::Hit> >& allHits);

    float GetTotalEnergyInHits(
        const detinfo::DetectorClocksData& clockData,
        std::vector<art::Ptr<recob::Hit> > hits);

    template <class T>
      void initTree(TTree* Tree, std::string branchName,
          std::map<std::string, T>& Metric,
          std::vector<std::string> fPFParticleLabels);

    struct TruthMatch {

      TruthMatch(): mRecoId(0), mTrueId(-999), mRecoPdg(-999), mNumHits(-999),
      mHitComp(-999), mHitPurity(-999), mEnergyComp(-999), mEnergyPurity(-999),
      mHitSPRatio(-999), mTrackScore(-999) {}

      TruthMatch(long unsigned int recoId, int trueId, int recoPdg, int numHits,
          float hitComp, float hitPurity, float energyComp, float energyPurity,
          float hitSPRatio, float trackScore):
        mRecoId(recoId), mTrueId(trueId), mRecoPdg(recoPdg), mNumHits(numHits),
        mHitComp(hitComp), mHitPurity(hitPurity), mEnergyComp(energyComp), mEnergyPurity(energyPurity),
        mHitSPRatio(hitSPRatio), mTrackScore(trackScore) {}

      long unsigned int mRecoId;
      int mTrueId, mRecoPdg, mNumHits;
      float mHitComp, mHitPurity, mEnergyComp, mEnergyPurity;
      float mHitSPRatio, mTrackScore;

      bool operator>(const TruthMatch& match) const {return mHitComp > match.mHitComp;}
    };

  private:

    std::string fHitLabel, fGenieGenModuleLabel, fLArGeantModuleLabel;
    std::vector<std::string> fPFParticleLabels;
    art::ServiceHandle<art::TFileService> tfs;
    art::ServiceHandle<cheat::BackTrackerService> bt_serv;
    art::ServiceHandle<cheat::ParticleInventoryService> particleInventory;

    // Declare member data here.
    unsigned int maxLabelLength;

    TTree* trueTree;
    TTree* pfpTree;
    TTree* eventTree;

    // Fill the tree once per particle
    int truePdg, motherPdg, numTrueHits;
    float trueEnergy, trueMomentum, trueDepositedEnergy, trueDepositedEnergyInHits;
    std::string trueProcess;
    std::map<std::string, int> recoPdg, recoPFPShowers, recoHits, recoPFPs, recoPFPTracks;
    std::map<std::string, float> hitPurity, energyPurity, hitComp, energyComp, hitSPRatio, recoTrackScore;

    std::string pfpModuleLabel;
    std::map<std::string, int> pfpPdg, pfpTruePdg, pfpNumHits;
    std::map<std::string, float> pfpTrueEnergy, pfpTrueMomentum, pfpTrueDepositedEnergy, pfpTrackScore;
    std::map<std::string, float> pfpHitPurity, pfpHitComp, pfpEnergyPurity, pfpEnergyComp, pfpHitSPRatio;

    std::map<std::string, int> eventNumPFP, eventNumPFPNu, eventNumPFPShower, eventNumPFPTrack;
};


ana::PFPValidation::PFPValidation(fhicl::ParameterSet const& pset)
  : EDAnalyzer{pset}
  , fHitLabel(pset.get<std::string>("HitLabel"))
  , fGenieGenModuleLabel(pset.get<std::string>("GenieGenModuleLabel"))
  , fLArGeantModuleLabel(pset.get<std::string>("LArGeantModuleLabel"))
  , fPFParticleLabels(pset.get<std::vector<std::string> >("PFParticleLabels"))
  , maxLabelLength(0)
{

  for (std::string const& fPFParticleLabel: fPFParticleLabels){
    if (fPFParticleLabel.length() > maxLabelLength){
      maxLabelLength = fPFParticleLabel.length();
    }
  }
}

void ana::PFPValidation::beginJob() {

  pfpTree   = tfs->make<TTree>("pfpTree", "Tree with metrics for each pfp");
  trueTree  = tfs->make<TTree>("trueTree", "Tree with metrics for each true particle");
  eventTree = tfs->make<TTree>("eventTree", "Tree with metrics for each event");

  trueTree->Branch("truePdg", &truePdg);
  trueTree->Branch("motherPdg", &motherPdg);
  trueTree->Branch("numTrueHits", &numTrueHits);
  trueTree->Branch("trueEnergy", &trueEnergy);
  trueTree->Branch("trueMomentum", &trueMomentum);
  trueTree->Branch("trueDepositedEnergy", &trueDepositedEnergy);
  trueTree->Branch("trueDepositedEnergyInHits", &trueDepositedEnergyInHits);
  trueTree->Branch("trueProcess", &trueProcess);

  initTree(trueTree, "recoPdg", recoPdg, fPFParticleLabels);
  initTree(trueTree, "recoTrackScore", recoTrackScore, fPFParticleLabels);
  initTree(trueTree, "recoHits", recoHits, fPFParticleLabels);
  initTree(trueTree, "recoPFPs", recoPFPs, fPFParticleLabels);
  initTree(trueTree, "recoPFPTracks", recoPFPTracks, fPFParticleLabels);
  initTree(trueTree, "recoPFPShowers", recoPFPShowers, fPFParticleLabels);
  initTree(trueTree, "hitPurity", hitPurity, fPFParticleLabels);
  initTree(trueTree, "hitComp", hitComp, fPFParticleLabels);
  initTree(trueTree, "energyPurity", energyPurity, fPFParticleLabels);
  initTree(trueTree, "energyComp", energyComp, fPFParticleLabels);
  initTree(trueTree, "hitSPRatio", hitSPRatio, fPFParticleLabels);

  initTree(pfpTree, "pfpPdg", pfpPdg, fPFParticleLabels);
  initTree(pfpTree, "pfpTruePdg", pfpTruePdg, fPFParticleLabels);
  initTree(pfpTree, "pfpNumHits", pfpNumHits, fPFParticleLabels);
  initTree(pfpTree, "pfpTrueEnergy", pfpTrueEnergy, fPFParticleLabels);
  initTree(pfpTree, "pfpTrueMomentum", pfpTrueMomentum, fPFParticleLabels);
  initTree(pfpTree, "pfpTrueDepositedEnergy", pfpTrueDepositedEnergy, fPFParticleLabels);
  initTree(pfpTree, "pfpTrackScore", pfpTrackScore, fPFParticleLabels);
  initTree(pfpTree, "pfpHitPurity", pfpHitPurity, fPFParticleLabels);
  initTree(pfpTree, "pfpHitComp", pfpHitComp, fPFParticleLabels);
  initTree(pfpTree, "pfpEnergyPurity", pfpEnergyPurity, fPFParticleLabels);
  initTree(pfpTree, "pfpEnergyComp", pfpEnergyComp, fPFParticleLabels);
  initTree(pfpTree, "pfpHitSPRatio", pfpHitSPRatio, fPFParticleLabels);

  initTree(eventTree,"numPFP", eventNumPFP, fPFParticleLabels);
  initTree(eventTree,"numPFPNu", eventNumPFPNu, fPFParticleLabels);
  initTree(eventTree,"numPFPShower", eventNumPFPShower, fPFParticleLabels);
  initTree(eventTree,"numPFPTrack", eventNumPFPTrack, fPFParticleLabels);
}

void ana::PFPValidation::analyze(art::Event const& evt)
{

  std::cout << std::setprecision(2) << std::fixed;

  // Get the true g4 particles and make a map form trackId
  std::map<int,const simb::MCParticle*> trueParticles;
  const sim::ParticleList& particles = particleInventory->ParticleList();
  for (auto const& particleIt: particles){
    const simb::MCParticle* particle = particleIt.second;
    trueParticles[particle->TrackId()] = particle;
  }

  std::map<int, std::vector<int> > showerMothers = ShowerUtils::GetShowerMothersCandidates(
      trueParticles);

  std::map<int, std::vector<int> > truePrimaries;
  for (auto const& [trackId, particle]: trueParticles){
    if (abs(particle->PdgCode())==11 || abs(particle->PdgCode())==22){
      auto const& showerMotherIter = showerMothers.find(trackId);
      if (showerMotherIter != showerMothers.end()){
        truePrimaries[trackId] = showerMothers[trackId];
      }
    } else {
      truePrimaries[trackId] = {trackId};
    }
  }

  // Get reco
  // Initialse some stuff???
  std::vector<art::Handle<std::vector<recob::Hit> > > hitHandles;
  evt.getManyByType(hitHandles);
  std::vector<art::Handle<std::vector<recob::Cluster> > > clusterHandles;
  evt.getManyByType(clusterHandles);
  art::Handle<std::vector<sim::SimChannel> > simChannelHandle;

  // Set the handles
  art::Handle<std::vector<recob::Hit> > hitHandle;
  art::Handle<std::vector<recob::SpacePoint> > spHandle;
  art::Handle<std::vector<recob::Cluster> > clusterHandle;
  art::Handle<std::vector<recob::PFParticle> > pfpHandle;

  // Get all the hits
  std::vector<art::Ptr<recob::Hit> > allHits;
  if(evt.getByLabel(fHitLabel,hitHandle))
  {art::fill_ptr_vector(allHits, hitHandle);}


  std::vector<art::Ptr<sim::SimChannel> > simChannels;
  if(evt.getByLabel(fLArGeantModuleLabel,simChannelHandle))
  {art::fill_ptr_vector(simChannels, simChannelHandle);}

  // Get map of true primary particle to number of reco hits / energy in reco hits
  auto const clockData = art::ServiceHandle<detinfo::DetectorClocksService const>()->DataFor(evt);
  std::map<int, int> truePrimaryHits = GetTruePrimaryHits(clockData, trueParticles, truePrimaries, allHits);
  std::map<int, float> truePrimaryEnergies = GetTruePrimaryEnergies(trueParticles, truePrimaries,
      simChannels);
  std::map<int, float> truePrimaryHitEnergies = GetTruePrimaryHitEnergies(clockData, trueParticles,
      truePrimaries, allHits);

  std::map<std::string, std::map<long unsigned int, art::Ptr<recob::PFParticle> > > pfpMap;
  std::map<std::string, std::map<long unsigned int, float> > pfpTrackScoreMap;

  std::map<std::string, std::vector<TruthMatch> > pfpTruthMatchMap;


  for (auto const fPFParticleLabel: fPFParticleLabels){

    // Get all the PFPs
    std::vector<art::Ptr<recob::PFParticle> > pfps;
    if(evt.getByLabel(fPFParticleLabel,pfpHandle))
    {art::fill_ptr_vector(pfps,pfpHandle);}
    std::vector<art::Ptr<recob::SpacePoint> > allSpacePoints;
    if(evt.getByLabel(fPFParticleLabel,spHandle))
    {art::fill_ptr_vector(allSpacePoints, spHandle);}

    art::FindManyP<recob::SpacePoint> fmpfsp(pfpHandle, evt, fPFParticleLabel);
    if(fmpfsp.isValid() && fmpfsp.size()>0){
      for (unsigned int fmpfspIter=0; fmpfspIter<fmpfsp.size(); fmpfspIter++){
        if (fmpfsp.at(fmpfspIter).size()==0) continue;
        evt.get(fmpfsp.at(fmpfspIter).front().id(),spHandle);
        if (!spHandle.isValid()) {
          std::cout<<"SpacePoint handle not valid"<<std::endl;
          return; // TODO: change to continue
        }
        break;
      }
    }
    if (!spHandle.isValid()) {
      std::cout<<"SpacePoint handle not valid"<<std::endl;
      return; // TODO: change to continue
    }

    art::FindManyP<recob::Cluster> fmpfc(pfpHandle, evt, fPFParticleLabel);
    if(fmpfc.isValid() && fmpfc.size()>0){
      for (unsigned int fmpfcIter=0; fmpfcIter<fmpfc.size(); fmpfcIter++){
        if (fmpfc.at(fmpfcIter).size()==0) continue;
        evt.get(fmpfc.at(fmpfcIter).front().id(),clusterHandle);
        if (!clusterHandle.isValid()) {
          std::cout<<"Cluster handle not valid"<<std::endl;
          return; // TODO: change to continue
        }
        break;
      }
    }
    if (!clusterHandle.isValid()) {
      std::cout<<"Cluster handle not valid"<<std::endl;
      return; // TODO: change to continue
    }

    art::FindManyP<recob::Hit> fmch(clusterHandle, evt, fPFParticleLabel);
    if(fmch.isValid() && fmch.size()>0){
      for (unsigned int fmchIter=0; fmchIter<fmch.size(); fmchIter++){
        if (fmch.at(fmchIter).size()==0) continue;
        evt.get(fmch.at(fmchIter).front().id(),hitHandle);
        if (!hitHandle.isValid()) {
          std::cout<<"Hit handle not valid"<<std::endl;
          return;
        }
        break;
      }
      if (!hitHandle.isValid()) {
        std::cout<<"Hit handle not valid"<<std::endl;
        return;
      }
    }


    // Create a map between PFParticles and their IDs
    art::FindManyP<larpandoraobj::PFParticleMetadata> fmpfpmd(pfps, evt, fPFParticleLabel);
    if (!fmpfpmd.isValid() || fmpfpmd.size()==0){
      std::cout<<"PFP MetaData handle not valid"<<std::endl;
      return;
    }

    // std::map<long unsigned int, float > pfpScoreMap;
    for (auto const& pfp: pfps){
      long unsigned int pfpId = pfp->Self();
      pfpMap[fPFParticleLabel][pfpId] = pfp;
      const std::vector< art::Ptr<larpandoraobj::PFParticleMetadata> > pfpMetaVec = fmpfpmd.at(pfpId);
      for (auto const& pfpMeta: pfpMetaVec)
      {
        larpandoraobj::PFParticleMetadata::PropertiesMap propertiesMap = pfpMeta->GetPropertiesMap();
        auto const pfpTrackScoreIter = propertiesMap.find("TrackScore");
        pfpTrackScoreMap[fPFParticleLabel][pfpId] =
          pfpTrackScoreIter == propertiesMap.end() ? -999 : pfpTrackScoreIter->second;
      }
    }

    clearEventTree();
    eventNumPFP[fPFParticleLabel] = pfps.size();

    for (auto const& pfp: pfps){

      clearPFPTree(fPFParticleLabel);

      // Update event wide metrics
      if (pfp->PdgCode()==11) {
        ++eventNumPFPShower[fPFParticleLabel];
      } else if (pfp->PdgCode()==13) {
        ++eventNumPFPTrack[fPFParticleLabel];
      } else {
        ++eventNumPFPNu[fPFParticleLabel];
      }

      // Get the hits from the PFParticle
      std::vector<art::Ptr<recob::Hit> > pfpHits;
      const std::vector< art::Ptr< recob::Cluster> >& clusters = fmpfc.at(pfp.key());
      for (const auto& cluster: clusters){
        const std::vector< art::Ptr< recob::Hit> >& hits = fmch.at(cluster.key());
        pfpHits.insert(pfpHits.end(), hits.begin(), hits.end());
      }

      const std::vector<art::Ptr<recob::SpacePoint> >& sps = fmpfsp.at(pfp.key());
      // const std::vector<art::Ptr<recob::SpacePoint> > sps;

      if (!pfpHits.size())
        continue;

      pfpPdg[fPFParticleLabel]       = pfp->PdgCode();
      pfpTrackScore[fPFParticleLabel]= pfpTrackScoreMap[fPFParticleLabel].at(pfp->Self());
      pfpNumHits[fPFParticleLabel]   = pfpHits.size();
      pfpHitSPRatio[fPFParticleLabel]= (float)sps.size() / pfpHits.size();

      std::pair<int, double> trueId = ShowerUtils::TrueParticleIDFromTrueChain(clockData, truePrimaries,
          pfpHits, 2);
      if (trueId.first==-99999){
        pfpTree->Fill();
        continue;
      }

      std::map<int, int> pfpTrueHitsMap = GetTruePrimaryHits(clockData, trueParticles, truePrimaries, pfpHits);
      std::map<int, float> pfpTrueEnergyMap = GetTruePrimaryHitEnergies(clockData, trueParticles,
          truePrimaries, pfpHits);

      const simb::MCParticle* trueParticle = trueParticles.at(trueId.first);

      pfpTruePdg[fPFParticleLabel]     = trueParticle->PdgCode();
      pfpTrueEnergy[fPFParticleLabel]  = trueParticle->E();
      pfpTrueMomentum[fPFParticleLabel]= trueParticle->P();

      int   pfpHitsTrueHits    = pfpTrueHitsMap.at(trueId.first);
      float pfpHitsTrueEnergy  = pfpTrueEnergyMap.at(trueId.first);
      float pfpHitsTotalEnergy = GetTotalEnergyInHits(clockData, pfpHits);

      pfpHitPurity[fPFParticleLabel]          = (float)pfpHitsTrueHits / pfpHits.size();
      pfpHitComp[fPFParticleLabel]            = (float)pfpHitsTrueHits / truePrimaryHits.at(trueId.first);
      pfpEnergyPurity[fPFParticleLabel]       = pfpHitsTrueEnergy / pfpHitsTotalEnergy;
      pfpEnergyComp[fPFParticleLabel]         = pfpHitsTrueEnergy / truePrimaryHitEnergies.at(trueId.first);
      // Energy is in all planes, hence divide deposited energy by 3
      pfpTrueDepositedEnergy[fPFParticleLabel]= truePrimaryHitEnergies.at(trueId.first)/3;

      pfpTree->Fill();

      TruthMatch match(pfp->Self(), trueId.first, pfpPdg[fPFParticleLabel], pfpNumHits[fPFParticleLabel],
          pfpHitComp[fPFParticleLabel], pfpHitPurity[fPFParticleLabel],
          pfpEnergyComp[fPFParticleLabel], pfpEnergyPurity[fPFParticleLabel],
          pfpHitSPRatio[fPFParticleLabel], pfpTrackScore[fPFParticleLabel]);

      pfpTruthMatchMap[fPFParticleLabel].push_back(match);
    }
    eventTree->Fill();
  }

  for (auto const& [trueId, trueDaughters]: truePrimaries){

    clearTrueTree();

    numTrueHits = truePrimaryHits.at(trueId);
    if (!numTrueHits)
      continue;

    const simb::MCParticle* trueParticle = trueParticles.at(trueId);
    truePdg = trueParticle->PdgCode();
    trueEnergy = (trueParticle->P()*trueParticle->P())/(2*trueParticle->Mass());
    trueMomentum = trueParticle->P();
    trueDepositedEnergy = truePrimaryEnergies.at(trueId)/3;
    trueDepositedEnergyInHits = truePrimaryHitEnergies.at(trueId)/3;
    trueProcess = trueParticle->Process();

    auto const trueMother = trueParticles.find(trueParticle->Mother());
    if (trueMother != trueParticles.end()){
      motherPdg = trueMother->second->PdgCode();
    }

    if (trueProcess=="primary"){
      std::cout << "True Particle: " << truePdg << " with true deposited energy: "
        << trueDepositedEnergy << std::endl;
    }
    for (auto const fPFParticleLabel: fPFParticleLabels){
      std::vector<TruthMatch> pfpMatches;
      unsigned int pfpTracks(0), pfpShowers(0);
      for (TruthMatch const& truthMatch: pfpTruthMatchMap[fPFParticleLabel]){
        // Check the pfp is matched to the true particle we care about
        if (trueId==truthMatch.mTrueId){
          pfpMatches.push_back(truthMatch);
          if (truthMatch.mRecoPdg==11)
            ++pfpShowers;
          if (truthMatch.mRecoPdg==13)
            ++pfpTracks;
        }
      }
      recoPFPs[fPFParticleLabel] = pfpTracks + pfpShowers;
      recoPFPTracks[fPFParticleLabel] = pfpTracks;
      recoPFPShowers[fPFParticleLabel] = pfpShowers;

      if (pfpMatches.size()>0){
        TruthMatch bestMatch;
        // Find the "Best Match" as the pfp with the highest completeness
        for (TruthMatch const& pfpMatch: pfpMatches){
          if (pfpMatch > bestMatch){
            bestMatch = pfpMatch;
          }
        }
        recoPdg[fPFParticleLabel]        = bestMatch.mRecoPdg;
        recoHits[fPFParticleLabel]       = bestMatch.mNumHits;
        hitComp[fPFParticleLabel]        = bestMatch.mHitComp;
        hitPurity[fPFParticleLabel]      = bestMatch.mHitPurity;
        energyComp[fPFParticleLabel]     = bestMatch.mEnergyComp;
        energyPurity[fPFParticleLabel]   = bestMatch.mEnergyPurity;
        hitSPRatio[fPFParticleLabel]     = bestMatch.mHitSPRatio;
        recoTrackScore[fPFParticleLabel] = bestMatch.mTrackScore;
      }
      if (trueProcess=="primary"){
        std::cout << "  " << std::setw(maxLabelLength) << fPFParticleLabel
          << ": particles: " << pfpTracks + pfpShowers << "(" << pfpTracks << "+" << pfpShowers << ")"
          << " and reco pdg: " << bestMatch.mRecoPdg
          << " and Comp: " << bestMatch.mHitComp
          << " and Purity: " << bestMatch.mHitPurity
          << " and Track Score: " << bestMatch.mTrackScore
          << std::endl;

        if (truePdg==2212 && recoPdg[fPFParticleLabel]==11){
          std::cout << "TEST123: " << evt.id() << std::endl;

        }
      }
    }
    trueTree->Fill();
  }
  std::cout<<"\n"<<std::endl;
}

std::map<int, int> ana::PFPValidation::GetTruePrimaryHits(
    const detinfo::DetectorClocksData& clockData,
    std::map<int,const simb::MCParticle*>& trueParticles,
    std::map<int,std::vector<int> >& truePrimaries,
    std::vector< art::Ptr< recob::Hit> >& allHits){

  std::map<int,int> trueParticleHits;
  for (const auto& hit: allHits){
    int trackID     = 0;
    float hitEnergy = 0;
    std::vector<sim::TrackIDE> trackIDEs = bt_serv->HitToTrackIDEs(clockData, hit);
    for (const auto& ide: trackIDEs) {
      if (ide.energy > hitEnergy){
        hitEnergy = ide.energy;
        trackID   = TMath::Abs(ide.trackID);
      }
    }
    ++trueParticleHits[trackID];
  }

  std::map<int, int> truePrimaryHits;
  for (const auto& truePrimary: truePrimaries){
    for (const auto& trueDaughter: truePrimary.second){
      truePrimaryHits[truePrimary.first] += trueParticleHits[trueDaughter];
    }
  }
  return truePrimaryHits;
}

std::map<int, float> ana::PFPValidation::GetTruePrimaryEnergies(
    std::map<int,const simb::MCParticle*>& trueParticles,
    std::map<int,std::vector<int> >& truePrimaries,
    std::vector<art::Ptr<sim::SimChannel> > simChannels){

  std::map<int, float> trueParticleEnergies;

  for (const auto& simChannel: simChannels){
    auto tdcideMap = simChannel->TDCIDEMap();
    for (const auto& [tdc, ideVec]: tdcideMap){
      for (const auto& ide: ideVec){
        trueParticleEnergies[TMath::Abs(ide.trackID)] += ide.energy;
      }
    }
  }
  std::map<int, float> truePrimaryEnergies;
  for (const auto& truePrimary: truePrimaries){
    for (const auto& trueDaughter: truePrimary.second){
      truePrimaryEnergies[truePrimary.first] += trueParticleEnergies[trueDaughter];
    }
  }
  return truePrimaryEnergies;
}

std::map<int, float> ana::PFPValidation::GetTruePrimaryHitEnergies(
    const detinfo::DetectorClocksData& clockData,
    std::map<int,const simb::MCParticle*>& trueParticles,
    std::map<int,std::vector<int> >& truePrimaries,
    std::vector< art::Ptr< recob::Hit> >& allHits){

  std::map<int, float> trueParticleEnergies;
  for (const auto& hit: allHits){
    std::vector<sim::TrackIDE> trackIDEs = bt_serv->HitToTrackIDEs(clockData, hit);
    for (const auto& ide: trackIDEs) {
      trueParticleEnergies[TMath::Abs(ide.trackID)] += ide.energy;
    }
  }
  std::map<int, float> truePrimaryHitEnergies;
  for (const auto& truePrimary: truePrimaries){
    for (const auto& trueDaughter: truePrimary.second){
      truePrimaryHitEnergies[truePrimary.first] += trueParticleEnergies[trueDaughter];
    }
  }
  return truePrimaryHitEnergies;
}

float ana::PFPValidation::GetTotalEnergyInHits(
    const detinfo::DetectorClocksData& clockData,
    std::vector<art::Ptr<recob::Hit> >  hits){

  float energy = 0;
  for (auto const& hit:hits){
    std::vector<sim::TrackIDE> trackIDEs = bt_serv->HitToTrackIDEs(clockData, hit);
    for (auto const& trackIDE: trackIDEs){
      energy+=trackIDE.energy;
    }
  }
  return energy;
}

template <class T>
void ana::PFPValidation::initTree(TTree* Tree, std::string branchName,
    std::map<std::string, T>& Metric, std::vector<std::string> fPFParticleLabels){

  for (auto const& fPFParticleLabel: fPFParticleLabels){
    std::string branchString = branchName + "_" + fPFParticleLabel;
    const char* branchChar   = branchString.c_str();
    Tree->Branch(branchChar, &Metric[fPFParticleLabel], 32000, 0);
  }
}

void ana::PFPValidation::clearPFPTree(std::string PFParticleLabel){
  pfpPdg[PFParticleLabel]                 = -999;
  pfpTruePdg[PFParticleLabel]             = -999;
  pfpTrueEnergy[PFParticleLabel]          = -999;
  pfpTrueMomentum[PFParticleLabel]        = -999;
  pfpTrueDepositedEnergy[PFParticleLabel] = -999;
  pfpTrackScore[PFParticleLabel]          = -999;
  pfpNumHits[PFParticleLabel]             = -999;
  pfpHitPurity[PFParticleLabel]           = -999;
  pfpHitComp[PFParticleLabel]             = -999;
  pfpEnergyPurity[PFParticleLabel]        = -999;
  pfpEnergyComp[PFParticleLabel]          = -999;
  pfpHitSPRatio[PFParticleLabel]          = -999;
}

void ana::PFPValidation::clearTrueTree(){
  truePdg                   = -999;
  motherPdg                 = -999;
  numTrueHits               = -999;
  trueEnergy                = -999;
  trueMomentum              = -999;
  trueDepositedEnergy       = -999;
  trueDepositedEnergyInHits = -999;
  trueProcess               = "";
  for (auto const& fPFParticleLabel: fPFParticleLabels){
    recoPFPs[fPFParticleLabel]       = -999;
    recoPFPTracks[fPFParticleLabel]  = -999;
    recoPFPShowers[fPFParticleLabel] = -999;
    recoPdg[fPFParticleLabel]        = -999;
    recoTrackScore[fPFParticleLabel] = -999;
    recoHits[fPFParticleLabel]       = -999;
    hitPurity[fPFParticleLabel]      = -999;
    energyPurity[fPFParticleLabel]   = -999;
    hitComp[fPFParticleLabel]        = -999;
    energyComp[fPFParticleLabel]     = -999;
    hitSPRatio[fPFParticleLabel]     = -999;
  }
}

void ana::PFPValidation::clearEventTree(){
  for (auto const& fPFParticleLabel: fPFParticleLabels){
    eventNumPFP[fPFParticleLabel]       = 0;
    eventNumPFPNu[fPFParticleLabel]     = 0;
    eventNumPFPTrack[fPFParticleLabel]  = 0;
    eventNumPFPShower[fPFParticleLabel] = 0;
  }
}

DEFINE_ART_MODULE(ana::PFPValidation)
