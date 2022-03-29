#! /bin/bash

# Loop over all installed fcl files.
# Ignoring known broken fcl files.

find $MRB_BUILDDIR/sbndcode/fcl -name \*.fcl \
  \! -name ANA.fcl \
  \! -name prodgenie_sbnd_proj.fcl \
  \! -name prodmarley_sbnd_1event.fcl \
  \! -name sbnd_buildopticallibrary.fcl \
  \! -name prodsingle_sbnd_crt.fcl \
  \! -name prodsingle_fastoptical_sbnd.fcl \
  \! -name prodsingle_fastoptical2.fcl \
  \! -name prodoverlay_corsika_cosmics_proton_genie_rockbox_intrnue_sbnd.fcl \
  \! -name prodmarley_nue_fermidirac.fcl \
  \! -name g4_noophybrid_sbnd.fcl \
  \! -name legacy_g4_5ms_electron_lifetime.fcl \
  \! -name legacy_g4_enable_doublespacecharge.fcl \
  \! -name anatree_prodoverlay_corsika_genie_3drift_windows.fcl \
  \! -name standard_reco_sbnd.fcl \
  \! -name reco_calorimetry_workshop2020_sbnd.fcl \
  \! -name spacepoint_sbnd.fcl \
  \! -name trackfinderalgorithms_sbnd.fcl \
  \! -name set_flux_config_b.fcl \
  \! -name set_flux_config_c.fcl \
  \! -name set_flux_config_d.fcl \
  \! -name set_flux_config_e.fcl \
  \! -name set_flux_config_f.fcl \
  \! -name set_flux_config_g.fcl \
  \! -name set_flux_config_h.fcl \
  \! -name set_genie_rotatedbuckets.fcl \
  \! -name set_genie_filter_ccpi0.fcl \
  \! -name detsimmodules_sbnd.fcl \
  \! -name mc_info_extraction.fcl \
  -print | while read fcl
do
  echo "Testing fcl file $fcl"

  # Parse this fcl file.

  fclout=`basename ${fcl}`.out
  larout=`basename ${fcl}`.lar.out
  larerr=`basename ${fcl}`.lar.err
  lar -c $fcl --debug-config $fclout > $larout 2> $larerr

  # Exit status 1 counts as success.
  # Any other exit status exit immediately.

  stat=$?
  if [ $stat -ne 0 -a $stat -ne 1 ]; then
    echo "Error parsing ${fcl}."
    exit $stat
  fi

  # Check for certain kinds of diagnostic output.

  if egrep -iq 'deprecated|no longer supported' $larerr; then
    echo "Deprecated fcl construct found in ${fcl}."
    exit 1
  fi

done
