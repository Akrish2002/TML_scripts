#!/bin/bash
#module load miniforge3/23.11.0-0
#source activate ~/source_code/envs/fpcsl/

#Singlephase 

##Meanflow profile
#snapshots=(16000 20000 23500 27500 30000)
#inputs=()
#for ts in "${snapshots[@]}"; do
#    inputs_for_mean_flow_profile+=("data/Aurora/SinglePhase_mu_ratio1.0/mean_flow_profile/mean_flow_profile_n1024_ts${ts}.npz")
#done
#python3 "$TML_scripts_postprocessing/plot_all.py" \
#    --function mean_flow_profile \
#    --singlephase \
#    --inputs "${inputs_for_mean_flow_profile[@]}" \
#    --output-path \
#        plots/Aurora/mean_flow_profile.png \
#    --zoom 0 \
#    --zoom-window 1.5

##Dissipation
#snapshots=(16000 20000 22500)
#for ts in "${snapshots[@]}"; do
#    inputs_for_mean_flow_profile+=("data/Aurora/SinglePhase_mu_ratio1.0/mean_flow_profile/mean_flow_profile_n1024_ts${ts}.npz")
#    inputs_for_dissipation+=("data/Aurora/SinglePhase_mu_ratio1.0/dissipation/Dissipation_n1024_ts${ts}.npz")
#done
#python3 "$TML_scripts_postprocessing/plot_all.py" \
#    --function dissipation \
#    --singlephase \
#    --inputs "${inputs_for_dissipation[@]}" \
#    --mean-flow-inputs "${inputs_for_mean_flow_profile[@]}" \
#    --output-path \
#        plots/Aurora/dissipation.png \
#    --zoom 0 \
#    --zoom-window 1.25

##TKE
#snapshots=(16000 20000 22500)
#for ts in "${snapshots[@]}"; do
#    inputs_for_mean_flow_profile+=("data/Aurora/SinglePhase_mu_ratio1.0/mean_flow_profile/mean_flow_profile_n1024_ts${ts}.npz")
#    inputs_for_TKE+=("data/Aurora/SinglePhase_mu_ratio1.0/TKE/TKE_n1024_ts${ts}.npz")
#done
#python3 "$TML_scripts_postprocessing/plot_all.py" \
#    --function TKE \
#    --singlephase \
#    --inputs "${inputs_for_TKE[@]}" \
#    --mean-flow-inputs "${inputs_for_mean_flow_profile[@]}" \
#    --output-path \
#        plots/Aurora/TKE.png \
#    --zoom 0 \
#    --zoom-window 1.25

##TKE Budget
#snapshots=(22500)
#snapshots=(16500 20000 22500)
#for ts in "${snapshots[@]}"; do
#    inputs_for_mean_flow_profile+=("data/Aurora/SinglePhase_mu_ratio1.0/mean_flow_profile/mean_flow_profile_n1024_ts${ts}.npz")
#    inputs_for_TKE_budget+=("data/Aurora/SinglePhase_mu_ratio1.0/TKE_budget/TKE_Budget_n1024_ts${ts}.npz")
#done
#python3 "$TML_scripts_postprocessing/plot_all.py" \
#    --function TKE_Budget \
#    --singlephase \
#    --inputs "${inputs_for_TKE_budget[@]}" \
#    --mean-flow-inputs "${inputs_for_mean_flow_profile[@]}" \
#    --output-path \
#        plots/Aurora/TKE_budget.png \
#    --zoom 0 \
#    --zoom-window 1.25 

##Reynolds Stresses
#snapshots=(16000 20000 23500 27500 30000)
#OUTPUTPATH="plots/Aurora"
#for ts in "${snapshots[@]}"; do
#    inputs_for_reynolds_stresses+=("data/Aurora/SinglePhase_mu_ratio1.0/reynolds_stresses/Reynolds_stresses_n1024_ts${ts}.npz")
#done
##All components
#for components in $(seq 1 1 4) ; do
#    python3 "$TML_scripts_postprocessing/plot_all.py" \
#        --function reynolds_stresses \
#        --reynolds_stress_components ${components} \
#        --singlephase \
#        --inputs "${inputs_for_reynolds_stresses[@]}" \
#        --output-path "$OUTPUTPATH/reynolds_stresses_${components}.png" \
#        --zoom 0 \
#        --zoom-window 1.00 
#done

##Total dissipation of TKE


##kx spectra
#snapshots=(16000 20000 23500 27500 30000)
#OUTPUTPATH="plots/Aurora"
#for ts in "${snapshots[@]}"; do
#    inputs_for_kx_spectra+=("data/Aurora/SinglePhase_mu_ratio1.0/kx_spectra/kx_spectra_n1024_ts${ts}.npz")
#done
#for components in $(seq 1 1 4) ; do
#    python3 "$TML_scripts_postprocessing/plot_all.py" \
#        --function kx_spectra \
#        --spectra_components ${components} \
#        --singlephase \
#        --inputs "${inputs_for_kx_spectra[@]}" \
#        --output-path "$OUTPUTPATH/kx_spectra_${components}.png" 
#done

##kz spectra
#snapshots=(16000 20000 23500 27500 30000)
#OUTPUTPATH="plots/Aurora"
#for ts in "${snapshots[@]}"; do
#    inputs_for_kz_spectra+=("data/Aurora/SinglePhase_mu_ratio1.0/kz_spectra/kz_spectra_n1024_ts${ts}.npz")
#done
#for components in $(seq 1 1 4) ; do
#    python3 "$TML_scripts_postprocessing/plot_all.py" \
#        --function kz_spectra \
#        --spectra_components ${components} \
#        --singlephase \
#        --inputs "${inputs_for_kz_spectra[@]}" \
#        --output-path "$OUTPUTPATH/kz_spectra_${components}.png" 
#done

##Spanwise two-point correlation: ensemble-averaged plot
snapshots=(16000 20000 23500 27500 30000)
OUTPUTPATH="plots/Aurora"
inputs_for_spanwise_two_point_correlation=()
for ts in "${snapshots[@]}"; do
    inputs_for_spanwise_two_point_correlation+=(
        "data/Aurora/SinglePhase_mu_ratio1.0/spanwise_two_point_correlation/spanwise_two_point_correlation_n1024_ts${ts}.npz"
    )
done

python3 "$TML_scripts_postprocessing/plot_all.py" \
    --function spanwise_two_point_correlation \
    --twopoint_correlation_components 1 2 3 4 \
    --singlephase \
    --ensemble-average \
    --inputs "${inputs_for_spanwise_two_point_correlation[@]}" \
    --output-path "${OUTPUTPATH}/spanwise_two_point_correlation_ensemble_avg_ts16000_30000.png"

##Streamwise two-point correlation: ensemble-averaged plot
snapshots=(16000 20000 23500 27500 30000)
OUTPUTPATH="plots/Aurora"
inputs_for_streamwise_two_point_correlation=()
for ts in "${snapshots[@]}"; do
    inputs_for_streamwise_two_point_correlation+=(
        "data/Aurora/SinglePhase_mu_ratio1.0/streamwise_two_point_correlation/streamwise_two_point_correlation_n1024_ts${ts}.npz"
    )
done

python3 "$TML_scripts_postprocessing/plot_all.py" \
    --function streamwise_two_point_correlation \
    --twopoint_correlation_components 1 2 3 4 \
    --singlephase \
    --ensemble-average \
    --inputs "${inputs_for_streamwise_two_point_correlation[@]}" \
    --output-path "${OUTPUTPATH}/streamwise_two_point_correlation_ensemble_avg_ts16000_30000.png"
