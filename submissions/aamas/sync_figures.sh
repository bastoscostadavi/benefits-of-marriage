#!/usr/bin/env bash
# Copy the figures this paper uses out of results/ so the submission directory
# is self-contained (AAMAS wants a single zip). Re-run after scripts/run_all.py.
set -euo pipefail
cd "$(dirname "$0")"
for f in fig01_initial_histograms fig02_trajectories fig03_lambda_comparison \
         fig04_marriage_vs_none fig05_final_histograms fig06_mu_robustness \
         fig07_liked_disliked fig08_married_share fig09_first_step_distributions \
         fig10_pt_evolution fig11_meanfield_validation fig12_p_infinity \
         fig13_benchmarks fig14_number_of_partners; do
  cp "../../results/$f.pdf" "figures/$f.pdf"
done
echo "synced $(ls figures/*.pdf | wc -l | tr -d ' ') figures from results/"
