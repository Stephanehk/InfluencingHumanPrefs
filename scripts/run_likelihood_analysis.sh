python3 -m prescriptive_effort.analysis.prescriptive_manual_likelihood_analysis --conditions "Privileged-Control, Pr-Privileged, Regret-Privileged"
python3 -m prescriptive_effort.analysis.mann_whitney_test

python3 -m prescriptive_effort.analysis.prescriptive_manual_likelihood_analysis --conditions "Trained-Control, Pr-Trained, Regret-Trained"
python3 -m prescriptive_effort.analysis.prescriptive_manual_likelihood_analysis --conditions "Question-Control, Pr-Question, Regret-Question"
python3 -m prescriptive_effort.analysis.wilcoxon_test

