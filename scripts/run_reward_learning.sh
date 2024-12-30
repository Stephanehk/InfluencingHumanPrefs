python3 -m prescriptive_effort.analysis.reward_learning_multiple_seeds --force_cpu --conditions "Privileged-Control, Pr-Privileged, Regret-Privileged"
python3 -m prescriptive_effort.analysis.reward_learning_multiple_seeds --force_cpu --conditions "Trained-Control, Pr-Trained, Regret-Trained"
python3 -m prescriptive_effort.analysis.reward_learning_multiple_seeds --force_cpu --conditions "Question-Control, Pr-Question, Regret-Question"

python3 -m prescriptive_effort.analysis.plot_reward_learning_results_multiple_seeds
