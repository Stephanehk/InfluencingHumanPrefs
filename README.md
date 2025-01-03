# Influencing Human Preferences to Conform to Different Preference Models
This is the official implementation for the paper *Influencing Human Preferences to Conform to Different Preference Models*.

## Setup

To install all necessary libraries, use ```pip install -r requirements.txt```.

## Dataset of Human Preferences
The dataset of human preferences collected from each experiment and condition is located in ```data/human_data```. The condition and experiment are specified via the file name. Files ending with ```_full_filtered_Y.npy``` contain the dataset of human preferences after filtering out subjects for task comprehension and attentiveness. The files ending with ```_full_filtered_X.npy``` contain the corresponding segment pair statistics, formatted as ```[difference in partial return, difference in start state value, difference in end state value]``` for each sample. The file ending in ```_full_filtered_segment_pairs.npy``` contains the corresponding segment pairs, formatted as a sequence of states and actions in the delivery task MDP, which itself can be found in ```data/delivery_mdp```. 

For example, to load the preference data for the Question expirement and the Partial-Return-Question condition---where subjects are influenced towards the partial return model by changing the preference elicitation question---the files of interest are ```data/human_data/Question-Control_full_filtered_Y.npy``` which contains the preferences over the segment pairs in  ```data/human_data/Question-Control_full_filtered_segment_pairs.npy``` with segment pair statistics saved in  ```data/human_data/Question-Control_full_filtered_X.npy```


Note that all other files in the ```data/human_data``` directory that end in ```_Y.npy``` contain synthetic preferences generated by various preference models for each segment pair.

## Mapping from Paper to Code

To run the likelihood analysis used to generate Figures 5, 7, and 9, run ```scripts/run_likelihood_analysis.sh```. To run the accuracy analysis detailed in Appendix I, run ```scripts/run_accuracy_analysis.sh```. Note that these scripts will also run the corresponding statistical tests.


To run the reward learning analysis used to generate Figures 6, 8, and 10, run ```scripts/run_reward_learning.sh```.
