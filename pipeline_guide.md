This pipeline guide describes the processing and analysis steps which lead to the findings by Yiu, Leuteb and Leibold (2022). See

<div class="csl-entry">Yiu, Y.-H., Leutgeb, J. K., &#38; Leibold, C. (2022). Directional Tuning of Phase Precession Properties in the Hippocampus. <i>The Journal of Neuroscience</i>, <i>42</i>(11), 2282–2297. https://doi.org/10.1523/jneurosci.1569-21.2021</div>

# Step 1 Matlab processing
Electrophysiological data recorded by Mankin is first processed by a Matlab script.
The script mainly segments and identifies the place fields. Pair identification is also done here.
1. Run ./matlab/emankin_preprocessing.m to pre-process the raw place cell data at ./data/emankin/
2. It generates the matlab structure data file ./data/emankin/emankindata_processed.mat

# Step 2 Conversion to python-readable data
The matlab data is then converted to a python dataframe.
1. Run ./Convert_mat2pickledDF.py
2. It generates a data file at ./data/emankindata_processed_withwave.pickle

# Step 3A Processing for the single-field analysis
The python-readable data is processed to produce computed results for data analysis and visualization.
This step segments the passes, classifies phase precessions and computes rate/precession directionality.
Each row of the dataframe contains the feature of one single field.
1. Run ./SingleField_preprocess.py
2. Data file ./results/emankin/singlefield_df.pickle is generated

Note: I ran ./determine_spike_counts.py beforehand in order to determine the spike count threshold for low-spike passes in this analysis.

# Step 3B Processing for the pair analysis
The python-readable data is processed to produce computed results for data analysis and visualization.
This step works on the identified pairs. All single-field features in Step 3A are computed again for each single field in the pairs.
The pair-specific features are also obtained, including pass segments across the pair, pass directions (AB or BA), pair correlations...etc
Each row of the dataframe contains the features of a pair.
1. Run ./PairField_preprocess.py
2. Data file ./results/emankin/pairfield_df_addedpf.pickle is generated

# Step 4 Single-field and Pair analyses
This part computes the results and plots figure 1-9
1. Run ./SingleField_Analysis.py and ./PairField_Analysis.py
2. Figures are plotted and saved in ./writting/figures/
3. Statistical test results are saved in ./writting/stats/


# Step 5 Control experiment by Romani and Tsodyks (2015) model
This part simulates the model, pre-processes and analyzes the results (Fig 10 and 11)
1. Run ./RomaniModel_Simulate.py and produce spike data file at ./results/sim/raw/squarematch.pickle
2. Run ./RomaniModel_Preprocess.py and produce preprocessed files at results/sim/singlefield_df.pickle and results/sim/pairfield_df.pickle
3. Run ./RomaniModel_Analysis.py and produce figure 10 and 11 at ./writting/figures/ , also statistical results at ./writting/stats/

For the details of the model, please see 
<div class="csl-entry">Romani, S., &#38; Tsodyks, M. (2015). Short-term plasticity based network model of place cells dynamics. <i>Hippocampus</i>, <i>25</i>(1), 94–105.</div>