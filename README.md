# EvaluatingHigherLevelSymbolicFeatures

This is the repository for the paper: "Evaluating Higher-Level and Symbolic Features in Deep Learning on Time Series: Towards Simpler Explainability"

It provides examples on how the feature extractors and symbolizing strategies can be used with a Transformer and a CNN model on time series data.

### Data

All univariate UCR & UEA datasets are supported.

### Dependencies installation guide

Use the environment.yaml file to create a conda environment:

conda env create -f environment.yaml


### How to run

You can run single experimentes by using the test.ipynb notebook or run multiple parameter combinations by starting SEML experiments

#### Just for testing

1. Go to test.ipynb
2. Change parameters
3. Run the cell

#### Multiple experiment settings with SEML

1. Set up seml with seml configure <font size="6"> (you need a mongoDB server for this and the results will be saved a in separate file, however seml does a really well job in managing the parameter combinations in combination with slurm) </font>
2. Configure the yaml file you want to run. Probably you only need to change the number of maximal parallel experiments ('experiments_per_job' and 'max_simultaneous_jobs') and the memory and cpu use ('mem' and 'cpus-per-task').
3. Add and start the seml experiment. For example like this:
	1. seml experiments add config.yaml
	2. seml experiments start
4. Check with "seml experiments status" until all your experiments are finished 
5. You can find the results in the results folder. They can be further evaluated with plotter.py

## Cite and publications

This code represents the used models for the following publication:<br>
"Evaluating Higher-Level and Symbolic Features in Deep Learning on Time Series: Towards Simpler Explainability" (TODO Link)


If you use, build upon this work or if it helped in any other way, please cite the linked publication.