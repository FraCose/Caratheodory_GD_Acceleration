---------------------------------------------------------
This Repository contains the Algorithms explained in<br />
Cosentino, Oberhauser, Abate<br />
"Caratheodory Sampling for Stochastic Gradient Descent"<br />
---------------------------------------------------------

The files are divided in the following way:<br />
- The ipython notebooks contain the experiments to be run;<br />
- The files *.py are the libraries with all the necessary functions.<br />

Some general notes:<br />
- The names of the ipynb files refer directly to the experiments in the cited work.<br />
- The last cells of the notebooks produce the pictures of the pdf, except for plots_rules.ipynb.<br />
- To reduce the running time the parameters can be easily changed, e.g. decreasing N, n or sample.<br />

---------------------------------------------------------
Libraries - CaGD_log.py, CaGD_ls.py
---------------------------------------------------------
They contain the algorithms relative to the acceleration of the GD-based methods via the<br />
Caratheodory's Theorem. The first is specialised in the logistic regression, while the second in<br />
the least-squares case. The second contains also functions which replicate the behaviour of ADAM and SAG.<br />
Only the functions in CaGD_ls.py are parallelized.<br />
Requirement: recombination.py.

---------------------------------------------------------
Libraries - train.py, Create_dataset_A.m, src folder
---------------------------------------------------------
These are the files necessary for the experiments done using the rules of [Nutini et al.].<br />
    - Create_dataset_A.m creates the Dataset A of [Nutini et al.] using Matlab. <br />
    - Train.py contains the functions which are the skeleton of the optimization procedure. <br />
      It corresponds to trainval.py in [Nutini URL]. We have :<br />
        •	removed dependencies not relevant for our experiments;<br />
        •	added the skeleton for the optimization procedure using the Caratheodory Sampling Procedure.<br />
    - src/losses.py follows the same logic as losses.py from [Nutini URL]. <br />
      We have kept only the least-squares object and we have modified it because the Cartheodory <br />
      Sampling procedure requires the gradient for any sample.<br />
    - src/update_rules/update_rules.py the same structure of the same file from [Nutini URL]. <br />
      The function update_Caratheodory(…) is the same as update(…) in the cited repository. We added the functions:<br />
      update_Caratheodory, recomb_step, Caratheodory_Acceleration.<br />
    - The rest of the files is the same as [Nutini URL].<br />

[Nutini et al.] Julie Nutini, Issam Laradji, and Mark Schmidt - "Let’s make block coordinate<br />
descent go fast: Faster greedy rules, message-passing, active-set complexity, and<br />
superlinear convergence", arXiv preprint arXiv:1712.08859, 2017.<br />
[Nutini URL] https://github.com/IssamLaradji/BlockCoordinateDescent


----------------------------------------------------------
Library - recombination.py
----------------------------------------------------------
It contains the algorithms relative to the reduction of the measure presented in <br />
COSENTINO, OBERHAUSER, ABATE - "A randomized algorithm to reduce the support of discrete measures",<br />
NeurIPS 2020, Available at https://github.com/FraCose/Recombination_Random_Algos

----------------------------------------------------------
Special Note to run the experiments
----------------------------------------------------------
The notebooks "CaGD_paths.ipynb" and "Comparison_GD_vs_CaGD.ipynb" contain multiple experiments.<br />
You have to comment/uncomment the respective parts of the code as indicated to reproduce the <br />
wanted experiments.

---------------------------------------------------------
To Run the Experiments - Datasets
---------------------------------------------------------
To run the experiments, the following dataset need to be donwloaded and saved in the /Datasets folder:<br />
- 3D_spatial_network.txt - <br />
https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt<br />
- household_power_consumption.txt - <br />
https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip<br />
(extract the .txt file)<br />
- NY_train.csv - <br />
https://www.kaggle.com/c/nyc-taxi-trip-duration/data?select=train.zip<br />
(extract the .csv file and rename it to NY_train.csv)<br />

---------------------------------------------------------
Funding
---------------------------------------------------------
The authors want to thank The Alan Turing Institute and the University of Oxford<br />
for the financial support given. FC is supported by The Alan Turing Institute, TU/C/000021,<br />
under the EPSRC Grant No. EP/N510129/1. HO is supported by the EPSRC grant Datasig<br />
[EP/S026347/1], The Alan Turing Institute, and the Oxford-Man Institute.
