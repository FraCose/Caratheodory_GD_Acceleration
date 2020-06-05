---------------------------------------------------
This Repository contains the Algorithms explained in<br />
Cosentino, Oberhauser, Abate<br />
"Acceleration of Descent-based Optimization Algorithms via Caratheodory’s Theorem"<br />
---------------------------------------------------

The files are divided in the following way:<br />
- The ipython notebooks contain the experiments to be run;<br />
- The files *.py are the libraries with all the necessary functions.<br />

Some general notes:<br />
- The names of the ipynb files refer directly to the experiments in the cited work.<br />
- The last cells of the notebooks produce the pictures of the pdf.<br />
- To reduce the running time the parameters can be easily changed, e.g. decreasing N, n or sample.<br />

---------------------------------------------------
Libraries - CaGD_log.py, CaGD_ls.py
---------------------------------------------------
They contain the algorithms relative to the acceleration of the GD-based methods via the<br /> 
Caratheodory's Theorem. The first is specialised in the logistic regression, while the second in<br /> 
the least-squares case. The second contains also functions which replicate the behaviour of ADAM and SAG.<br />
Only the functions in CaGD_ls.py are parallelized.<br />
Requirement: recombination.py.

----------------------------------------------------
Library - recombination.py 
----------------------------------------------------
It contains the algorithms relative to the reduction of the measure presented in <br /> 
COSENTINO, OBERHAUSER, ABATE - "A randomized algorithm to reduce the support of discrete measures",<br /> 
Available at https://github.com/FraCose/Recombination_Random_Algos

----------------------------------------------------
Special Note to run the experiments
----------------------------------------------------
The notebooks "CaGD_paths.ipynb" and "Comparison_GD_vs_CaGD.ipynb" contain multiple experiments.<br />
You have to comment/uncomment the respective parts of the code as indicated to reproduce the <br />
wanted experiments. <br />

---------------------------------------------------
To Run the Experiments - Datasets
---------------------------------------------------
To run the experiments, the following dataset need to be donwloaded and saved in the /Datasets folder:<br />
- 3D_spatial_network.txt - <br />
https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt<br />
- household_power_consumption.txt - <br />
https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip<br />
(extract the .txt file)<br />
- NY_train.csv - <br />
https://www.kaggle.com/c/nyc-taxi-trip-duration/data?select=train.zip<br />
(extract the .csv file and rename it to NY_train.csv)<br />

---------------------------------------------------
Funding
---------------------------------------------------
The authors want to thank The Alan Turing Institute and the University of Oxford<br /> 
for the financial support given. FC is supported by The Alan Turing Institute, TU/C/000021,<br />
under the EPSRC Grant No. EP/N510129/1. HO is supported by the EPSRC grant Datasig<br />
[EP/S026347/1], The Alan Turing Institute, and the Oxford-Man Institute.