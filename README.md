---------------------------------------------------
This Repository contains the Algorithms explained in<br />
Cosentino, Oberhauser, Abate<br />
Acceleration of Descent-based Optimization Algorithms via Caratheodory’s Theorem<br />
---------------------------------------------------

The files are divided in the following way:<br />
- The ipython notebooks contain the experiments to be run;<br />
- The files *.py are the libraries with all the necessary functions.<br />

Some general notes:<br />
- The names of the ipynb files refer directly to the experiments in the cited work.<br />
- The last cells of the notebooks produce the pictures of the pdf.<br />
- To reduce the running time the parameters can be easiliy changed, e.g. decreasing N, n or sample.<br />

----------------------------------------------------
SPECIAL NOTE TO RUN - Comparison_GD_vs_CaGD.ipynb 
----------------------------------------------------
This is set to reproduce the experiment of the first part (sin), if you
want to reproduce the other experiments you have to comment/uncomment the respective parts of the code
as indicated.

---------------------------------------------------
LIBRARIES - CaGD_log.py, CaGD_ls.py
---------------------------------------------------
They contain the algorithms relative to the the acceleration of the GD-based methods via the 
Caratheodory's Theorem. The first is specialised in the logistic regression, while the second in 
the least-squares case. The second contains also functions which replicate the behaviour of ADAM and SAG.
Only the functions in CaGD_ls.py are parallelized.
Requirements: recombination.py

----------------------------------------------------
LIBRARIES - recombination.py 
----------------------------------------------------
It contains the algorithms relative to the reduction of the measure presented in 
COSENTINO, OBERHAUSER, ABATE - "A randomized algorithm to reduce the support of discrete measures".

---------------------------------------------------
DATASETS
---------------------------------------------------
Please, to run the experiments donwload the following dataset and put them in the /Dataset folder:
	- 3D_spatial_network.txt - 
      https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt
	- household_power_consumption.txt - 
      https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
      (extract the .txt file)
    - NY_train.csv - 
      https://www.kaggle.com/c/nyc-taxi-trip-duration/data?select=train.zip
      (extract the .csv file and rename it to NY_train.csv)

