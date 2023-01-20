# grayareaPy
Method for focusing and fitting a mixture model to perturbed elements of a categorical ensemble data set.

## Scientific purpose

This github repository proposes codes to apply a method consisting of a filtering part and a mixture model applied to an ensemble of categorical variables $(y_{n1},\cdots,y_{nE})$, $\forall n \in\\{1,...,N\\}$, where E is the ensemble size and N is the sample size. The aim is to remove the element where there is a high agreement between the categorical variables $y_{n1} \approx \cdots \approx y_{nE}$ and fit a mixture model to the remaining data. This method has been developed to highlight the uncertainty in human sleep stage classification in the field of sleep studies.

The method is parameterised with two main parameters. The first is a **threshold** $\in [0,1]$ to be set by the user to remove elements where the ensemble of categorical variables takes a majority of the same value. The threshold is linked to a measure of the dispersion of the ensemble of categorical variables. Its meaning is that the larger the threshold, the more elements are removed where the ensemble of categorical variables does not take a same value, thus reducing the uncertainty of the dataset. The threshold is set to 0.2 by default. The other is the number of mixture components $M$. By default, $M$ is automatically selected using BIC selection over multiple values for $M$. The EM algorithm is initialised by default with the k-means++ algorithm.

The proposed mixture model is based on a multinomial or Dirichlet distribution (work in progress for Dirichlet). 

The Python scripts and notebook files of this project are organised as follows:
* **/src/MixtureModels.py**: All method components (filtering, mixture model, EM algorithm, initialisation method);
* **/src/SimuMM.py**: Class to simulate, run and validate data and method over different parameter sets and validation criteria;
* **/SimuMM.ipynb**: Experiment parameter sets and call of simulation method;
* **/ResultPlot.ipynb**: All produced plots from experiment results.

## Simulation Results

<div class="image-wrapper" >
    <img src="/figure/parameters_table.png" alt=""/>
</div>


<div class="image-wrapper" >
    <img src="/figure/IntroSimu.png" alt=""/>
  </a>
      <p class="image-caption">Figure 1: Comparison between real and simulated data with equal proportion mixture components. Top row: Hypnograms of real ensemble on the left and simulated ensemble with their associated clusters on the right and over 25 epochs. Bottom row: Density of the coefficient of unalikeability estimated on real dataset on the left and on simulated data on the right. Red doted lines symbolized the threshold $\delta=2$.</p>
</div>

<div class="image-wrapper" >
    <img src="/figure/SimuRes.png" alt=""/>
  </a>
      <p class="image-caption">Figure 2: Estimated criteria over 30 repeated experiments of simulated dataset for different datasize $N$ along x-axis and different set of $pi$ for each columns. First row: Number of clusters selected by BIC estimation. Second row: Root mean square error (RMSE) estimated. Final row: estimation of the accuracy.</p>
</div>

<div class="image-wrapper" >
    <img src="/figure/CMSIMU.png" alt=""/>
  </a>
      <p class="image-caption">Figure 3: Confusion matrix by experiment, averaged over 30 randomly generated datasets and normalized. Each panel represent the averaged and normalized confusion matrix for one specific experiment.</p>
</div>
