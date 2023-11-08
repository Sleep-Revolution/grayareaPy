# grayareaPy
Method for clustering categorical ensemble data set and resuming clusters in two supra-clusters according to their dispersion.

## Scientific purpose

This github repository proposes codes to apply a clustering method based on two steps and a mixture model to fit an ensemble of categorical variables $(y_{n1},\cdots,y_{nE})$, $\forall n \in\\{1,...,N\\}$, where E is the ensemble size and N is the sample size. The aim of this method is to first automatically select and fit an appropriate number of clusters using the BIC criteria and then a threshold is selected using a distance measurement of cluster dispersion to generate two supra-clusters with high and low perturbed clusters !(more details in [](https://www.sciencedirect.com/science/article/pii/S0377221723007567)).

The proposed mixture model is based on a multinomial or Dirichlet distribution (work in progress for Dirichlet). 

The Python scripts and notebook files of this project are organised as follows:
* **/src/MixtureModels.py**: All method components (filtering, mixture model, EM algorithm, initialisation method);
* **/src/SimuMM.py**: Class to simulate, run and validate data and method over different parameter sets and validation criteria;
* **/SimuMM.ipynb**: Experiment parameter sets and call of simulation method;

## Simulation Study

Experimental examples are provided in the operational domain of human sleep stages. The aim is to validate the proposed method on a simulated ensemble sleep stage dataset of $E=10$ categorical variables with 4 classes corresponding to $M=5$ multinomial mixture components. The method is evaluated in terms of its ability to select the correct number of mixture components and the accurate threshold to construct the two supra-clusters. Then, the method is assessed by evulating the accuracy of the estimation of the the model parameters.

Experiments with different data sets and method parameterisations are proposed to be studied here.

<div class="image-wrapper" >
    <img src="/figure/table1.png" alt=""/>
</div>

<div class="image-wrapper" >
    <img src="/figure/SimuSelec0.png" alt=""/>
  </a>
      <p class="image-caption">Figure 1: .</p>
</div>

<div class="image-wrapper" >
    <img src="/figure/SimuSelec1.png" alt=""/>
  </a>
      <p class="image-caption">Figure 2: .</p>
</div>


<div class="image-wrapper" >
    <img src="/figure/SimuSelec2.png" alt=""/>
  </a>
      <p class="image-caption">Figure 3: .</p>
</div>

<div class="image-wrapper" >
    <img src="/figure/SimuSelec3.png" alt=""/>
  </a>
      <p class="image-caption">Figure 4: .</p>
</div>
