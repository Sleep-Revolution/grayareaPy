# grayareaPy
Method to analyze uncertainty into ensemble of scorers from sleep studies.

## Scientific purpose
<br/>
In this github repository, codes are proposed to deploy a method composed of a filtering part and a mixture model to fit ensemble data sampled from a vector of E categorial variable $Y_{n}=(Y_{n1},\cdots,Y_{nE})$, $\forall+n+\in\\{1,...,N\\}$ where E denotes the ensemble size and N the sample size.

The proposed mixture model are based on a multinomial or dirichlet (work in progress for dirichlet).

<div class="image-wrapper" >
    <img src="/figures/example.png" alt=""/>
  </a>
      <p class="image-caption">Figure 1: Comparison between real and simulated data with equal proportion mixture components. Top row: Hypnograms of real ensemble on the left and simulated ensemble with their associated clusters on the right and over 25 epochs. Bottom row: Density of the coefficient of unalikeability estimated on real dataset on the left and on simulated data on the right. Red doted lines symbolized the threshold $\delta=2$.</p>
</div>

