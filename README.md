# Bundel Recommendation

## Worker Basic Information
Figure 1 shows the distribution of workers' age, education, country, occupation, gender and shopping frequency for the two batches.


## Parameter Tuning And Settings For Bundle Detection
A grid search in {0.0001, 0.001, 0.01} is applied to find out the optimal settings for *support* and *confidence*, and both are set as 0.001 across the three domains.


## Parameter Tuning And Settings For Bundle Completion
The dimension (*d*) of item and bundle representations for all methods is 20. Grid search is adopted to find out the best settings for other key parameters. In particular, learning rate ![](https://latex.codecogs.com/svg.image?(\eta)) and regularization coefficient ![](https://latex.codecogs.com/svg.image?(\lambda)) are searched in {0.0001, 0.001, 0.01}; the number of neighbors (*K*) in ItemKNN is searched in {10, 20, 30, 50}; the weight of KL divergence ![](https://latex.codecogs.com/svg.image?(\alpha)) in VAE is searched in {0.001, 0.01, 0.1}; and the batch size is searched in {64, 128, 256}. The optimal parameter settings are shown in Table 1. 


## Parameter Tuning And Settings For Bundle Ranking
The dimension (*d*) of representations is set as 20. We apply a same grid search for ![](https://latex.codecogs.com/svg.image?(\eta)), ![](https://latex.codecogs.com/svg.image?(\lambda)), ![](https://latex.codecogs.com/svg.image?K) and batch size as in bundle completion. Besides, the predictive layer *D* for AttList is searched from {20, 50, 100}; the node and message dropout rate for GCN and BGCN is searched in {0, 0.1, 0.3, 0.5}. As the training complexity for GCN and BGCN is quite high, we set the batch size as 2048 as suggested by the original paper. The optimal parameter settings are presented in Table 2. Note that the parameter settings for BGCN is the version without pre-training (i.e. ![](https://latex.codecogs.com/svg.image?BGCN_{w/o\&space;pre})).  

## Acknowledgements
