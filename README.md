# Non-Crossing Dual Neural Network
This is a repository in regards of the article "Non-Crossing Dual Neural Network: Joint Value at Risk and Conditional Tail Expectation estimations with Non-Crossing Conditions" [(Working Paper)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4351877).


## Authors
- Xenxo Vidal-Llana (Universitat de Barcelona, Barcelona, Spain)
- Carlos Salort Sanchez (Universitat de Barcelona, Barcelona, Spain)
- Vincenzo Coia (University of British Columbia, Vancouver, BC Canada)
- Montserrat Guillen (Universitat de Barcelona, Barcelona, Spain)


## Abstract
When datasets present long conditional tails on their response variables, algorithms based on Quantile Regression have been widely used to assess extreme quantile behaviors. Value at Risk (VaR) and Conditional Tail Expectation (CTE) allow the evaluation of extreme events to be easily interpretable. The state-of-the-art methodologies to estimate VaR and CTE controlled by covariates are mainly based on linear quantile regression, and usually do not have in consideration non-crossing conditions across VaRs and their associated CTEs. We implement a non-crossing neural network that estimates both statistics simultaneously, for several quantile levels and ensuring a list of non-crossing conditions. We illustrate our method with a household energy consumption dataset from 2015 for quantile levels 0.9, 0.925, 0.95, 0.975 and 0.99, and show its improvements against a Monotone Composite Quantile Regression Neural Network approximation.


## Publication
Under revision in the journal **Insurance: Mathematics and Economics**.


## How to cite
Vidal-Llana, X., Salort Sánchez, C., Coia, V., and Guillén, M. (2022). Non-crossing dual neural network: Joint value at risk and conditional tail expectation estimations with non-crossing conditions. Documents de Treball (IREA), 2022(15):1.


## How to use
This is a self-contained repository. Clone the repo and modify the following files:
- The file *environment.yml* contains information for creating a working conda environment
- Add your data file to *data/*
- Manage the reading of your dataset in *data/prepare_data.py*. Keep in mind that the function read_data must return three values, *df* (pandas data frame containing response, weights and covariates), *response_name* (string with the response name) and *weights_name* (string with weights name)
- Modify *main.py*, by choosing the number of desired quantiles, the number of seeds to run and the epochs per seed
- Run *main.py* when finishing previous steps. It will create the following files in the folder *export/*:
	- *preds_q_XXX.csv* (quantile predictions for quantile level XXX. Row: observation, column: seed)
	- *preds_cte_XXX.csv* (CTE predictions for quantile level XXX. Row: observation, column: seed)
	- *loss_history.csv* (history of losses across epochs. Row: epoch, column: seed)
	- *lf_sum.csv* (sum of losses by using the scoring functions used for training, useful for choosing which seed performs better. Row: seed, column: quantile)


## Contact
For any issue, doubt, comment, alegation, praise, accusation, please contact Xenxo: xenxovidal (at) gmail (dot) com.


## Notes
In this code, the CTE scoring functions are prepared for the right part of the distributions, by following Nolde and Ziegel, 2017 approach. If you want to modify the loss function to adapt the model to the left part in the sense of Acerbi and Szekely, 2015, modify the function *general_loss_function* in *aux_code/loss_functions.py* and change the corresponding subfunctions G1, G2, pG2 and a.

Acerbi, C. and Szekely, B. (2014). Back-testing expected shortfall. Risk, 27(11):76–81.

Nolde, N. and Ziegel, J. F. (2017). Elicitability and backtesting: Perspectives
for banking regulation. The Annals of Applied Statistics, 11(4):1833–1874.


## Acknowledgements
We want to have a special acknowledgement to Fundació Banc Sabadell: “Ajudes a la investigació 2022”, Fundación BBVA: “Ayudas a proyectos de investigación en Big Data”, AGAUR: “PANDÈMIES” grant and the Spanish Ministry of Science grant PID2019-105986GB-C21 for their support to our research.
