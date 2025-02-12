# Replication materials

Replication materials for "Behind the Mask: Random and Selective Masking in Transformer Models Applied to Specialized Social Science Texts" (2025), by Joan C. Timoneda and Sebastián Vallejo Vera.

> __Abstract:__
> Transformer models such as BERT and RoBERTa are increasingly popular in the social sciences to generate data through supervised text classification. These models can be further trained through Masked Language Modeling (MLM) to increase performance in specialized applications. MLM uses a default masking rate of 15 percent, and few works have investigated how different masking rates may affect performance. Importantly, there are no systematic tests on whether selectively masking certain words improves classifier accuracy. In this article, we further train a set of models to classify fake news around the coronavirus pandemic using 15, 25, 40, 60 and 80 percent random and selective masking. We find that a masking rate of 40 percent, both random and selective, improves within-category performance but has little impact on overall performance. This finding has important implications for scholars looking to build BERT and RoBERTa classifiers, especially those where one specific category is more relevant to their research.

A link to the article is available [here](FILL).

This README file provides an overview of the replications materials for the article. The [Data](https://github.com/svallejovera/masking_plosone#data) section describes the main dataset required to reproduce the tables and figures in the paper. The [Code](https://github.com/svallejovera/masking_plosone#code) section provides the code necessary to run different masking techniques, and to replicate Figure 1 in the main text. 

## Data

- `/data/pretraining_tweets_en_full.txt`: text data used to further pre-train a RoBERTa model, as in `/code/pt_sel_mask.ipynb`.
- `/data/fake_news_covid.xlsx`: training set to fine-tune a machine-learning model, as in `/code/ft_from_pt.ipynb`.

## Code

- `/code/pt_sel_mask.ipynb` to run selective masking during further pre-training of a RoBERTa model. In cell 12 we provide the two lines of code required to change the masking rate, and selectively mask at that rate custom tokens.
- `/code/pt_custom.ipynb` to run further pre-training of a RoBERTa model at a masking rate determined by the user. In cell 12 we provide the parameter required to change the masking rate.
- `/code/ft_from_pt.ipynb` to fine-tune the further pre-trained model from `/code/pt_sel_mask.ipynb` or `/code/pt_custom.ipynb`.
- `/code/helper_functions.py` required to run `/code/ft_from_pt.ipynb`.
- `/code/read_results.py` to read the results from `/code/ft_from_pt.ipynb` and order them in table format. 
