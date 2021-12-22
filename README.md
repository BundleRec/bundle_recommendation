# Bundle Recommendation
This project aims to provide new data sources for product bundling in real e-commerce platforms with the domains of Electronic, Clothing and Food. We construct three high-quality bundle datasets with rich meta information, particularly bundle intents, through a carefully designed crowd-sourcing task.


### Worker Basic Information
Figure 1 shows the distribution of workers' age, education, country, occupation, gender and shopping frequency for the two batches. In particular, `Others' in the country distribution includes Argentina, Australia, Anguilla, Netherlands, Albania, Georgia, Tunisia, Belgium, Armenia, Guinea, Austria, Switzerland, Iceland, Lithuania, Egypt, Venezuela, Bangladesh, American Samoa, Vanuatu, Colombia, United Arab Emirates, Ashmore and Cartier Island, Estados Unidos, Wales, Turkey, Angola, Scotland, Philippines, Iran and Bahamas.


![basic_information](img/worker_basic_information.png)
<p align="center">Figure 1: Worker basic information in the first and second batches.</p>

### Parameter Tuning and Settings for Bundle Detection
A grid search in {0.0001, 0.001, 0.01} is applied to find out the optimal settings for *support* and *confidence*, and both are set as 0.001 across the three domains.


### Parameter Tuning and Settings for Bundle Completion
The dimension *d* of item and bundle representations for all methods is 20. Grid search is adopted to find out the best settings for other key parameters. In particular, learning rate ![](https://latex.codecogs.com/svg.image?\eta)  and regularization coefficient ![](https://latex.codecogs.com/svg.image?\lambda)  are searched in {0.0001, 0.001, 0.01}; the number of neighbors *K* in ItemKNN is searched in {10, 20, 30, 50}; the weight of KL divergence ![](https://latex.codecogs.com/svg.image?\alpha) in VAE is searched in {0.001, 0.01, 0.1}; and the batch size is searched in {64, 128, 256}. The optimal parameter settings are shown in Table 1. 

#### Table 1: Parameter settings for bundle completion (*d=20*).
|  | Electronic | Clothing | Food |
| :------: | :------: | :------: | :------: |
| ItemKNN | ![equation](https://latex.codecogs.com/svg.image?K=10)| ![equation](https://latex.codecogs.com/svg.image?K=10) | ![equation](https://latex.codecogs.com/svg.image?K=10) |
| BPRMF | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.01)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.01)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.01)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) |
| mean-VAE | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.001)<br>![equation](https://latex.codecogs.com/svg.image?\alpha=0.01)<br>![equation](https://latex.codecogs.com/svg.image?hid\\_layers=[100,50])<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=64) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\alpha=0.001)<br>![equation](https://latex.codecogs.com/svg.image?hid\\_layers=[100,50])<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.001)<br>![equation](https://latex.codecogs.com/svg.image?\alpha=0.001)<br>![equation](https://latex.codecogs.com/svg.image?hid\\_layers=[100,50])<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=64) |
| concat-VAE | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.01)<br>![equation](https://latex.codecogs.com/svg.image?\alpha=0.001)<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.001)<br>![equation](https://latex.codecogs.com/svg.image?\alpha=0.1)<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=64) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\alpha=0.001)<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=64) |


### Parameter Tuning and Settings for Bundle Ranking
The dimension *d* of representations is set as 20. We apply a same grid search for ![](https://latex.codecogs.com/svg.image?\eta), ![](https://latex.codecogs.com/svg.image?\lambda), ![](https://latex.codecogs.com/svg.image?K) and batch size as in bundle completion. Besides, the predictive layer *D* for AttList is searched from {20, 50, 100}; the node and message dropout rate for GCN and BGCN is searched in {0, 0.1, 0.3, 0.5}. As the training complexity for GCN and BGCN is quite high, we set the batch size as 2048 as suggested by the original paper. The optimal parameter settings are presented in Table 2. Note that the parameter settings for BGCN is the version without pre-training (i.e. ![](https://latex.codecogs.com/svg.image?BGCN_%7Bw/o%5C%20pre%7D)). 



#### Table 2: Parameter settings for bundle ranking (*d=20*).
|  | Electronic | Clothing | Food |
| :------: | :------: | :------: | :------: |
| ItemKNN | ![equation](https://latex.codecogs.com/svg.image?K=10)| ![equation](https://latex.codecogs.com/svg.image?K=10) | ![equation](https://latex.codecogs.com/svg.image?K=10) |
| BPRMF | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.01)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) |
| DAM | ![equation](https://latex.codecogs.com/svg.image?\eta=0.01)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=1)<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.01)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=1)<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.01)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=1)<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5) |
| AttList | ![equation](https://latex.codecogs.com/svg.image?\eta=0.001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=2)<br>![equation](https://latex.codecogs.com/svg.image?\\&hash;bundles/user=5)<br>![equation](https://latex.codecogs.com/svg.image?\\&hash;items/bundle=10)<br>![equation](https://latex.codecogs.com/svg.image?D=100)<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=64) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=2)<br>![equation](https://latex.codecogs.com/svg.image?\\&hash;bundles/user=5)<br>![equation](https://latex.codecogs.com/svg.image?\\&hash;items/bundle=10)<br>![equation](https://latex.codecogs.com/svg.image?D=50)<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=2)<br>![equation](https://latex.codecogs.com/svg.image?\\&hash;bundles/user=5)<br>![equation](https://latex.codecogs.com/svg.image?\\&hash;items/bundle=10)<br>![equation](https://latex.codecogs.com/svg.image?D=50)<br>![equation](https://latex.codecogs.com/svg.image?dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=256) |
| GCN | ![equation](https://latex.codecogs.com/svg.image?\eta=0.01)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.01)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=1)<br>![equation](https://latex.codecogs.com/svg.image?msg\\_dropout=0.3)<br>![equation](https://latex.codecogs.com/svg.image?node\\_dropout=0)<br>![equation](https://latex.codecogs.com/svg.image?prop\\_layers=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=2048) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=1)<br>![equation](https://latex.codecogs.com/svg.image?msg\\_dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?node\\_dropout=0)<br>![equation](https://latex.codecogs.com/svg.image?prop\\_layers=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=2048) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.01)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.0001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=1)<br>![equation](https://latex.codecogs.com/svg.image?msg\\_dropout=0.5)<br>![equation](https://latex.codecogs.com/svg.image?node\\_dropout=0)<br>![equation](https://latex.codecogs.com/svg.image?prop\\_layers=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=2048) |
| BGCN | ![equation](https://latex.codecogs.com/svg.image?\eta=0.001)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=1)<br>![equation](https://latex.codecogs.com/svg.image?msg\\_dropout=0.1)<br>![equation](https://latex.codecogs.com/svg.image?node\\_dropout=0)<br>![equation](https://latex.codecogs.com/svg.image?prop\\_layers=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=2048) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.01)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.01)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=1)<br>![equation](https://latex.codecogs.com/svg.image?msg\\_dropout=0)<br>![equation](https://latex.codecogs.com/svg.image?node\\_dropout=0)<br>![equation](https://latex.codecogs.com/svg.image?prop\\_layers=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=2048) | ![equation](https://latex.codecogs.com/svg.image?\eta=0.01)<br>![equation](https://latex.codecogs.com/svg.image?\lambda=0.001)<br>![equation](https://latex.codecogs.com/svg.image?neg\\_sample=1)<br>![equation](https://latex.codecogs.com/svg.image?msg\\_dropout=0.1)<br>![equation](https://latex.codecogs.com/svg.image?node\\_dropout=0.1)<br>![equation](https://latex.codecogs.com/svg.image?prop\\_layers=2)<br>![equation](https://latex.codecogs.com/svg.image?batch\\_size=2048) |

### Statistics of Datasets
This data set consists of three domains' data collected from Mturk website. Table 3 shows the statistics of data.

#### Table 3: Statistics of datasets.

|      | Electronic | Clothing | Food |
|:------|:------------:|:----------:|:------:|
| #Users |    888   |   965    | 879  |
| #Items |    3499  |   4487   | 3767 |
| #Sessions | 1145  |   1181   | 1161 |
| #Bundles | 1750 | 1910 | 1784 |
| #Intents | 1537 | 1590 | 1323 |
| Avg. Bundle Size | 3.52 | 3.31 | 3.58 |
| #U-I Interactions | 6165 | 6326 | 6395 |
| #U-B Interactions | 1753 | 1912 | 1785 |
| Density of U-I Interactions | 0.20% | 0.15% | 0.19% |
| Density of U-B Interactions | 0.11% | 0.10% | 0.11% |

### Detailed Descriptions if Data Files
Table 4 shows brief descriptions of the data files.

#### Table 4: the brief descriptions of the data files
| File Name | Descriptions |
|-----------|--------------|
| session_item.csv | This file contains the affiliation of session and its items. Each session has at least 2 items.<br> This is a tab separated list: `session ID \| item ID \|` <br>The session IDs are the ones used in the session_bundle.csv and user_session.csv data sets.  |
| session_bundle.csv| This file contains the affiliation of session and detected bundles. Each session has at least 1 bundle.<br> This is a tab separated list: `session ID \| bundle ID \|` <br>The bundle IDs are the ones used in the bundle_item.csv ,user_bundle.csv and bundle_intent.csv data sets. |
| bundle_item.csv | This file contains the affiliation of bundle and its items. Each bundle has at least 2 items.<br> This is a tab separated list: `bundle ID \| item ID \|` <br>The item IDs are the ones used in the user_item.csv and item_categories.csv data sets. |
| item_categories.csv| This file contains item and its categories.<br> This is a tab separated list: `item ID \| categories \|` <br> The format of data in `categories` column is a list of string. |
| user_session.csv| This file contains the interaction of user and session.<br> This is a tab separated list: `user ID \| session ID \| timestamp \|` <br>The user IDs are the ones used in the user_bundle.csv and user_item.csv data sets. |
| user_bundle.csv | This file contains the interaction of user and bundle.<br> This is a tab separated list: `user ID \| bundle ID \| timestamp \|`  |
| user_item.csv | This file contains the interaction of user and item.<br> This is a tab separated list: `user ID \| item ID \| timestamp \|`  |
| user_item_pretrain.csv| This file contains the interaction of user and item aiming to obtain pre-trained item representations via BPRMF for model initialization .<br> This is a tab separated list: `user ID \| item ID \| timestamp \|`  |
| bundle_intent.csv | This file contains bundle and its annotated intent.<br> This is a tab separated list: `bundle ID \| intent \|`  |




### Acknowledgements

Our datasets are constructed on the basis of Amazon datasets (http://jmcauley.ucsd.edu/data/amazon/links.html).
