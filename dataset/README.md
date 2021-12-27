## Summary
This data set consists of three domains' data collected from Mturk website. Table 1 shows the statistics of data.

<p align="center">Table 1: Statistics of datasets.</p>

|      | Electronic | Clothing | Food |
|:------|:------------:|:----------:|:------:|
| #Users |    888   |   965    | 879  |
| #Items |    3499  |   4487   | 3767 |
| #Sessions | 1145  |   1181   | 1161 |
| #Bundles | 1750 | 1910 | 1784 |
| #Intents | 1422 | 1466 | 1156 |
| Avg. Bundle Size | 3.52 | 3.31 | 3.58 |
| #U-I Interactions | 6165 | 6326 | 6395 |
| #U-B Interactions | 1753 | 1912 | 1785 |
| Density of U-I Interactions | 0.20% | 0.15% | 0.19% |
| Density of U-B Interactions | 0.11% | 0.10% | 0.11% |

## Detailed Descriptions of Data Files
Table 2 shows brief descriptions of the data files.

<p align="center">Table 2: the brief descriptions of the data files.</p>

| File Name | Descriptions |
|-----------|--------------|
| user_item_pretrain.csv| This file contains the interaction of user and item aiming to obtain pre-trained item representations via BPRMF for model initialization.<br> This is a tab separated list: `user ID \| item ID \| timestamp \|`  <br>The user IDs are the ones used in the `user_bundle.csv` and `user_item.csv` data sets. The item IDs are the ones used in the `user_item.csv`, `session_item.csv` and `item_categories.csv` data sets.|
| user_item.csv | This file contains the interaction of user and item.<br> This is a tab separated list: `user ID \| item ID \| timestamp \|`  |
| session_item.csv | This file contains the affiliation of session and its items. Each session has at least 2 items.<br> This is a tab separated list: `session ID \| item ID \|` <br>The session IDs are the ones used in the `session_bundle.csv` and `user_session.csv` data sets.  |
| user_session.csv| This file contains the interaction of user and session.<br> This is a tab separated list: `user ID \| session ID \| timestamp \|`  |
| session_bundle.csv| This file contains the affiliation of session and detected bundles. Each session has at least 1 bundle.<br> This is a tab separated list: `session ID \| bundle ID \|` <br>The bundle IDs are the ones used in the `bundle_item.csv` ,`user_bundle.csv` and `bundle_intent.csv` data sets. |
| bundle_intent.csv | This file contains bundle and its annotated intent.<br> This is a tab separated list: `bundle ID \| intent \|`  |
| bundle_item.csv | This file contains the affiliation of bundle and its items. Each bundle has at least 2 items.<br> This is a tab separated list: `bundle ID \| item ID \|` |
| user_bundle.csv | This file contains the interaction of user and bundle.<br> This is a tab separated list: `user ID \| bundle ID \| timestamp \|`  |
| item_categories.csv| This file contains item and its categories.<br> This is a tab separated list: `item ID \| categories \|` <br> The format of data in `categories` column is a list of string. |
