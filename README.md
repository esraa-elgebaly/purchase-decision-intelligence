\# Cart Recovery Intelligence: Who Will Buy From Their Cart?



A purchase-intent model for an online cosmetics store, built on \*\*5 months of real events\*\* (October 2019 – February 2020) from the \*\*REES46\*\* dataset, and turned into a weekly scoring pipeline and a Streamlit app.



\*\*Goal:\*\* among users who added something to their cart, find the ones who will buy in the next 7 days, so the store contacts fewer people and still reaches most buyers.



| Target | Task | Definition |

|---|---|---|

| Purchase in the next 7 days | Binary classification | 1 if a cart user buys anything in the 7 days after the cutoff date |



!\[App overview](reports/figures/app\_overview.png)



\---



\## Results at a glance



Tested on \*\*February\*\*, a month the model never saw:



| Measure | Result |

|---|---|

| PR-AUC | \*\*0.158\*\*, 3.7× random (0.043) |

| Buyers reached by contacting the top 10% | \*\*38.7%\*\* of all buyers |

| Buy rate among contacted users | \*\*16.7%\*\*, 3.9× random (4.3%) |

| Same budget (5,040 users) | about \*\*840 buyers\*\* reached, vs about \*\*220\*\* at random |



\---



\## Pipeline



```mermaid

flowchart LR

&#x20;   A\["Raw events<br/>5 monthly CSVs"] --> B\["Cleaning<br/>duplicates, prices, bot"]

&#x20;   B --> C\["Features<br/>14 days before the cutoff"]

&#x20;   C --> D\["Time-based split<br/>train / validation / test"]

&#x20;   D --> E\["Baselines and models<br/>rule, LR, RF, LightGBM"]

&#x20;   E --> F\["Final test<br/>February, used once"]

&#x20;   F --> G\["Weekly scoring<br/>ranked list + reasons"]

&#x20;   G --> H\["Streamlit app<br/>and Power BI tables"]

```



\*\*Rules followed throughout:\*\*

1\. The raw data is never changed.

2\. Nothing is deleted before it is understood.

3\. Features use only events \*\*before\*\* the cutoff; the target uses only events \*\*after\*\* it.

4\. The split is by time, never random.

5\. Anything learned from the data (caps, scaling) uses training months only.

6\. The test month is used once, after every decision is locked.



\---



\## Repository structure



```

├── notebooks/

│   └── 01\_loading\_and\_cleaning.ipynb   # full analysis, cleaning to final test

├── src/

│   ├── features.py                     # cleaning and feature building

│   ├── score.py                        # weekly scoring: ranked list + reasons

│   ├── make\_dashboard.py               # numbers for the app

│   └── export\_powerbi.py               # clean tables for Power BI

├── app/

│   ├── Overview.py                     # app main page

│   ├── pages/                          # Campaign planner, Customer lookup, Model health

│   ├── common.py, charts.py, style.css

│   └── model.txt, scores.csv, ...      # everything the app needs to run

├── models/                             # saved LightGBM model and its settings

├── powerbi/                            # tables and theme for Power BI

├── reports/figures/                    # figures used in this README

├── .streamlit/config.toml              # app theme

├── requirements.txt

└── README.md

```



\*\*Notebook outline\*\*



| Stage | What it covers |

|---|---|

| Loading and cleaning | Duplicates, missing values, bad prices, before/after check |

| EDA | 11 questions on buyers, timing, price, brand, and months |

| Target and features | Cutoff on day 15, 14 days of history, 7-day target |

| Modeling | Rule baseline, Logistic Regression, LightGBM, Random Forest, tuning |

| Evaluation | Time-based CV, learning curves, gains curve, threshold choice |

| Explanation and monitoring | SHAP, final February test, PSI drift on November |



\---



\## Data



| | |

|---|---|

| \*\*Source\*\* | \[REES46 eCommerce events, cosmetics shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop) (Kaggle) |

| \*\*Period\*\* | October 2019 to February 2020 · about 3.5 million events per month |

| \*\*Events\*\* | `view`, `cart`, `remove\_from\_cart`, `purchase` |

| \*\*Columns\*\* | `event\_time`, `event\_type`, `product\_id`, `category\_id`, `category\_code`, `brand`, `price`, `user\_id`, `user\_session` |

| \*\*Not included\*\* | Raw CSVs are too large for GitHub; download them into `data/raw/` |



\---



\## Cleaning



| Step | What was checked | Decision |

|---|---|---|

| Duplicates | 184K duplicate rows, 91% of them `remove\_from\_cart` logged once per item in the same second | Removed: the model counts actions, not items |

| `category\_code` | 98% empty; `category\_id` is complete | Dropped the column |

| `brand` | 43% empty; "unknown" brands sell above average | Filled with `"unknown"` |

| Prices | Zero or negative prices, mostly on real view events | Kept the row, cleared the price |

| Unusual users | One user with 5,343 visits in 14 days: a cart action every \~1.5 seconds, product IDs in sequence, no purchases | Confirmed bot, removed from training only |

| Before / after | Event mix, prices and users per day | Same shape, errors removed |



!\[Before and after cleaning](reports/figures/before\_after\_cleaning.png)



\---



\## EDA: questions and answers



| # | Question | Answer |

|---|---|---|

| \*\*Q1\*\* | Where is the store losing buyers? | \*\*At the cart.\*\* 70% of cart users never buy |

| \*\*Q2\*\* | How are buyers different from cart users who don't buy? | About \*\*3× more active\*\*, and they edit their cart more |

| \*\*Q3\*\* | How fast do buyers buy? | Half check out within an hour of their first cart; \*\*84% within 7 days\*\* |

| \*\*Q4\*\* | Does price separate buyers? | \*\*No.\*\* Buyers and non-buyers look at similar prices |

| \*\*Q5\*\* | Does brand matter? | \*\*Yes.\*\* Buy rates range from 0.3% to 50% across brands |

| \*\*Q6\*\* | Do sales follow traffic or buy rate? | \*\*Buy rate.\*\* Visitors stay stable every month |

| \*\*Q7\*\* | Why was November the best month? | Black Friday: \*\*more visitors bought\*\*, not more visitors came |



| Monthly sales and buy rate | December, day by day |

|---|---|

| !\[Monthly comparison](reports/figures/monthly\_comparison.png) | !\[December daily](reports/figures/december\_daily.png) |



\---



\## From EDA to the model



| What we found | What we do |

|---|---|

| Most losses happen at the cart | Score only users with cart activity |

| 84% of buyers buy within 7 days | 7-day target window |

| Buying happens soon after activity | Recency features: days since last activity, first activity, last cart action |

| Buyers are more active and edit their cart | Counts of views, carts, removes, visits; removes per cart item |

| Brand matters, price barely does | Average brand buy rate as a feature; price kept as one feature only |

| Only about 5% of cart users buy | PR-AUC instead of accuracy; threshold set by budget |

| November behaves differently | Kept out of training, used to check drift |

| Counts are very skewed | `log1p` and 99% caps learned from training months |

| One bot in the data | Removed from training only, scored like anyone else |



\*\*14 features:\*\* views, cart adds, cart removes, past purchases, visits, days since last activity, days since first activity, days since last cart action, views per visit, removes per cart item, average price, average brand buy rate, hour and weekday of the last visit.



\---



\## Modeling setup



\### Time-based split



| Part | Months | Used for |

|---|---|---|

| Train | October, December | Learning |

| Validation | January | Comparing models, tuning, choosing the threshold |

| Test | February | \*\*Used once\*\*, after every decision was locked |

| Monitoring | November | Checking behavior change (Black Friday) |



The final model was retrained on October, December and January before the February test.



\### Metrics



| Metric | Meaning |

|---|---|

| \*\*PR-AUC\*\* | How well the model ranks buyers above non-buyers; compared with the random rate |

| \*\*vs random\*\* | PR-AUC divided by the share of buyers |

| \*\*Buyers reached at top 10%\*\* | Share of all buyers found by contacting the top 10% |

| \*\*Buy rate at top 10%\*\* | Share of contacted users who buy |

| \*\*Train − validation gap\*\* | How much the model memorizes the training months |

| Accuracy | Not used: 95% of users don't buy, so "nobody buys" already scores 95% |



\---



\## Models



\### Validation (January)



| Model | PR-AUC | vs random | Train − January gap |

|---|---|---|---|

| Rule: most recent cart first | 0.104 | 2.1× | — |

| Logistic Regression | 0.142 | 2.8× | 0.005 |

| Random Forest | 0.156 | 3.1× | 0.055 |

| \*\*LightGBM (tuned)\*\* | \*\*0.159\*\* | \*\*3.2×\*\* | \*\*0.021\*\* |



\- \*\*Logistic Regression was stable but underfit:\*\* even on its own training data it stayed below 0.15.

\- \*\*LightGBM and Random Forest scored almost the same\*\*, so the limit is in the features, not the algorithm. LightGBM was chosen because it memorizes less than half as much, runs faster, and handles missing values on its own.

\- \*\*Tuning:\*\* the first LightGBM had a gap of 0.079. The selection rule was set before looking at results; at least 500 users per leaf cut the gap about 4× with no loss on January.



| Model comparison | LightGBM tuning |

|---|---|

| !\[Model comparison](reports/figures/final\_model\_comparison.png) | !\[Tuning](reports/figures/lightgbm\_tuning.png) |



\### Final test (February)



| Model | PR-AUC | vs random |

|---|---|---|

| Random guess | 0.043 | 1.0× |

| Rule: most recent cart first | 0.099 | 2.3× |

| Logistic Regression | 0.140 | 3.3× |

| \*\*LightGBM\*\* | \*\*0.158\*\* | \*\*3.7×\*\* |



The expected score was written down before the test (0.14–0.15). February matched January almost exactly, so the model held up on new data.



| Campaign decision (top 10%) | Value |

|---|---|

| Users contacted | 5,040 |

| Buyers reached | 38.7% of all buyers |

| Buy rate among contacted | 16.7% (3.9× random) |



| Gains on February | Choosing the threshold |

|---|---|

| !\[Final test gains](reports/figures/final\_test\_gains.png) | !\[Threshold choice](reports/figures/threshold\_choice.png) |



The threshold is a budget decision ("contact the top 10%"), and it landed almost exactly on the best-F1 point.



\---



\## What drives the model



SHAP shows the model relies on behavior, matching the EDA. \*\*Recent activity is the strongest signal:\*\* after about a week without activity, the chance of buying drops sharply. Price, hour and weekday barely matter once activity is known.



| Main drivers | The one-week rule |

|---|---|

| !\[SHAP drivers](reports/figures/shap\_drivers.png) | !\[SHAP week rule](reports/figures/shap\_week\_rule.png) |



\---



\## From model to product



\*\*Weekly scoring\*\* (`src/score.py`): cleans a new month, builds features, scores every cart user, and writes a ranked list with each user's top reasons in plain words, ready for a CRM. Running it on February reproduces the notebook exactly (PR-AUC 0.158, 39% of buyers reached).



\*\*Streamlit app\*\* (`app/`):



| Page | For | Answers |

|---|---|---|

| Overview | Management | What's the situation, what should we do, and why trust it? |

| Campaign planner | Marketing | With this budget, how many buyers do we reach, and who do we contact? |

| Customer lookup | Customer service | Will this customer buy, why, and what should we do? |

| Model health | Data team | Does the model still fit new data? (PSI drift check) |



!\[Campaign planner](reports/figures/app\_campaign\_planner.png)



\*\*Action plan by segment:\*\*



| Segment | Share of users | Action |

|---|---|---|

| High intent | Top 10% | Send a reminder, \*\*no discount\*\* (likely to buy anyway) |

| Medium intent | Next 20% | Test a small discount |

| Low intent | Remaining 70% | Don't contact this week |



\---



\## Next steps

1\. \*\*Uplift modeling:\*\* predict who buys \*because of\* an offer, using A/B test data.

2\. \*\*Features:\*\* price compared with the category average, history from earlier months, product-level buy rates.

3\. \*\*Survival analysis\*\* to predict \*when\* a user will buy.

4\. \*\*Bot filter:\*\* a rule-based filter instead of removing one confirmed bot by ID.

5\. \*\*Serving:\*\* FastAPI and Docker for real-time scoring.



\## Limitations

\- \*\*One store:\*\* a European cosmetics shop, 2019–2020; other markets may behave differently.

\- \*\*No quantity column:\*\* revenue figures are approximate.

\- \*\*One holiday season:\*\* the model can't learn holiday effects.

\- \*\*Likely, not caused:\*\* the model finds who is likely to buy, not who buys because of a reminder. That needs an A/B test with a holdout group.



\---



\## How to run



```bash

pip install -r requirements.txt



\# open the app (works without the raw data)

python -m streamlit run app/Overview.py



\# score a new week (needs the Kaggle CSVs in data/raw/)

python src/score.py --data data/raw/2020-Feb.csv --cutoff 2020-02-15 --budget 10

python src/make\_dashboard.py

python src/export\_powerbi.py

```



The full analysis is in `notebooks/01\_loading\_and\_cleaning.ipynb`.



\## Author

\*\*Esraa Elgebaly\*\* · Machine Learning Engineer

