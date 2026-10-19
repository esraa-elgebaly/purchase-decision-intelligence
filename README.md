\# Cart Recovery Intelligence: Who Will Buy From Their Cart?



Purchase-intent modeling for an online cosmetics store, built on \*\*5 months of real e-commerce events\*\* (October 2019 – February 2020, about 3.5 million events per month), and turned into a weekly scoring pipeline and a 4-page Streamlit app.



📓 \*\*Notebook:\*\* \[01\_loading\_and\_cleaning.ipynb](notebooks/01\_loading\_and\_cleaning.ipynb) · 🖥️ \*\*App:\*\* \[app/Overview.py](app/Overview.py)



!\[App overview](reports/figures/app\_overview.png)



\---



\## Contents

\- \[The story](#the-story)

\- \[Goal](#goal)

\- \[Results at a glance](#results-at-a-glance)

\- \[Data](#data)

\- \[Pipeline](#pipeline)

\- \[Repository structure](#repository-structure)

\- \[Cleaning](#cleaning)

\- \[Findings](#findings)

\- \[From EDA to the model](#from-eda-to-the-model)

\- \[Modeling setup](#modeling-setup)

\- \[Models and results](#models-and-results)

\- \[What drives the model](#what-drives-the-model)

\- \[From model to product](#from-model-to-product)

\- \[Next steps](#next-steps)

\- \[Limitations](#limitations)

\- \[How to run](#how-to-run)



\---



\## The story



The store gets about \*\*390,000 visitors a month\*\*, and only \*\*6–9% of them buy\*\*. Most visitors look at one product and leave, and that is normal. The real loss is at the cart: in December about \*\*84,000 users added something to their cart, and only about 26,000 bought\*\*. Seven out of ten cart users walked away.



The usual answer is a reminder or a discount for everyone. That spends money on people who will never buy, and gives discounts to people who would have bought anyway.



So the question became: \*\*if the store can contact only 1 in 10 cart users this week, who should it be?\*\*



In February, a month the model never saw, there were 50,392 cart users and 2,179 of them bought in the next 7 days. Picking 5,040 users at random would find about \*\*218 buyers\*\*. The model's top 5,040 found \*\*844\*\*, almost 4 times more for the same budget.



\---



\## Goal



| Target | Task | Definition |

|---|---|---|

| Purchase in the next 7 days | Binary classification | 1 if a cart user buys anything in the 7 days after the cutoff (day 15 of the month) |



Each row is \*\*one cart user at one cutoff date\*\*. Features use only the 14 days \*\*before\*\* the cutoff; the target uses only the 7 days \*\*after\*\* it.



\---



\## Results at a glance

\- \*\*Sales follow the buy rate, not traffic.\*\* November had 8% fewer visitors than October but 26% more sales, because the buy rate jumped from 6.5% to 8.6%.

\- \*\*December lost basket size, not buyers.\*\* It had as many buyers as October (25.6K) but each spent 10% less, so it was the weakest month.

\- \*\*Recency is everything.\*\* Cart users active in the last day bought at 12.3%; after a week of silence, only 2.0%.

\- \*\*Price doesn't decide.\*\* Its correlation with purchase is −0.01, and it is one of the weakest model inputs.

\- \*\*The model:\*\* PR-AUC \*\*0.158\*\* on February, \*\*3.7× random\*\* (0.043), almost the same as on January (0.159).

\- \*\*The business result:\*\* contacting the top 10% reaches \*\*38.7% of all buyers\*\*, and 1 in 6 contacted users buys (1 in 23 at random).



\---



\## Data



| | |

|---|---|

| \*\*Source\*\* | \[REES46 eCommerce events, cosmetics shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop) (Kaggle) |

| \*\*Period\*\* | October 2019 to February 2020 · 5 monthly CSV files · about 3.5 million events per month |

| \*\*Events\*\* | `view`, `cart`, `remove\_from\_cart`, `purchase` |

| \*\*Columns (9)\*\* | `event\_time`, `event\_type`, `product\_id`, `category\_id`, `category\_code`, `brand`, `price`, `user\_id`, `user\_session` |

| \*\*Note\*\* | The raw CSVs are too large for GitHub; download them into `data/raw/` |



\---



\## Pipeline



```mermaid

flowchart LR

&#x20;   A\["Raw events<br/>5 monthly CSVs"] --> B\["Cleaning<br/>duplicates, prices, bot"]

&#x20;   B --> C\["EDA<br/>questions → decisions"]

&#x20;   C --> D\["Features<br/>14 days before the cutoff"]

&#x20;   D --> E\["Time-based split<br/>train / validation / test"]

&#x20;   E --> F\["Models<br/>rule → LR → RF → LightGBM"]

&#x20;   F --> G\["Final test<br/>February, used once"]

&#x20;   G --> H\["Weekly scoring<br/>ranked list + reasons"]

&#x20;   H --> I\["Streamlit app<br/>+ Power BI tables"]

```



\*\*Rules followed throughout:\*\*

1\. The raw data is never changed.

2\. Nothing is deleted before it is understood.

3\. Features use only the past; the target uses only the future.

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

│   └── ...                             # styles, charts, model and scores the app needs

├── models/                             # saved LightGBM model and its settings

├── powerbi/                            # tables and theme for Power BI

├── reports/figures/                    # figures used in this README

├── .streamlit/config.toml              # app theme

├── requirements.txt

└── README.md

```



| Notebook stage | What it covers |

|---|---|

| Loading and cleaning | Duplicates, missing values, bad prices, the bot, before/after check |

| EDA | Funnel, timing, price, brand, months, December day by day, correlations |

| Target and features | Cutoff on day 15, 14 days of history, 7-day target, 14 features |

| Modeling | Rule baseline, Logistic Regression, LightGBM, Random Forest, tuning |

| Evaluation | Time-based CV, learning curves, gains curve, threshold choice |

| Explanation and monitoring | SHAP, final February test, PSI drift on November |



\---



\## Cleaning



| Step | What was checked | Result | Decision |

|---|---|---|---|

| Duplicates | Identical rows | 184K rows, 91% of them `remove\_from\_cart` logged once per item in the same second | Removed: the model counts actions, not items |

| `category\_code` | Missing values | 98% empty; `category\_id` is complete | Dropped |

| `brand` | Missing values | 43% empty; same median price, but "unknown" brands sell above average | Filled with `"unknown"` |

| `price` | Impossible values | Zero or negative prices, mostly on real view events | Row kept, price cleared |

| Outliers | Users far above the rest | One user: 5,343 visits in 14 days, a cart action every \~1.5 s, product IDs in sequence, no purchases | Confirmed bot, removed from training only |

| Before / after | Event mix, prices, users per day | Same shape, errors gone | Clean data saved |



!\[Before and after cleaning](reports/figures/before\_after\_cleaning.png)



\---



\## Findings



\### 1. Sales follow the buy rate, not traffic

Visitors barely move from month to month (368K–410K). What moves is how many of them buy.



| Month | Visitors | Buyers | Buy rate | Spend per buyer | Sales |

|---|---|---|---|---|---|

| Oct 2019 | 399,634 | 25,762 | 6.45% | 47.00 | 1.21M |

| \*\*Nov 2019\*\* | 368,183 | \*\*31,524\*\* | \*\*8.56%\*\* | \*\*48.56\*\* | \*\*1.53M\*\* |

| Dec 2019 | 370,108 | 25,613 | 6.92% | \*\*42.08\*\* | \*\*1.08M\*\* |

| Jan 2020 | \*\*410,018\*\* | 28,220 | 6.88% | 46.84 | 1.32M |

| Feb 2020 | 391,005 | 25,759 | 6.59% | 46.86 | 1.21M |



November (Black Friday) had \*\*fewer\*\* visitors than October but the best sales: more visitors bought, and they spent more. January had the \*\*most\*\* visitors and still sold about 14% less than November. Growth comes from converting visitors, not from bringing more of them, which is exactly what this project targets.



!\[Monthly sales story](reports/figures/monthly\_comparison.png)



\### 2. December lost basket size, not buyers

December had almost the same number of buyers as October (25,613 vs 25,762), but each buyer spent about 10% less, which made it the weakest month. Day by day, activity peaks in the second week (about 157K events a day) and then falls by about 80% by 31 December, as shoppers leave for the holidays.



!\[December day by day](reports/figures/december\_daily.png)



\### 3. The cart is where buyers are lost

In December, about 359K users viewed a product, about 84K added something to the cart, and about 26K bought. Viewing and leaving is normal; \*\*adding to the cart and leaving is lost revenue\*\*, so the model scores cart users only.



Purchases move with cart activity, not with price:



| Behavior | Correlation with purchase |

|---|---|

| Cart removes | 0.55 |

| Cart adds | 0.54 |

| Views | 0.29 |

| Visits | 0.29 |

| Average price | −0.01 |



Buyers are about 3× more active than cart users who don't buy, and they edit their cart more (add, remove, add again).



\### 4. Buyers decide fast

Half of buyers check out within an hour of their first cart, and \*\*84% buy within 7 days\*\*, which sets the 7-day target window. The hour and the weekday matter much less: the buy rate moves between about 3% (late night) and 5.5% (10 AM), and only between 5% and 6% across weekdays.



\### 5. Brand matters, price doesn't

Buy rates range from \*\*0.3% to 50%\*\* across brands, while buyers and non-buyers look at similar prices. So the model gets the brand's buy rate (from the history window only), and price stays as a single, weak feature.



\---



\## From EDA to the model



| What we found | What we do |

|---|---|

| Losses happen at the cart | Score only users with cart activity |

| 84% of buyers buy within 7 days | 7-day target window |

| Buying follows recent activity | Days since last activity, since first activity, since last cart action |

| Buyers are more active and edit their cart | Counts of views, carts, removes, visits; removes per cart item; views per visit |

| Brand matters, price barely does | Brand buy rate as a feature; average price kept as one feature |

| Hour and weekday matter little | Kept as the hour and weekday of the last visit, one-hot |

| Only \~4–5% of cart users buy | PR-AUC instead of accuracy; threshold set by budget |

| November behaves differently | Kept out of training, used to check drift |

| Counts are very skewed | `log1p` and 99% caps learned from training months |

| One bot in the data | Removed from training only |



\---



\## Modeling setup



\### Time-based split

The split is by time, never random, so the model never learns from the future. January chooses the settings; February is used once, at the end.



| Part | Months | Used for |

|---|---|---|

| Train | October, December | Learning |

| Validation | January | Comparing models, tuning, choosing the threshold |

| Test | February · 50,392 cart users · 2,179 buyers (4.3%) | \*\*Used once\*\*, after every decision was locked |

| Monitoring | November | Checking behavior change (Black Friday) |



The final model was retrained on October, December and January before the February test.



\### Metrics



| Metric | Meaning |

|---|---|

| \*\*PR-AUC\*\* | How well buyers are ranked above non-buyers |

| \*\*vs random\*\* | PR-AUC divided by the share of buyers (random = 1×) |

| \*\*Buyers reached at top 10%\*\* | Share of all buyers found by contacting the top 10% |

| \*\*Buy rate at top 10%\*\* | Share of contacted users who buy |

| \*\*Train − validation gap\*\* | How much the model memorizes the training months |

| Accuracy | Not used: "nobody buys" is already 95% accurate |



\---



\## Models and results



\### Validation (January): from simplest to strongest



| Model | PR-AUC | vs random | Train − January gap |

|---|---|---|---|

| Rule: most recent cart first | 0.104 | 2.1× | — |

| Logistic Regression | 0.142 | 2.8× | 0.005 |

| Random Forest | 0.156 | 3.1× | 0.055 |

| \*\*LightGBM (tuned)\*\* | \*\*0.159\*\* | \*\*3.2×\*\* | \*\*0.021\*\* |



\- \*\*A simple rule is already 2× better than random\*\*, so recency carries real signal.

\- \*\*Logistic Regression was stable but underfit:\*\* even on its own training data it stayed below 0.15. A straight line had reached its ceiling.

\- \*\*LightGBM and Random Forest scored almost the same\*\*, so the limit is in the features, not the algorithm. LightGBM won because it memorizes less than half as much, runs faster, and handles missing values on its own.

\- \*\*Tuning:\*\* the first LightGBM memorized the training months (gap 0.079). The selection rule was set before looking at results; requiring at least 500 users per leaf cut the gap about 4× with no loss on January. Final model: 75 trees, 15 leaves.



| Model comparison | LightGBM tuning |

|---|---|

| !\[Model comparison](reports/figures/final\_model\_comparison.png) | !\[Tuning](reports/figures/lightgbm\_tuning.png) |



\### Final test (February, never seen before)

Before running it, I wrote down what I expected: 0.14–0.15.



| Model | PR-AUC | vs random |

|---|---|---|

| Random guess | 0.043 | 1.0× |

| Rule: most recent cart first | 0.099 | 2.3× |

| Logistic Regression | 0.140 | 3.3× |

| \*\*LightGBM\*\* | \*\*0.158\*\* | \*\*3.7×\*\* |



February (0.158) matched January (0.159), so the model held up on new data.



\### What a campaign gets



| Contact the top… | Buyers reached |

|---|---|

| 5% | 25.4% |

| \*\*10%\*\* | \*\*38.7%\*\* |

| 20% | 55.4% |

| 30% | 65.9% |

| 50% | 81.8% |



Contacting half of the cart users reaches 82% of buyers. The budget decides where to stop; the model decides who comes first.



| Gains on February | Choosing the threshold |

|---|---|

| !\[Final test gains](reports/figures/final\_test\_gains.png) | !\[Threshold choice](reports/figures/threshold\_choice.png) |



The threshold is a budget decision ("contact the top 10%"), and it landed almost exactly on the best-F1 point.



\---



\## What drives the model



SHAP confirms the EDA: the model relies on \*\*behavior\*\*, not on price or time of day.



| Feature | Average SHAP effect |

|---|---|

| Days since last activity | \*\*0.54\*\* |

| Cart adds | 0.31 |

| Views | 0.24 |

| Cart removes | 0.20 |

| Days since last cart action | 0.16 |

| … | |

| Average price | 0.03 |

| Hour of last visit | 0.01 |

| Weekday of last visit | 0.005 |



The strongest signal is recency, and February shows it plainly:



| Last activity | Cart users | Bought in the next 7 days |

|---|---|---|

| Less than 1 day ago | 5,597 | \*\*12.3%\*\* |

| 1–3 days ago | 9,943 | 6.0% |

| 3–7 days ago | 14,372 | 3.5% |

| More than 7 days ago | 20,480 | 2.0% |



\*\*After about a week without activity, the chance of buying drops sharply.\*\* A reminder is worth the most in the first days after the cart.



| Main drivers | The one-week rule |

|---|---|

| !\[SHAP drivers](reports/figures/shap\_drivers.png) | !\[SHAP week rule](reports/figures/shap\_week\_rule.png) |



\---



\## From model to product



A model only creates value when a team can act on it.



\*\*Weekly scoring\*\* (`src/score.py`): cleans a new month, builds features, scores every cart user, and writes a ranked list with each user's top reasons in plain words, ready for a CRM. Running it on February reproduces the notebook exactly (PR-AUC 0.158, 38.7% of buyers reached). One row of the output:



| Rank | Score | Segment | Action | Why |

|---|---|---|---|---|

| 1 | 0.43 | High intent | Send reminder, no discount | active 0.1 days ago; last cart action 0.1 days ago |



\*\*Action plan by segment (February):\*\*



| Segment | Cart users | Buyers | Buy rate | Share of all buyers | Action |

|---|---|---|---|---|---|

| High intent (top 10%) | 5,039 | 844 | \*\*16.7%\*\* | \*\*38.7%\*\* | Send a reminder, \*\*no discount\*\* (likely to buy anyway) |

| Medium intent (next 20%) | 10,078 | 592 | 5.9% | 27.2% | Test a small discount |

| Low intent (other 70%) | 35,275 | 743 | 2.1% | 34.1% | Don't contact this week |



\*\*Streamlit app\*\* (`app/`): one tool, four teams.



| Page | For | Answers |

|---|---|---|

| Overview | Management | What's the situation, what should we do, and why trust it? |

| Campaign planner | Marketing | With this budget, how many buyers do we reach, and who do we contact? |

| Customer lookup | Customer service | Will this customer buy, why, and what should we do? |

| Model health | Data team | Does the model still fit new data? (PSI drift check) |



!\[Campaign planner](reports/figures/app\_campaign\_planner.png)



\---



\## Next steps

1\. \*\*Uplift modeling:\*\* find who buys \*because of\* an offer, using A/B test data.

2\. \*\*Features:\*\* price compared with the category average, history from earlier months, product-level buy rates.

3\. \*\*Survival analysis:\*\* predict \*when\* a user will buy, not only whether.

4\. \*\*Bot filter:\*\* a rule-based filter instead of removing one confirmed bot by ID.

5\. \*\*Serving:\*\* FastAPI and Docker for real-time scoring.



\## Limitations

\- \*\*One store:\*\* a European cosmetics shop, 2019–2020; other markets may behave differently.

\- \*\*No quantity column:\*\* revenue figures are approximate.

\- \*\*One holiday season:\*\* the model can't learn holiday effects.

\- \*\*Likely, not caused:\*\* the model finds who is likely to buy, not who buys because of a reminder. Measuring that needs an A/B test with a holdout group.



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



The full analysis is in \[notebooks/01\_loading\_and\_cleaning.ipynb](notebooks/01\_loading\_and\_cleaning.ipynb).



\## Author

\*\*Esraa Elgebaly\*\* · Machine Learning Engineer

