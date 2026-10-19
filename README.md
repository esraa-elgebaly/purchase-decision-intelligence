\# 🛒 Cart Recovery Intelligence



\### Predicting which cart users will buy in the next 7 days, so a store contacts fewer people and still reaches most buyers.



!\[Python](https://img.shields.io/badge/Python-3776AB?logo=python\&logoColor=white)

!\[LightGBM](https://img.shields.io/badge/LightGBM-2E7D32)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn\&logoColor=white)

!\[SHAP](https://img.shields.io/badge/SHAP-explainability-8E44AD)

!\[Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit\&logoColor=white)

!\[Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly\&logoColor=white)



| \*\*3.7×\*\* | \*\*38.7%\*\* | \*\*16.7%\*\* | \*\*4 teams\*\* |

|:---:|:---:|:---:|:---:|

| better than random (PR-AUC) on a month the model never saw | of all buyers reached by contacting only the top 10% | of contacted users buy (4.3% at random) | served by one Streamlit app |



!\[App overview](reports/figures/app\_overview.png)



\---



\## Contents



\- \[The business problem](#the-business-problem)

\- \[What makes this project different](#what-makes-this-project-different)

\- \[Data and cleaning](#data-and-cleaning)

\- \[What I found in the data](#what-i-found-in-the-data)

\- \[How the model works](#how-the-model-works)

\- \[Results](#results)

\- \[What drives the model](#what-drives-the-model)

\- \[From model to product](#from-model-to-product)

\- \[How to run it](#how-to-run-it)

\- \[Limitations and next steps](#limitations-and-next-steps)



\---



\## The business problem



\*\*70% of people who add something to their cart never buy.\*\* Most stores answer this by sending the same reminder or discount to everyone. That costs money, annoys people who were never going to buy, and gives discounts to people who would have bought anyway.



This project ranks every cart user by their chance of buying in the next 7 days, so a store can choose who to contact on a fixed budget.



\*\*Same budget, about 4× the buyers.\*\* In February, contacting 5,040 users (the top 10%):



| Who gets contacted | Buyers reached |

|---|---|

| 5,040 users picked at random | \~220 |

| \*\*5,040 users picked by the model\*\* | \*\*\~840\*\* |



| Choice | Decision |

|---|---|

| Unit | One user at one cutoff date (day 15 of each month) |

| Features | Only events in the 14 days \*\*before\*\* the cutoff, so nothing leaks from the future |

| Target | 1 if the user buys anything in the 7 days after the cutoff |

| Who is scored | Users with cart activity, because that is where the store loses money |

| Main metric | PR-AUC compared with the random rate (only \~5% of cart users buy, so accuracy would be misleading) |



The 7-day window comes from the data: 84% of buyers buy within 7 days of their first view.



\---



\## What makes this project different



\- \*\*No leakage by design:\*\* a time-based split (train on past months, test on a later month), and features built only from events before the cutoff.

\- \*\*The test month was used once.\*\* I wrote down my expected score before running it (0.14–0.15); the model scored 0.158.

\- \*\*I investigated before deleting.\*\* Every cleaning step comes from a question about the data, including a user who turned out to be a bot.

\- \*\*The threshold is a business decision:\*\* "contact the top 10%" fits a real campaign budget better than a default 0.5 cutoff.

\- \*\*Every score comes with reasons\*\* in plain words (from SHAP), so the marketing team knows \*why\* a user was picked.

\- \*\*It runs like a real system:\*\* a weekly scoring script that reproduces the notebook exactly, an app for four teams, and a drift check that says when to retrain.



\---



\## Data and cleaning



\[REES46 eCommerce events from a cosmetics shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop) (Kaggle): five months of real events (views, cart adds, cart removes, purchases), October 2019 to February 2020, about 3.5 million events per month.



| Problem | What I found | Decision |

|---|---|---|

| 184K duplicate rows | 91% were `remove\_from\_cart`, logged once per item in the same second | Removed: I count actions, not items |

| `category\_code` 98% empty | The store only fills it for clear categories; `category\_id` is complete | Dropped the column |

| `brand` 43% empty | Same median price with or without a brand, but "unknown" brands sell above average | Filled with `"unknown"` |

| Zero or negative prices | Mostly on real view events | Kept the row, cleared the price |

| A user with 5,343 visits in 14 days | A new cart action every \~1.5 seconds, product IDs in sequence, no purchases | Confirmed bot, removed from training only |



A before-and-after check confirmed that cleaning removed errors without changing the shape of the data.



!\[Before and after cleaning](reports/figures/before\_after\_cleaning.png)



\---



\## What I found in the data



I asked 11 questions, and each answer fed a decision.



\- \*\*The cart is the real problem.\*\* Most non-buyers view one product and leave. Among cart users, 70% never buy; buyers are about 3× more active and edit their cart more.

\- \*\*Timing matters.\*\* Half of buyers check out within an hour of their first cart, and 84% buy within 7 days.

\- \*\*Price doesn't separate buyers, brand does.\*\* Buy rates range from 0.3% to 50% across brands.

\- \*\*Sales follow the buy rate, not traffic.\*\* Visitors stay stable every month. November (Black Friday) sold the most because more visitors bought, not because more came.



!\[Monthly sales story](reports/figures/monthly\_comparison.png)



!\[December day by day](reports/figures/december\_daily.png)



\---



\## How the model works



\### Features (14, each tied to an EDA finding)



| Group | Features |

|---|---|

| Activity | views, cart adds, cart removes, past purchases, visits |

| Recency | days since last activity, since first activity, since last cart action |

| Behavior ratios | views per visit, removes per cart item |

| Product signals | average price, average brand buy rate (from the history window only) |

| Timing | hour and weekday of the last visit |



\### Time-based split



| Part | Months | Used for |

|---|---|---|

| Train | October, December | Learning |

| Validation | January | Comparing models, tuning, choosing the threshold |

| Test | February | \*\*Used once\*\*, after every decision was locked |

| Monitoring | November | Checking behavior change (Black Friday) |



A random split would let the model learn from the future. The final model was retrained on October, December and January

