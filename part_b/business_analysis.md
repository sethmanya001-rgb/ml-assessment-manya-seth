# Part B: Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target Variable:** items_sold (number of items sold per store per month)

**Candidate Input Features:**
- Store features: store_id, store_size, location_type, 
  monthly_footfall, competition_density
- Promotion features: promotion_type
- Time features: month, year, is_weekend, is_festival
- Customer features: customer demographics per store

**Type of ML Problem:** 
This is a Supervised Regression problem because:
- We are predicting a continuous numerical value (items sold)
- We have historical labeled data (past months with known items sold)
- The goal is to predict future items sold for each 
  store-promotion combination

We could also frame it as a Recommendation problem — 
recommending the best promotion for each store each month 
by predicting items_sold for all 5 promotions and 
selecting the one with the highest predicted value.

---

### B1(b) — Why Items Sold is Better Than Revenue

Revenue is influenced by many factors outside the promotion's 
control such as:
- Product price changes
- Discount percentage applied
- Product mix (expensive vs cheap items)
- Seasonal pricing strategies

Items sold is a cleaner and more direct measure of 
promotion effectiveness because it purely reflects 
how many products the promotion moved off the shelf, 
regardless of their price.

**Broader Principle:** 
Target variable selection should reflect the actual 
business objective as directly as possible. Using a 
proxy variable (like revenue) that is affected by 
unrelated factors introduces noise and makes the model 
harder to interpret and trust. Always choose the target 
that most directly measures what you are trying to 
optimise.

---

### B1(c) — Alternative to One Global Model

A single global model assumes all 50 stores respond 
identically to promotions, which ignores important 
differences in location, customer demographics, 
and competition.

**Proposed Strategy: Location-Type Segmented Models**

Train separate models for each location type:
- Model 1: Urban stores
- Model 2: Semi-urban stores  
- Model 3: Rural stores

**Why this works better:**
- Urban customers may respond better to BOGO and 
  Category-Specific Offers
- Rural customers may respond better to Flat Discounts
- Semi-urban stores may behave differently from both

This balances the trade-off between:
- Too specific: 50 individual models (not enough data 
  per store)
- Too general: 1 global model (ignores store differences)

Alternatively, include store_id and location_type as 
features in one model with interaction terms, allowing 
the model to learn store-specific patterns automatically.

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables

**Tables:**
1. transactions — one row per transaction
2. store_attributes — one row per store
3. promotion_details — one row per promotion
4. calendar — one row per date with weekend/festival flags

**Join Strategy:**
- Join transactions with store_attributes on store_id
- Join result with promotion_details on promotion_id
- Join result with calendar on transaction_date

**Grain of Final Dataset:**
One row = one store × one month × one promotion type

This means we aggregate all daily transactions for a 
given store in a given month under the promotion 
used that month.

**Aggregations performed:**
- Sum items_sold per store per month
- Count total transactions per store per month
- Average basket_size per store per month
- Attach store attributes (size, location, footfall)
- Attach promotion type used that month
- Attach calendar flags (is_festival, is_weekend count)

---

### B2(b) — EDA Strategy

**Analysis 1: Items Sold by Promotion Type (Bar Chart)**
- What to look for: Which promotion drives highest 
  average items sold across all stores
- Impact: Identifies baseline promotion effectiveness 
  before accounting for store differences

**Analysis 2: Items Sold by Location Type (Box Plot)**
- What to look for: Do urban stores sell more than 
  rural stores? Is there high variance within groups?
- Impact: If location types differ significantly, 
  we should build separate models or add interaction 
  features between location_type and promotion_type

**Analysis 3: Monthly Sales Trend (Line Chart)**
- What to look for: Seasonal peaks (festive months), 
  year-on-year growth trend, any sudden drops
- Impact: Confirms that month and is_festival are 
  important features. May suggest adding lag features 
  (last month sales)

**Analysis 4: Promotion Type vs Location Type Heatmap**
- What to look for: Does BOGO work better in urban 
  areas while Flat Discount works better in rural areas?
- Impact: Reveals interaction effects that should be 
  captured as interaction features in the model

---

### B2(c) — Handling Promotion Imbalance

**Problem:**
80% of transactions have no promotion. The model may 
learn to ignore promotions entirely and just predict 
based on store size, season etc., making promotion 
recommendations unreliable.

**Steps to address:**

1. Stratified Sampling: Ensure promoted and 
   non-promoted records are proportionally represented 
   in both train and test sets

2. Separate Evaluation: Evaluate model performance 
   separately on promoted vs non-promoted transactions 
   to ensure it learns both patterns well

3. Add Binary Feature: Create a has_promotion column 
   (1 if any promotion was active, 0 if not) so the 
   model explicitly knows when a promotion is running

4. Oversample Promoted Records: Use techniques like 
   SMOTE or simple random oversampling on the minority 
   promoted class to balance the training data

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split Strategy

**Setup:**
With 3 years of monthly data across 50 stores we have 
approximately 1800 store-month records (50 stores × 36 months)

**Split Strategy:**
- Train: Months 1 to 30 (first 2.5 years)
- Test: Months 31 to 36 (last 6 months)

This is a time-based split — all training data comes 
before all test data.

**Why Random Split is Inappropriate:**
Random split mixes future months into training data, 
causing data leakage. The model learns from future 
information it would not have in real deployment, 
producing artificially optimistic performance metrics.

**Evaluation Metrics:**

- RMSE (Root Mean Squared Error): Measures average 
  prediction error in units of items sold. Penalises 
  large errors more heavily. Important for inventory 
  planning where large errors are costly.

- MAE (Mean Absolute Error): Average absolute difference 
  between predicted and actual items sold. Easier for 
  business stakeholders to interpret — "on average we 
  are off by X items per month"

- MAPE (Mean Absolute Percentage Error): Percentage 
  error — useful for comparing performance across 
  stores of very different sizes (a 50-item error 
  means different things for a large vs small store)

---

### B3(b) — Feature Importance for Explaining Recommendations

**Scenario:** Model recommends Loyalty Points Bonus 
for Store 12 in December but Flat Discount in March.

**Investigation Steps:**

1. Extract the top 10 feature importances from the 
   Random Forest model globally to understand which 
   features drive predictions most.

2. For December prediction: Check the values of top 
   features for Store 12 in December:
   - is_festival = 1 (Christmas/New Year season)
   - month = 12 (peak shopping month)
   - These high-engagement conditions favour Loyalty 
     Points Bonus because customers are already 
     motivated to buy — rewarding loyalty maximises 
     long-term value

3. For March prediction: Check the same features:
   - is_festival = 0 (no major festivals)
   - month = 3 (lower baseline demand)
   - In low-demand periods, Flat Discount drives 
     immediate volume more effectively than loyalty rewards

**Communication to Marketing Team:**
Present a simple bar chart showing "Top reasons for 
this recommendation" with feature values that 
influenced the decision. Use plain language:
"In December, high festive season activity makes 
Loyalty Points more effective. In March, lower footfall 
means a direct discount is needed to drive purchases."

---

### B3(c) — End-to-End Deployment Process

**Step 1 — Save the Model**
After training, save the entire pipeline including 
preprocessor and model:
- Use joblib.dump(pipeline, 'promotion_model_v1.pkl')
- Store it in a secure location (cloud storage or server)
- Version the model so older versions can be restored

**Step 2 — Monthly Data Preparation**
At the start of every month:
- Collect previous month transactions from database
- Aggregate to store × month grain
- Join with store attributes and calendar flags
- Apply the same preprocessing pipeline 
  (already fitted on training data)
- Generate 5 rows per store (one per promotion type)
  with all features filled in

**Step 3 — Generate Recommendations**
- Feed prepared data into the saved model
- Model predicts items_sold for each store × promotion 
  combination
- Select the promotion with highest predicted 
  items_sold for each store
- Output a recommendation table: 
  Store ID → Recommended Promotion for the month

**Step 4 — Monitoring and Retraining**

Monitoring:
- Every month compare predicted vs actual items_sold
- Track RMSE on a rolling 3-month window
- Set alert threshold: if RMSE increases by more 
  than 20% compared to baseline, trigger review

Retraining Signals:
- Model performance degrades beyond threshold
- New promotion types are introduced
- Major external change occurs (new competitor, 
  economic shift, pandemic-like event)
- Data distribution shifts significantly 
  (detected via feature drift monitoring)

Retraining Process:
- Add new months of data to training set
- Retrain model using updated dataset
- Validate on most recent 3 months before deploying
- Replace old saved model with new version
- Document the retraining date and reason
