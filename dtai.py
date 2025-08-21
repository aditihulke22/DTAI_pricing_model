import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("wcoo.csv")
df1 = pd.read_csv('upccoo.csv')
df = df.dropna(subset=['WEEK'])

df2 = df1[['UPC', 'SIZE']]
df = pd.merge(df, df2, how='left', on=['UPC'])

df = df.dropna(subset=['SIZE'])

# column to clean
COL = "SIZE"          # <-- change to your column name
NEW = "SIZE_FLOAT"    # new numeric column

# 1) clean text and strip OZ/O/Z; keep only number chars; to float
tmp = (df[COL]
       .astype(str)
       .str.upper()
       .str.replace(r'\bOZ\b', '', regex=True)   # remove whole-word 'OZ'
       .str.replace(r'[OZ]', '', regex=True)     # remove stray O or Z
       .str.replace(r'[^0-9.\-]', '', regex=True)  # keep digits, dot, minus
       .str.strip()
       .replace('', np.nan))

df[NEW] = pd.to_numeric(tmp, errors='coerce')

# 2) drop rows where conversion failed
df = df.dropna(subset=[NEW]).reset_index(drop=True)

df = df[~(df['MOVE'] == 0.0)]
df['PRICE_PER_OZ'] = df['PRICE']/df['SIZE_FLOAT']

df3 = pd.DataFrame()
df3['MOVE'] = df.groupby(['UPC','STORE','WEEK','SIZE_FLOAT'])['MOVE'].sum()
df3['AVG_PRICE_PER_OZ'] = df.groupby(['UPC','STORE','WEEK','SIZE_FLOAT'])['PRICE_PER_OZ'].mean()
df3 = df3.reset_index()

X = df3[['UPC', 'STORE', 'WEEK', 'SIZE_FLOAT','AVG_PRICE_PER_OZ']]
y = df3[['MOVE']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#print('XGBoost R2 score:', r2_score(y_test, y_pred))

#UPC_v = 76672750557
#STORE_v = 144.0
#WEEK_v = 353
#SIZE_v = 15
#PRICE_v = 0.5


features = [[UPC_v, STORE_v, WEEK_v, SIZE_v, PRICE_v]]
sales = model.predict(features)[0]

print("The predicted sales for product id " + str(UPC_v) + " in week " + str(WEEK_v) + " and store number " + str(STORE_v) + " is " + str(round(sales)) + " units")
print("The corresponding SKU size is " + str(SIZE_v) + " Oz,at per Oz price of $" + str(PRICE_v))