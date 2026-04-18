import pandas as pd

data = [
    ["MFCC + MLP",40.97,39.94],
    ["Wav2Vec2 Frozen",62.50,62.51],
    ["EMADropFormer",82.00,82.00],
]

df=pd.DataFrame(data,columns=["Model","Accuracy","Weighted_F1"])
print(df)

df.to_csv("results/metrics.csv",index=False)
print("Saved results/metrics.csv")