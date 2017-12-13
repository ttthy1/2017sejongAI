from sklearn.datasets import load_linnerud
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


linnerud = load_linnerud()

print(linnerud.DESCR)

df = pd.concat([pd.DataFrame(linnerud.data, columns=linnerud.feature_names),
                pd.DataFrame(linnerud.target, columns=linnerud.target_names)],
               axis=1)
df

print (df)

sns.pairplot(df.ix[:,:4])
plt.show()
