# Useful Py functions for DS when building ML models

## 1. Exploratory Data Analysis

### VIF

To recursively eliminate features with high VIF
```Py
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def list_high_vif(df, thresh=4.0):
    X = df
    X = X.assign(const=1)  # faster than add_constant from statsmodels
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]
        vif = vif[:-1]  # don't let the constant be removed in the loop.
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc])
            del variables[maxloc]
            dropped = True
    return list(np.setdiff1d(df.columns, X.columns[variables[:-1]]))
```

### Histogram

To visualize the distribution of a feature (normalized or not) within a dataset
```Py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_hist(df, feature, dropna=True, pct=True, save_pic=False, **kwargs):
    if dropna:
        plt_data = df[df[feature].notnull()][feature]
    else:
        plt_data = df[feature]
    # plt.xticks(np.arange(0,100,5))
    if pct:
        wgt = np.ones(len(plt_data)) / len(plt_data)
    else:
        wgt = None
    plt.hist(plt_data, weights=wgt, bins=kwargs.get('bins'), range=kwargs.get('range'))
    plt.xlabel(feature)
    if pct:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    if save_pic:
        plt.savefig(feature + '.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Box Plots

To compare the distributions of a feature across datasets (eg train vs test)
```Py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_box_plot(df1,df2,feature, outliers=False):
    feat1=df1[feature]
    feat2=df2[feature]
    dict = {'df1':feat1, 'df2':feat2}
    df = pd.DataFrame(dict)
    ax = sns.boxplot(x='variable', y='value', data=pd.melt(df), showfliers = outliers)
    ax.set(xlabel=None)
    ax.set(ylabel=feature)
    plt.show()
```

![image](https://github.com/user-attachments/assets/f7f922fd-ef35-4f72-982c-1e815244d3c6)




## 2. Feature Engineering

To add a percentile column based on a value column
```Py
import pandas as pd

def add_percentile_column(df, value_column, qtl, new_column, higher=True, drop=True):
    qtl = df[value_column].quantile(qtl)
    if higher:
        df[new_column] = df[value_column].ge(qtl).astype(int)
    else:
        df[new_column] = df[value_column].le(qtl).astype(int)
    if drop:
        df = df.drop(value_column, axis=1)
    return df
```


## 3. Model Performance
### Classification models

#### Thresholds

To help with deciding probability threshold
```Py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_thresholds(y_test, y_prob, step=0.1):
    thresholds = np.arange(0, 1, step)
    # evaluate each threshold
    clas = [convert_prob_to_class(y_prob, t) for t in thresholds]
    size = [sum(c) for c in clas]
    precision = [precision_score(y_test, c) for c in clas]
    recall = [recall_score(y_test, c) for c in clas]
    accuracy = [accuracy_score(y_test, c) for c in clas]
    f1 = [f1_score(y_test, c) for c in clas]
    df = pd.DataFrame({'threshold':thresholds, 'size':size, 'precision':precision, 'recall':recall, 'accuracy':accuracy, 'f1':f1})
    pct_cols = ['threshold','precision','recall', 'accuracy', 'f1']
    df[pct_cols] = df[pct_cols].applymap("{:.0%}".format)
    df['size'] = df['size'].apply(lambda x: "{:,}".format(x))
    return df
```
![image](https://github.com/user-attachments/assets/169d0aa0-b2a1-4984-a8bd-23fa66587529)


#### Precision Recall Curves

To visualize pr and size curves. Takes a test vector and a list of probability vectors (to compare models) 
```Py
import pandas as pd
import matplotlib.pyplot as plt

def plot_pr_size(y_test, y_probs, labels, xmin=0.6, xmax=0.9, p_ymin=0.2, p_ymax=0.5, r_ymin=0, r_ymax=0.05, s_ymin=0, s_ymax=500):
    plt.clf()
    i=0
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10,10))
    ax3 = ax1.twinx()
    while i < len(y_probs):
        y_prob = y_probs[i]
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        lbl = labels[i]
        ax1.plot(thresholds, precision[:-1], label = 'precision - ' + lbl, linewidth=3)
        ax3.plot(thresholds, recall[:-1], label =  'recall - ' + lbl)
        sizes = [ len(y_prob[y_prob>=p]) for p in thresholds ]
        ax2.plot(thresholds, sizes, label =  'size - ' + lbl)
        i+=1
    ax1.grid(True)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.tick_params(labelbottom=True)
    ax1.set_ylabel('Precision')
    ax1.set_ylim(bottom=p_ymin, top=p_ymax)
    ax1.legend(loc='upper right')
    ax3.set_ylabel('Recall')
    ax3.set_ylim(bottom=r_ymin, top=r_ymax)
    ax3.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_ylabel('Bin Sizes')
    ax2.set_ylim(bottom=s_ymin, top=s_ymax)
    ax2.legend(loc='lower left')
    ax2.set_xlabel('Threshold')
    plt.show()
```
![image](https://github.com/user-attachments/assets/f7925423-a70a-4ace-bb52-d70fb498ae95)


