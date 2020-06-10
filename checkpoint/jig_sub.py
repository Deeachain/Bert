import pandas as pd
import numpy as np

test_file = '/media/ding/Storage/competition/kaggle/jigsaw/data/test.csv'
predict = '/media/ding/Storage/competition/kaggle/jigsaw/checkpoint/jigsaw//test_results.tsv'

out_path = 'bert_multilingua  l.csv'


df = pd.read_csv(test_file, sep=',')

temp = pd.read_csv(predict, sep='\t', names=['class0', 'class1'])
array = np.array(temp)
print(array.shape)
print(array[:, 1])
df['toxic'] = array[:, 1]

df[['id', 'toxic']].to_csv(out_path, index=False, sep=',')