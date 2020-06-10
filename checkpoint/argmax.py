import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_file", default=None, type=str, required=True)
parser.add_argument("--predict", default=None, type=str, required=True)
parser.add_argument("--out_path", default=None, type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.test_file, sep=',')

temp = pd.read_csv(args.predict, sep='\t', names=['class0', 'class1'])
array = np.array(temp)

df['toxic'] = np.argmax(array, axis=1)
print(df['toxic'])
# print(df['id'])
df[['id', 'toxic']].to_csv(args.out_path, index=False, sep=',')
