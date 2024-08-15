import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
from scipy import stats
from scipy.io.arff import loadarff


def convert_to_categorical_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.Categorical(df[col]).codes
    return df


def convert_to_wines_df(txt_file):
    df = pd.read_csv(txt_file)
    return df


def convert_to_abalone_df(txt_file):
    txt_file = open(txt_file, 'r')
    lines = txt_file.readlines()
    txt_file.close()
    lines = [line[:-1] for line in lines]
    df = pd.DataFrame([sub.split(",") for sub in lines])
    df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight',
                  'Viscera_weight', 'Shell_weight', 'Rings']
    return df


def convert_data_type_to_df(txt_file, columns):
    txt_file = open(txt_file, 'r')
    lines = txt_file.readlines()
    txt_file.close()
    lines = [line[:-1] for line in lines]
    df = pd.DataFrame([sub.split(",") for sub in lines])
    df.columns = columns
    return df


def set_new_label(rings, edge):
    for label, e in enumerate(edge):
        if rings <= e:
            return label
    return label + 1


class Process_data:
    def __init__(self, path, name):
        if name == "servo":
            df = pd.read_csv(path)
            df[['Motor', 'Screw']] = df[['Motor', 'Screw']].applymap(lambda letter: ord(letter) - ord('A') + 1)

        elif name == "abalone" or name.startswith("abalone"):
            df = convert_to_abalone_df(path)
            df = pd.get_dummies(df, columns=['Sex'])
            df['Length'] = pd.to_numeric(df['Length'])
            df['Diameter'] = pd.to_numeric(df['Diameter'])
            df['Height'] = pd.to_numeric(df['Height'])
            df['Whole_weight'] = pd.to_numeric(df['Whole_weight'])
            df['Shucked_weight'] = pd.to_numeric(df['Shucked_weight'])
            df['Viscera_weight'] = pd.to_numeric(df['Viscera_weight'])
            df['Shell_weight'] = pd.to_numeric(df['Shell_weight'])

        elif name == "tae":
            df = convert_data_type_to_df(path, ["English", "instructor", "Course", "semester", "size", "label"])
            df = df.apply(lambda x: x.astype(int))

        elif name == "autos":
            df = pd.read_csv(path)
            non_nominal_columns = ['make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location',"engine-type","num-of-cylinders","fuel-system"]

            non_nominal_df = df[non_nominal_columns]
            nominal_df = df.drop(columns=non_nominal_columns)

            one_hot_encoded_df = pd.get_dummies(non_nominal_df, drop_first=True)

            df = pd.concat([nominal_df, one_hot_encoded_df], axis=1)
            df.replace('?', None, inplace=True)



        elif name == "balance-scale":
            df = convert_data_type_to_df(path,
                                         ["label", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"])
            mapping = {'L': 1, 'B': 2, 'R': 3}
            df['label'] = df['label'].map(mapping)
            df = df.apply(lambda x: x.astype(int))

        elif name == "eucalyptus":
            df = pd.read_csv(path)
            mapping = {'none': 1, 'low': 2, 'average': 3, 'good': 4, 'best': 5}
            df['Utility'] = df['Utility'].map(mapping)
            df = df.apply(pd.to_numeric, errors='ignore')
            df = convert_to_categorical_numeric(df)

        elif name == "thyroid":
            df = pd.read_csv(path)
            df.columns = ["c1", "c2", "c3", "c4", "c5", "c6"]

        else:
            df = pd.read_csv(path)
        self.df = df
        self.name = name

    def get_binned_data(self, edge):
        if self.name == "wine" or self.name.startswith("wine"):
            self.df['y'] = self.df['quality'].apply(lambda x: set_new_label(x, edge))
            self.df.drop('quality', axis=1, inplace=True)

        elif self.name == "boston" or self.name.startswith("boston"):
            self.df['y'] = self.df['MEDV'].apply(lambda x: set_new_label(x, edge))
            self.df.drop('MEDV', axis=1, inplace=True)

        elif self.name == "servo":
            self.df['y'] = self.df['Class'].apply(lambda x: set_new_label(x, edge))
            self.df.drop('Class', axis=1, inplace=True)

        elif self.name == "abalone" or self.name.startswith("abalone"):
            self.df['y'] = self.df['Rings'].apply(lambda x: set_new_label(float(x), edge))
            self.df.drop('Rings', axis=1, inplace=True)

        elif self.name == "tae" or self.name == "balance-scale":
            self.df['y'] = self.df['label'].apply(lambda x: set_new_label(x, edge))
            self.df.drop('label', axis=1, inplace=True)

        elif self.name == "eucalyptus":
            self.df['y'] = self.df['Utility'].apply(lambda x: set_new_label(x, edge))
            self.df.drop('Utility', axis=1, inplace=True)

        elif self.name == "thyroid":
            self.df['y'] = self.df['c6'].apply(lambda x: set_new_label(x, edge))
            self.df.drop('c6', axis=1, inplace=True)

        elif self.name == "SWD" or self.name == "LEV":
            self.df['y'] = self.df['Out1'].apply(lambda x: set_new_label(x, edge))
            self.df.drop('Out1', axis=1, inplace=True)

        else:
            self.df['y'] = self.df['out1'].apply(lambda x: set_new_label(x, edge))
            self.df.drop('out1', axis=1, inplace=True)

        X = self.df.drop('y', axis=1).copy()
        y = self.df['y'].copy()

        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=4, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.75, random_state=4,
                                                          stratify=y_train)

        return x_train, y_train, x_val, y_val, x_test, y_test
