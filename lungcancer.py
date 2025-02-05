{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>SMOKING</th>\n",
       "      <th>YELLOW_FINGERS</th>\n",
       "      <th>ANXIETY</th>\n",
       "      <th>PEER_PRESSURE</th>\n",
       "      <th>CHRONIC DISEASE</th>\n",
       "      <th>FATIGUE</th>\n",
       "      <th>ALLERGY</th>\n",
       "      <th>WHEEZING</th>\n",
       "      <th>ALCOHOL CONSUMING</th>\n",
       "      <th>COUGHING</th>\n",
       "      <th>SHORTNESS OF BREATH</th>\n",
       "      <th>SWALLOWING DIFFICULTY</th>\n",
       "      <th>CHEST PAIN</th>\n",
       "      <th>LUNG_CANCER</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>M</td>\n",
       "      <td>69</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>M</td>\n",
       "      <td>74</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>F</td>\n",
       "      <td>59</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>M</td>\n",
       "      <td>63</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>F</td>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
       "0      M   69        1               2        2              1   \n",
       "1      M   74        2               1        1              1   \n",
       "2      F   59        1               1        1              2   \n",
       "3      M   63        2               2        2              1   \n",
       "4      F   63        1               2        1              1   \n",
       "\n",
       "   CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  COUGHING  \\\n",
       "0                1         2         1         2                  2         2   \n",
       "1                2         2         2         1                  1         1   \n",
       "2                1         2         1         2                  1         2   \n",
       "3                1         1         1         1                  2         1   \n",
       "4                1         1         1         2                  1         2   \n",
       "\n",
       "   SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN LUNG_CANCER  \n",
       "0                    2                      2           2         YES  \n",
       "1                    2                      2           2         YES  \n",
       "2                    2                      1           2          NO  \n",
       "3                    1                      2           2          NO  \n",
       "4                    2                      1           1          NO  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(r\"C:\\Agile\\CSV_or_Data_Files\\lungcancer.csv\")\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['LUNG_CANCER'] = df['LUNG_CANCER'].apply(lambda x: 1 if x == 'YES' else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\anand\\AppData\\Local\\Temp\\ipykernel_6496\\1272081959.py:2: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  df.GENDER.replace(to_replace=['M', 'F'], value=[0, 1], inplace=True)\n",
      "C:\\Users\\anand\\AppData\\Local\\Temp\\ipykernel_6496\\1272081959.py:2: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`\n",
      "  df.GENDER.replace(to_replace=['M', 'F'], value=[0, 1], inplace=True)\n"
     ]
    }
   ],
   "source": [
    "df.LUNG_CANCER.replace(to_replace=['no', 'yes'], value=[0, 1], inplace=True)\n",
    "df.GENDER.replace(to_replace=['M', 'F'], value=[0, 1], inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>SMOKING</th>\n",
       "      <th>YELLOW_FINGERS</th>\n",
       "      <th>ANXIETY</th>\n",
       "      <th>PEER_PRESSURE</th>\n",
       "      <th>CHRONIC DISEASE</th>\n",
       "      <th>FATIGUE</th>\n",
       "      <th>ALLERGY</th>\n",
       "      <th>WHEEZING</th>\n",
       "      <th>ALCOHOL CONSUMING</th>\n",
       "      <th>COUGHING</th>\n",
       "      <th>SHORTNESS OF BREATH</th>\n",
       "      <th>SWALLOWING DIFFICULTY</th>\n",
       "      <th>CHEST PAIN</th>\n",
       "      <th>LUNG_CANCER</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>69</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>74</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>59</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>63</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
       "0       0   69        1               2        2              1   \n",
       "1       0   74        2               1        1              1   \n",
       "2       1   59        1               1        1              2   \n",
       "3       0   63        2               2        2              1   \n",
       "4       1   63        1               2        1              1   \n",
       "\n",
       "   CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  COUGHING  \\\n",
       "0                1         2         1         2                  2         2   \n",
       "1                2         2         2         1                  1         1   \n",
       "2                1         2         1         2                  1         2   \n",
       "3                1         1         1         1                  2         1   \n",
       "4                1         1         1         2                  1         2   \n",
       "\n",
       "   SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN  LUNG_CANCER  \n",
       "0                    2                      2           2            1  \n",
       "1                    2                      2           2            1  \n",
       "2                    2                      1           2            0  \n",
       "3                    1                      2           2            0  \n",
       "4                    2                      1           1            0  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:  0.967741935483871\n",
      "Accuracy in percentage:  96.7741935483871 %\n"
     ]
    }
   ],
   "source": [
    "x = df.drop('LUNG_CANCER', axis=1)\n",
    "y = df['LUNG_CANCER']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)\n",
    "model = GaussianNB()\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy: \", accuracy)\n",
    "print(\"Accuracy in percentage: \", accuracy*100, \"%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:  0.967741935483871\n",
      "Accuracy in percentage:  96.7741935483871 %\n"
     ]
    }
   ],
   "source": [
    "#random forest\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)\n",
    "model = RandomForestClassifier()\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy: \", accuracy)\n",
    "print(\"Accuracy in percentage: \", accuracy*100, \"%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:  0.967741935483871\n",
      "Accuracy in percentage:  96.7741935483871 %\n"
     ]
    }
   ],
   "source": [
    "#decision tree\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)\n",
    "model = DecisionTreeClassifier()\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy: \", accuracy)\n",
    "print(\"Accuracy in percentage: \", accuracy*100, \"%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:  0.9838709677419355\n",
      "Accuracy in percentage:  98.38709677419355 %\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Agile\\.venv\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "#logistic regression\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)\n",
    "model = LogisticRegression()\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy: \", accuracy)\n",
    "print(\"Accuracy in percentage: \", accuracy*100, \"%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:  0.9787234042553191\n",
      "Accuracy in percentage:  97.87234042553192 %\n"
     ]
    }
   ],
   "source": [
    "#SVM\n",
    "from sklearn.svm import SVC\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)\n",
    "model = SVC()\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy: \", accuracy)\n",
    "print(\"Accuracy in percentage: \", accuracy*100, \"%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:  0.9516129032258065\n",
      "Accuracy in percentage:  95.16129032258065 %\n"
     ]
    }
   ],
   "source": [
    "#KNN\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)\n",
    "model = KNeighborsClassifier()\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy: \", accuracy)\n",
    "print(\"Accuracy in percentage: \", accuracy*100, \"%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA2MAAAImCAYAAADe01JiAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAkKRJREFUeJzt3Qd4VFX6x/EXQiD0Jr333rvSpKMoiF2wFyxgd113rWtBXBcFewHFXRVRxIqAgGKhS+899BY6oYb8n98JM/8kBMhAkjvl+3meIcPMnTtnZs7cue8973lvtsTExEQDAAAAAGSp7Fn7dAAAAAAAIRgDAAAAAA8QjAEAAACABwjGAAAAAMADBGMAAAAA4AGCMQAAAADwAMEYAAAAAHiAYAwAAAAAPEAwBgBAhEpMTPS6CQAQ0QjGACCEPfLII1ajRg0bPny4100JKcePH7e///3v1qhRI2vcuLFNnz493Y/t0KGDe6yP3v833njD//8ff/zRLr74Yqtbt649/fTTtnXrVuvTp4/Vq1fPWrVqZYcOHbJgMGnSJHv88cct2KR+fzPrMQAQDHJ43QAAwLnZv3+/TZw40apXr25ffPGF3XrrrZYtWzavmxUSfv/9dxszZozde++9duGFF1rt2rXPeV1670uWLOn//7/+9S+rWLGivfzyy1aiRAkbMWKEzZs3z/7973+7/+fOnduCwccff+x1EwAg4hGMAUCI+uGHH9zff/7zn3bzzTe70R2NvODs9uzZ4/727t3bypUrd17ratiw4Snrvuiii6xFixb+/xcvXtwuueSS83oeAED4IU0RAELU6NGjXfDVsmVLq1Chgo0cOfKUZb755hu74oorrEGDBta+fXv7z3/+Y0ePHvXfrxGb2267zaXqaT0PP/ywbdu2zd339ddfuxS8jRs3njVN780333SBTf369d11mTVrlt1+++3WrFkzl7Knxymd78SJE/7HHjhwwJ5//nlr06aNC2quvPJK+/XXX919gwYNcuvTCGByb7/9tjVp0uS06X4JCQn26aef2mWXXeYer9f96quv2pEjR9z9aruv/Z06dbIbb7zxtO/xsmXL3Iij0hmVevjdd9+dsowvTXHGjBnuurz11lvuul6z3sfNmzenSGdUgKYURo3KKX3xmmuusWnTpp2y3rTeV61Ln1Pz5s3d56pAfMmSJf7H6fPSY3/66Se7//77Xdu17JNPPmnx8fFuGb3mmTNnuouWVdvTovdJn6FG//ReqR3XXXedrV271n755Rf3HqsNV199tS1dujTFY//880+74YYb3GelwFQptVu2bAn4/VV/ef/9961z586uH3Xt2tX++9//2tkOVFx++eWuverXjz76qL9fA0AwIRgDgBC0cuVKW7hwofXq1cv9X381B2jnzp3+ZRSQaE5QnTp13I78XXfd5XZiX3jhBXe/duD79u3rgpRXXnnFnnvuOVu0aJHb+dacqkC8++67bsd86NChbmdZO9m33HKLFSpUyF577TV75513rGnTpq4dChJ8QZMCwe+//9769evngqzKlSvbfffdZ7Nnz7arrrrKtW3cuHEpnuvbb791o0ynS/dTkDNw4EAXPOh5NV/rf//7n0tJVMEK/b3nnnvcsmrPM888k+Z6tPOu90fBoFIMH3jgARfUnW6nXu+zghZR23Vdr71du3ZWrFgx938FLXpNCqD0eT300EOuDUpzvOOOO04JyFK/r7t27XLB0OLFi+2pp55ywbWCFb3G1atXp3isXleZMmXc+6rP9KuvvnLvh+8+pWbqonap7aczd+5c9/4pMNP7qudRX9J1fW6DBw92QZYCnuQHAfTZlipVyt3/xBNPuPVce+21FhcXF9D7++yzz7rXr+BK70e3bt3spZdecgFvWv766y/729/+Zl26dLEPPvjAPbdGjRUMAkDQSQQAhJyBAwcmNm/ePPHIkSPu/5s3b06sWbNm4jvvvOP+n5CQkNiqVavEe++9N8XjPvzww8Qrrrgi8ejRo4kDBgxIvOiiixIPHz7sv3/OnDmJF198ceKSJUsSR48enVi9evXEDRs2pFiH7n/88cf9/9cyN998c4plxowZk3jHHXe4dvjoepMmTRKfeuop9//Jkye7x/78888plrn22msT33jjDfd/Xe/Tp4///r/++ss9Ru1My8qVK9397733Xorbv/nmG3f7r7/+6v5/uteW3Msvv5zYsGHDxLi4OP9t8+bNc49L/fqHDh162v9rWb1nPl988YVbRuvyOXHihHudvXv3PuP7Onjw4MR69eolbty40X+b+kDHjh3d5yl6TXrso48+muKxN954Y2KPHj38/+/bt6+7nInarnWtWrXKf9vTTz/tbps6dar/tmHDhrnb9u7d6z5D9avbbrstxbpiY2MT69Spkzho0KB0v79r1qxJrFGjximf52uvvebeh127dp3SJ7Vso0aN/N8N0eeuPqX3GQCCCSNjABBijh075tK5NPJz+PBh27dvn+XNm9elg40aNcqNlCiNTCMQSu1KTiMkSpuLjo52Iwht27a1XLly+e9XutjkyZOtVq1aAbUp9fIaqdOohNqqUbLx48e70Q2Nhuk20fOrHUrl88mePbtLt+zfv7/7v9IWNUq2adMm938V3ahUqZJrZ1qUdieXXnppitv1/6ioqNOm46VF7VPqZJEiRfy3KSWvdOnSdj40+qWRMo1GaQRSF70vStPTyOTevXtP+77qsbpNhUB8j9V7ps9x6tSpZ5zLptE3X5piIAoWLGhVqlTx//+CCy7wvxc+GgEV9UX1vR07dliPHj1SrKd8+fLuc/N9Rul5fzWipdFM9RHf69VF/9cIo9aRmtJilcKq59fIofpP69atXZ+iwA2AYEMBDwAIMZpTpUBLaWe6pFUpMF++fO560aJFT7sezVs60/2ByJMnT4r/K0jUXDClFGrnuWzZsm5HPEeOHP5zW+n5tROvYOJ0lI6olDStR4GkUhyVInc6vkBGwU5yet7ChQufMv/sTLQutTu11OsOlF63gpXTpQbqPgVAab2vemxsbOxpH5t8Hl3qNE69z+dyXjFfX0otdduStzF50JacbvPNb0vP++tbV+rg2ietlFH1M80xU7XIjz76yF3X8959991nnB8IAF4gGAOAECzcoQqAL774YorbtaOto/8aWVKBB9Eco+R2797tdoa1w5o/f/5T7pcpU6a40RffKELyghty8ODBs7ZRbdNo2Ouvv+6KVPh23JNXe9Tza2db7U4+YqH26TYFHBrx0xwhBWEq4a+RnZ49e572eX1BjAIazZfy0WicXrsCsvTSssnn4KUOEM6VXrdK32t+VFrSClCSP1bFODQnKi05c+Y0r/lGydJ67/S5+D6D9Ly/BQoUcH91egD1hdRON0qpgjC6KDjV6Nonn3zi5kpq5E1FPQAgWJCmCAAhRDuzGvnSSIEq1CW/qGqcAhcFU9qJ1c6uKt4lpxEmjSwpOFFBDVW8S15dUYGQ7leBCN+IiE5a7KPiDekJRpQ+pjYpldIXiCkFT8GfL7jT86sdv/32m/9xCsJUcOG9997z36ZiGCtWrHA75ArslKJ3OgpUfCdeTk7/VyqgUjnTS++nik4kH31ZtWqVbdiwId3rOF0bVfBCo5KqpOi76LP48MMPXTrlmR6rNEClaiZ/rD5XjZKe6bGpnWlE8nyobRrd8p16wUfvm6p3qnJnet9f9RFRIJ389aofDRkyJM2+qCqcSm9VX9LooNI/fSe3ViVKAAgmjIwBQAhRlTql/Z0ubUtztb788ks3d2zAgAHuBMTa6dccG+3Ea96WKu9pBElVBVXdThXxbrrpJpdaqJEsjRzoPFn6f0xMjDt5sSrdaURMj/eNfJyJ1qHRrM8//9zNN9K8MVXy0wiYL5VOJec1QqcqfQ8++KAb7VNQoYBPKY4+CqC0g6+5RqpOeCZVq1Z1pfzVTj2P5g+p5LoqFio41GhJeqnioQIcpUfqvVQwp+fXPLfzoVL1qk6oku5KnVPFQc330hw7VRc80/pVoVLvkf6qWqEC7rFjx7rPW0FsIBSwKxjSPDRVVfSNKp4vBXkamVV7VMFQVRAVTOkz0HPodaf3/VXZfT1elSM1b1Cl7dWPtZxGEDXCmJqCPKUnql/psQr4FeSq3+o+AAgmBGMAEEJUfKNatWouZS8tCly0k6qATKNiGpUaNmyYK1+uAg533nmnu4h2wFXqXkUOFAxpJExl2FWiXOluuui8WLpf5eaV9qc0SAWEZ6MdYe0EK7jTyJvapHLyGvlQgRDteGsURwGI0vU0yqHgSTvfw4cPPyWVTIGbRkM00paeFEmdd03pnFq/TrisYFPBZyCjQQp0FExqfXo9SpNT+XkFP+dDn4lOO6D3VSXdNY9N760CFwVYZ6JRQaWh6rEq+a4iFgpI1EaNIAZCQblGK9UfVKZeJfQzigJOvV8a4VTfUd9SIKwgzTcnLL3vr9qm9eh1a5RWBxc0l1B9Nq2RQPVh9Sn1I1/RDn0vlKqYngMJAJCVsqmkYpY+IwAAAdDPlEYCVRHvH//4h9fNAQAgwzAyBgAISgcOHHAV8XRya80johIeACDcEIwBAIKS5qspNU0FP1TeXnPKAAAIJ6QpAgAAAIAHKG0PAAAAAB4gGAMAAAAADxCMAQAAAIAHKOCRAXTSTE29O98TgQIAAAAIbTrPps5x2KhRo7Muy8hYBlAgFix1UNQOnWA1WNqD4EefQaDoMwgUfQaBos8glPtMILGB5yNjKln85ptv2pdffmn79++3Zs2a2dNPP33aEsbr1q1zJY7nzJljefLksauuusruvfdey5Hj/1/KJ598Yv/9739tx44dVrlyZXvggQesXbt2/vvfeecde/31109Z9/Lly8/pNfhGxOrVq2dei4+Pt6VLl1rVqlXd+wOcDX0GgaLPIFD0GQSKPoNQ7jM6P2Z6eT4y9vbbb9tnn31mzz//vP98MnfccYeLbFPbu3ev9enTxw4dOmQjRoywwYMH208//eSCN5+vv/7aXnvtNXvkkUfs+++/d0HYfffdZ8uWLUsRdPXs2dP++OOPFBcAAAAAyCqeBmMKuIYPH27333+/tW/f3mrWrOkCqa1bt9qECRNOWX7MmDEu6h0yZIjVqVPHmjZtai+88IKNHj3aNm7c6JaZOHGitW7d2rp16+ZG1zQqpuh42rRp/vWsWLHCateubcWKFUtxAQAAAICICMY0WnXw4EFr1aqV/7YCBQq4QGnWrFmnLB8bG+vSDosUKeK/TcvK7Nmz3d+iRYu6x2rdytUcO3asS3/0pRAqAFSqo9YDAAAAAF7xdM6YRsCkVKlSKW4vXry4/77Ut2/fvt0SEhIsKirK3bZp0yb3Ny4uzv0dMGCArVq1yqUhahmlPT777LNuFE10nx4/fvx4e/HFF+3IkSNuntpjjz3m1g8AAAAAYR+Mae6X5MyZM8XtuXLlcvPDUuvevbubYzZw4EB7+OGHXcqi0hRVvEMlJGX9+vUuAHvllVesWrVqLt1RQVeZMmWsTZs2LkVRcufO7dIdFcRp7tlNN91k33zzjcXExJzTa9EonNrjNd976vsLnA19BoGizyBQ9BkEij6DUO4zigtU2j7ogzFf4KPUweRBkEarFCylVrFiRRdAqWDHp59+6uaC+UbC8ufP74IhFet44okn3MiYL41Ro2evvvqqC8Z69eplbdu2TZHqqKBNt02ePNkuueSSc3otCgZVwSVYKBUTCAR9BoGizyBQ9BkEij6DUO0zqQebgjIY86UnKvWwfPny/tv1/xo1aqT5mA4dOriLlilUqJAdP37cXn75ZVesY/Xq1bZnz55TSsw3bNjQfv75Z///kwdiovRErSut1MhAyturlKbXdDRAnVCBa1oBLZAafQaBos8gUPQZBIo+g1DuMxooSi9PgzFVT8yXL5/NmDHDH4zt27fPlixZYn379j1leRXp0MjYRx995J/fpQIdesMbN27sH5ZU6foqVar4H6f/64MRVWscN26cu/iGD1WJcffu3ecVTGldXp/TIDm9J8HUHgQ/+gwCRZ9BoOgzCBR9BqHYZ9Kbouh5NUUN3ynoUgrhpEmTXAXEhx56yEqWLGldunRxhTZ04ubDhw+75VUBUYHVoEGDbMOGDa6MveaM9evXzwV1Kk/fo0cPd1JorU/L6ATQKn1/9913u3V07tzZpS2qqMfatWtd5UWlOiqYUxojAAAAAGQFT0fGROcYU6rhk08+6YIuVTYcNmyYS/vTiFXHjh1dwY7evXu79MJ3333XpSUq6FLw1b9/f7vlllv861Oxjnfeeccts3PnTqtUqZIr0NG1a1d3f926de2DDz5wI2xapwJCPcfjjz8eUBQLAAAAACEdjKn8vMrK65Ja2bJl3UhYchrBGjVq1GnXp0IgGl3T5XR0XrPk5zYDAAAAgKzmaZoiAAAAAEQqgjEAAAAA8ADBGAAAAAB4gGAMAABkmoQTibZ47S5buC7e/dX/AQBBUsADAACEp6kLNtv73yy0uL1Jp6gZPXWXFS242O7qVc8urF/a6+YBgOcYGQMAAJkSiA0cMcsfiPno/7pd9wNApCMYAwAAGUqpiBoRO5MPvl1EyiKAiEcwBgAAMtSSNXGnjIiltnPPIbccAEQygjEAAJChdu07nKHLAUC4IhgDAAAZqkiBmHQtF7f3kCUmkqoIIHIRjAEAgAxVu3JRK1rw7AHZRz8ssYeH/GbTFm62E8wfAxCBCMYAAECGisqezepUKnrGZZrWKmE5o6Ns1YY99tLHs6z/q5Nt8uwNdjzhRJa1EwC8xnnGAABAhlq7ea9NXZhUuj5fnmg7EH/Mf98FhXLbnT3ruvOM7T1wxL77fY39+Mca27DtgL32+Rz7dNxS631xNevUvLzlio7y8FUAQOYjGAMAABlGI1uvfz7XjickWos6Je3vNzezucs22+Jla61OzUrWuFYZN3ImBfPlshu717Le7ava2Klr7bvf1tj23Yfs3a8X2Mifl1vPtlXskgsrWp6YaK9fFgBkCoIxAACQYb6cuMLWbN5r+fNE231XNbAcUdmtTqUilv3wNqtVqYg/EEsub+5ou7pjdbu8bRWbOCPWRv+6ynbsPmQjflxiX01eaT0uqmSXtansgjcACCcEYwAAIEOs3rjHvpi4wl2/u3d9K5zOqoo+Sku8tHVl69qqok2Zs9EFYhu3H3Dr/Oa31da1ZQW7ol1Vl+oIAOGAYAwAAJy3Y8dP2Osj51rCiURrVa+UtWlY5pzXpdG0js3K28VNytn0RVvsy0krbNXGvS6Nceyfa93tV3WoZqWL5cvQ1wAAWY1gDAAAnLcvfl5u67bsswJ5c9q9VzawbNlOTUcMVPbs2VyhDwV3c1fscEHZotVx9vPM9TZp1np3n9IbK5cpmCGvAQCyGsEYAAA4LypP/+Xkle76PVfWt0L5M3ZulwK7xjWKu8vStbvsy8krbNaSbfbH/M3uojL5V3esZrXPUk4fAIINwRgAADhnx44n2Gsj57iTNrduUNpaNzj39MT0UBGQp29v6crnfzVppf0xf5PNXrrNXepULmrXdKxujWoUy5CROQDIbARjAADgnH0+Ybmt37rfCuXL5Yp2ZJVKpQvaYzc2tT7da9rXv6xyaYuL18TZM2umubRFBWUt65VKs3ojAAQLgjEAAHBOVqzfbaOTpSd6UXq+9AX5rP/VDe26zjXsmymrbdz0dbZm0157+ZNZVqZYPruqQ1Vr17icRefInuVtA4CzYcsEAAACdvRYgr2u9MREs7aNyrhiGl5Sufs7eta1Yf/s7AIznbts044DNuSLeXbXwIn2/e9r7PDR4562EQBSIxgDAAAB+2z8Mtuw7YAr1tHviqxLTzwbjc716VbThj/Z2W7tUdsK589lO/ccsve/WWh3vPizq8h48NAxr5sJAA5pigAAICDL1u2yMb+uctfvu6qBK2cfbPLERFvvi6tZj9aV3Xyyr35ZZdt3xdsnY5e6k0lfelElu7xNlQyv/AgAgSAYAwAA6XYkWXpi+yZlrWXdUhbMckZHWfcLK1mXFhXst3mb7MtJK23Dtv3u77dTVluXlhXsivZVrXjhPF43FUAEIhgDAADp9r+fltqmHQetSIFcdlevehYqoqKy28VNylm7RmVt5pKtLl1xxfo99sMfa+2nqetcYHnlxdWsXIn8XjcVQAQhGAMAAOmyZG2cffvbanf9vqsbWv48wZeeeDbZs2dzo3kt6pS0BSt32qhJK2zBqp02adYGmzx7g11Yr7Rd1bGaVS1byOumAogABGMAAOCsVInw9ZFzLTHRrGOzcta8dkkLZTopdIPqxdxleewul7Y4Y/FW+3PBZndpXKO4Xd2xmjuRNCeQBpBZCMYAAMBZ/XfsUtuy86AVLRhjd/QMnfTE9KhRoYg9eVsLi92yzxX3+G3uRpuzfLu71KpYxK7pVN2a1CxOUAYgw1HaHgAAnNGi1Tvtu9/XuOs6wXK+3NEWjiqUKmCP9Gli7z3Rybq3qmg5orLb0nW77LkPp9sDg3+13+dusgRVLgGADEIwBgAATuvwkeM25Iu57nrn5uWtaa0SFu5KFs1r917VwIY92dlVWsydK8rWbt5nr/xvtt0zaJKNnx5rx44neN1MAGGAYAwAAJzWiB+X2Na4eLugUG67/fK6FkmKFIix2y6rY8Oe7GI3dK1p+fNEu1TNN7+cZ3e+NNEVM1GwCgDnimAMAACkacGqHfbDn2vd9fuvaWh5wzQ98WxUNfL6LjVcUHb75XVcWf+4vYftw28X2W0v/Gxf/LzcDsQf9bqZAEIQBTwAAMAp4g8fsyFfzHPXu7asYI1qFLdIlztXDuvVrqpdelElVwZ/9ORVtiXuoP1v3DIb/csqu+TCitazbRUrXCDG66YCCBEEYwAA4BQf/7DEtu+Kt+KFc7tUPfy/6BxR1rVlRevUrLz9MX+zq8C4bss+F5Cp0Inm1vW+uJqVKJLH66YCCHIEYwAAIIV5K7bbT9PWuev3X9PI8sREZnri2URFZbd2jcta20ZlbNbSbTZq4gpbHrvbxk5dZ+Omx1q7RmXsqg7VrHzJAl43FUCQIhgDAAAp0hOHjkpKT1TanU6KjDPT+cd0EuxmtUrYotVxNmrSCpu3Yof98tdGd2lVr5QLyqqXL+x1UwEEGYIxAADgN/z7xbZj9yGXYndLD9ITAw3K6lW9wF1WbthtX05aadMWbvFfGlYrZld3qmb1qlzACaQBOARjAADAmbNsuzuHljxwbSNXsALnplq5wvaPW5rbhm373ZyyX+dstHkrd7hLjQqF7ZqO1d0527JnJygDIhml7QEAgB08dMzeGJV0cucerSu50R2cv3Il8ttD1ze295/o5KowRufI7uaVPT98ht3/n19ckJaQcMLrZgLwCMEYAACwYd8tsp17D1uponnt5ktqe92csKO0z7t717dh/+xsV15c1Y06xm7db//59C+7e9AkGzdtnR07nuB1MwFkMYIxAAAi3Oyl2+znmetN05geuK6RxZCemGl0DjLNxRv+VBfr272mFcib07bGxdtbX823O1782cb8usoOHTnudTMBZBG2tgAARLAD8UftjZPVEy9rU9nqVC7qdZMiQr7c0XZtpxrWs00VmzAz1sb8ssqNTKqAypeTVthlrStbjzaVLX+enF43FUAmIhgDACCCffDtItu177CVviCv3di9ltfNiTgahby8TRXr3qqS/frXBlfsY/POg/bZhOX29a+rrFuritarXRUrWjC3100FkAkIxgAAiFAzF2+1ybM3uPTEB69rbDE52S3wigp7dG5RwTo0K29TF2x2o2NrN++zb6asth/+WGudmpd3c81KFs3rdVMBZCC2ugAARKD98UftzS+T0hN7tatqtSoV8bpJMLOo7NmsTcMy1rpBaftr2XYXlC1Zu8sV+JgwfZ21aVjWrupYzSqWKuB1UwFkAIIxAAAi0PtjFtru/UesTLF81qdbTa+bg1R0Umidh0yXxWviXFCm4GzK3I3u0qJOSReU1axAEA2EMoIxAAAizLSFW9z5rXS+4Qevb2S5oqO8bhLOQEVV6lRuZas37rEvJ690aYwzFm91l/pVL7CrO1azBtWKuQAOQGghGAMAIILsPXDE3v5qvrt+RfuqjKyEkCplC9nfb2pmG7fvt9GTV9kvf22wBat2uku1coXs6o7V3YhZdkXZAEIC5xkDACDC0hP3HDhi5Urktxu6kp4YisoWz+/OB/f+Pzq50xHkjI6ylRv22Esfz7T+r/7iirIcTzjhdTMBpAPBGAAAEeLP+Zvtt3mb3MjJg9c1cjvxCF3FC+exu3rVs+FPdrZrOlW3vDE5bMO2/fba53Os38uTbOzUtXb0WILXzQRwBgRjAABEgD37j9jbo5PSE1UivXr5wl43CRmkYL5c7hxxw57sYjddUssK5stp23fF2zujF9jtL/5soyevtPjDx7xuJoBgDMZOnDhhQ4cOtTZt2ljDhg3tzjvvtA0bNpx2+XXr1tldd91lTZs2tbZt27rHHj9+PMUyn3zyiXXu3Nmtr3fv3jZlypQU92/cuNH69etnjRs3ttatW9vrr79uCQkcOQIAhK93v15g+w4etQol89v1XWp43Rxkgry5o928sQ//2dn6XVHPihXO7YLwj39cYre98LP9b9xSN2cQQPDwPBh7++237bPPPrPnn3/eRo4c6YKzO+64w44ePXrKsnv37rU+ffrYoUOHbMSIETZ48GD76aef7Omnn/Yv8/XXX9trr71mjzzyiH3//ffWrl07u++++2zZsmXu/mPHjtntt9/uruv5nn32Wfv888/trbfeysJXDQBA1vl93ib7c8HmpPTE6xtbdA7SE8OZTt7do3Vle/+JTi4dVacvOHjomH3x8wo3UvbBtwtt555DXjcTgNfBmAKu4cOH2/3332/t27e3mjVrukBq69atNmHChFOWHzNmjMXHx9uQIUOsTp06bnTshRdesNGjR7vRLpk4caIb7erWrZuVK1fOHnjgAcuTJ49NmzbN3T9+/HjbvHmzvfLKK1a9enXr1KmTPfzwwy64SysABAAglO3ef9ilq8k1Hatb1bKFvG4SskiOqOzWsVl5e+tvHezvNzezqmUL2pGjCfbdb2vszpd+tjdGzbPNOw543UwgonkajGm06uDBg9aqVSv/bQUKFLDatWvbrFmzTlk+NjbWKleubEWK/H8ZXi0rs2fPdn+LFi3qHqt1JyYm2tixY23//v1Wr149/3IK5AoWLOhfR8uWLe3AgQO2dOnSTH29AABkJf0OKhDbH3/UKpUu4Io8IPJEZc9mF9UvbYMfbGfP3dXK6lYpascTEm3CjFi7Z9Ake+W/s23t5r1eNxOISJ6eZ0wjYFKqVKkUtxcvXtx/X+rbt2/f7uZ3RUUlpVhs2rTJ/Y2Li3N/BwwYYKtWrbKePXu6ZZT2qFREjaL5nrNkyZKnrFe2bNliDRo0yJTXCgBAVpsyd5M7wbN2xh+8TumJns9OgId0UujGNYq7y9K1u+zLySts1pJtLo1Vl6a1SrjR01qVOPccEBHBmOZ+Sc6cOVPcnitXLjc/LLXu3bu7OWYDBw50qYVKWVSaYo4cOdxcMFm/fr0LwJSGWK1aNZfu+OKLL1qZMmVckZDDhw+70bfUzydHjhw5r6OPao/XfO+p7y9wNvQZBIo+Exp27z9i736dVD2xd/tKVrJwtGe/U/SZ4FOhRIw9en19W7dlv337+1qbtmibzV6adKlVsZD1alvJGlQt6gI4L9BnEMp9RnFBer87ngZjMTEx7q/mavmu+4Ki3Llzn7J8xYoV3XwxFez49NNP3Vww30hY/vz53Y+MinU88cQTbmTMl8ao0bNXX33VBWN6ntRzw3xBmNZ3rhQMBlOao6pOAoGgzyBQ9JngpR2Bz3+Ls4OHjrsgrGbxI0HxG0WfCU5d6kVbkwol7M+lB2ze2oO2dN0eW7purpUqHG1t6uS3muVyW3aPgjL6DEK1z6QebArKYMyXnqjUw/Lly/tv1/9r1Ei77G6HDh3cRcsUKlTIlbV/+eWXXbGO1atX2549e/zzw3xU4v7nn39215WiuGLFihT3a11SokSJc34t0dHRVrVqVfOajgaoEypwTSugBVKjzyBQ9Jng99u8zbZi0yaLispmj9zQxMqXzO9pe+gzoaF1C7O4vYftx6mxNnHWRtuy+5iN+mOXlb4gj/VsU8laNyjpioJkBfoMQrnPaKAovTwNxlQ9MV++fDZjxgx/MLZv3z5bsmSJ9e3b95TlVXxDI2MfffSRf56XCnToDdc5w3zDksuXL7cqVar4H6f/64ORZs2a2TfffOMKdui5Zfr06ZY3b17XnnOlocjzGVnLaHpPgqk9CH70GQSKPhOc4vYeso/HJh101PnEalY+9wONGY0+E/z0+dx9ZRG7vmtt++GPtfb9H2ts8854e2fMYvvq1zXWu31V69yiguWKzprTI9BnEIp9JpD03uxeD98p6FIK4aRJk1wFxIceesiNXnXp0sUV6tixY4eb5yWqpKjAatCgQe7E0CpjrzljOoGzAqtixYpZjx497KWXXnLr0zI6AbRK3999991uHSplr+UefPBB93xah85Xdtttt6V7OBEAgGBNT3zzy/nunFIqY37VxdW8bhJCVMF8uaxPt5o2/MnOdmuP2lYofy7bsfuQvTdmod3xws/25aQVrp8BOD+ejoyJzjGmVMMnn3zSBV0auRo2bJhL+9O5wzp27OgKdvTu3duVtH/33XddWqKCLgVV/fv3t1tuucW/PhXreOedd9wyO3futEqVKrlgq2vXrv5iHR9++KE999xzds0117gS9zfccIPde++9Hr4LAACcv0mzNrgCDEol08mdo7IopQzhK09MtPW+uJo7ifTEWett9C+rbPuuePtk7FL7avJKu/SiSnZ5myouWAMQuGyJOoyG87Jw4UL3N/VcNS+oiIkmadeqVcvzIVqEBvoMAkWfCU479xyy+/492eIPH7ebL61tV3UInlEx+kz4SEg4Yb/N22RfTlppG7btd7fljI6yLi3K2xXtq1rxwhnz+dJnEMp9JpDYwPORMQAAcH50XPWNUfNcIFa9fCG7ot3/z5sGMpJGWy9uUs7aNSprMxZvdemKKzfscfPLfpq6zt13ZYeqVra4t0VjgFBBMAYAQIibMGO9zVm+3Z3UWSd3Jj0RmS179mzWql4pa1m3pC1Y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KOFJ2wAAkYdgDAAQMfLkyWPt2rVLMxgbO3asde3a1XLkyOG/7ciRI/bWW29Zt27drF69etalSxd7//337cSJEykeO3LkSPfY+vXrW9++fW3z5s2nrF+3Pfzww9a8eXNr0KCB3XzzzbZkyZLTtlUnQH722Wetbdu2VrduXdeGYcOGnfd7AAAIHgRjAICIcskll5ySqnjgwAH77bffrEePHv7bEhMT7e6777YPP/zQrr76anv33XddQPT666/bM88841/uf//7n/u/gry3337bBVpPPfVUiufctWuXm6u2ePFid99//vMfF9D16dPHVq9enWY7X3rpJdemxx9/3AVhHTt2tFdeecVGjx6dKe8LACDr/f/hPwAAIkD79u1dOqJGx2655RZ3288//2xFixa1Jk2a+JdTIDR16lQbPHiwXXrppe62iy66yGJiYmzIkCF20003WdWqVV0ApgDvH//4h1umdevWLrjTaJnPiBEjbM+ePfb5559bmTJl3G0a8dLjtK6hQ4ee0s6ZM2e65/M9d4sWLdzIntoJAAgPjIwBACKKgqkOHTqkSFX88ccfrXv37pYtW7YUwZBSFjUaltzll1/uv3/NmjUWFxdnF198cYpltK7kpk2bZrVq1XLz0Y4fP+4uKiaigEwBX1oUfI0aNcruvPNON/q2YcMGu++++1wwCQAID4yMAQAijoKl/v37u1RFFfRQsPTggw+mWGbv3r1WuHBhi4qKSnF7sWLF3N/9+/e7ZUTLpbWMj0bFYmNjrU6dOmm259ChQ6fc9s9//tNKlixp3333nT3//PPu0qhRIzePrGbNmuf4ygEAwYRgDAAQcTQilTdvXjc6ptS/smXLuiIZyRUsWNB2795tCQkJKQKy7du3+wMwXxCm0bHUwVdy+fPnd4U7/va3v6XZnpw5c6Z52z333OMuKv6hao9KiXzkkUfcSB4AIPSRpggAiDgKdDp16mTjx4+3n376yT8vKzkFT0onTF15USNVovllFStWtFKlSp2yTOrzlWlda9eutUqVKrmqjL7Lt99+a1999dUpo2+qpKjqjMOHD3f/L126tCv2oXamVakRABCaGBkDAEQkFc/o16+fm7v15JNPpjl6pnlbum/btm0uNVDzxD744AO74oorXPEOefTRR91olZbT/DJValShjuRUKESBl/7edtttbkRNpfQ1J+yJJ55Ic16bUhrffPNNi46Otho1arhgbsyYMS5IAwCEB4IxAEBEuvDCC61AgQJuZKtKlSqn3K9iHu+9956rdPjxxx+78vRKZ9S5wm699Vb/ciqHr4BOKYQKuKpXr27/+te/3HI+Ktyh6ooqaa85Xzp/mUbVXnzxRbvqqqvSbJ/WoTL6Gh3bsWOHq6KoZR944IFMekcAAFktW6JOpAIAAAAAyFLMGQMAAAAADxCMAQAAAIAHCMYAAAAAwAMEYwAAAADgAYIxAAAAAPAAwRgAAAAAeIBgDAAAAAA8QDAGAAAAAB4gGAMAAAAADxCMAQAAAIAHCMYAAAAAwAMEYwAAAABgWe//AJrMqsaq+1vxAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#compare all accuracy graphically for these accuracies 0.967741935483871, 0.967741935483871, 0.967741935483871, 0.9838709677419355, 0.9787234042553191, 0.9516129032258065 use line plot\n",
    "sns.set(style=\"whitegrid\")\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(['GaussianNB', 'RandomForest', 'DecisionTree', 'LogisticRegression', 'SVM', 'KNN'], [0.967741935483871, 0.967741935483871, 0.967741935483871, 0.9838709677419355, 0.9787234042553191, 0.9516129032258065], marker='o')\n",
    "plt.title('Accuracy of different models')\n",
    "plt.xlabel('Models')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
