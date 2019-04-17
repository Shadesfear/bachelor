import bohrium as np
import pandas as pd
from bohrium_kmeans import *
# from sklearn.cluster import KMeans

data_path = "../data/heart.csv"

"""
1. #3 (age)
2. #4 (sex)
3. #9 (cp)
4. #10 (trestbps)
5. #12 (chol)
6. #16 (fbs)
7. #19 (restecg)
8. #32 (thalach)
9. #38 (exang)
10. #40 (oldpeak)
11. #41 (slope)
12. #44 (ca)
13. #51 (thal)
14. #58 (num) (the predicted attribute)
"""

data_frame = pd.read_csv(data_path)
data_frame.convert_objects(convert_numeric=True)
data_frame.fillna(0, inplace=True)

def handle_non_numerics(data_frame):
    """
    Removes all non numeric data from the data_frame

    Example:
    0 Sex
    1 Male
    2 Male
    3 Female

    Now becomes
    0 Sex
    1 0
    2 0
    3 1

    """

    columns = data_frame.columns.values

    for column in columns:
        text_digit_vals = {}

        def conv_to_int(val):
            return text_digit_vals[val]

        if data_frame[column].dtype != np.int64 and data_frame[column].dtype != np.float64:

            column_contents = data_frame[column].values.tolist()
            unique_elements = set(column_contents)

            x = 0

            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            data_frame[column] = list(map(conv_to_int, data_frame[column]))

    return data_frame

data_frame = handle_non_numerics(data_frame)

X = np.array(data_frame.drop(['target'], 1).astype(float))
Y = np.array(data_frame['target'])


# data_frame.drop(['suicides_no', 'suicides/100k pop'], 1, inplace=True)

# clf = KMeans(n_clusters=2)
# clf.fit(X)

kmeans = bohrium_kmeans(k = 2, userkernel=False)
# X = kmeans.scale_data(X)
closest, centroids, iterations = kmeans.run(X)



correct = 0
for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    # prediction = clf.predict(predict_me)
    prediction, min_dist = kmeans.centroids_closest(predict_me, centroids, 'squared')

    if prediction[0] == Y[i]:
        correct += 1

print(correct/len(X))



# print(data_frame)
