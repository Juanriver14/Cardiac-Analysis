
README
#Description: This project was created as a way to use machine learning methods to analyze the SPECTF heart data located at http://archive.ics.uci.edu/ml/datasets/SPECTF+Heart


import numpy as np    
import pandas as pd

For optimization We’ll use the SciPy optimize package to find the optimal values
of Lagrange multipliers, and compute the soft margin and the separating hyperplane.

from scipy.optimize import Bounds, BFGS                     
from scipy.optimize import LinearConstraint, minimize   
# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
# For generating dataset
import sklearn.datasets as dt

url = 'https://raw.githubusercontent.com/Juanriver14/Datasets/main/SPECTF_test.csv'
df = pd.read_csv(url)


#We also need the following constant to detect all alphas numerically close to zero, so we need to define our own threshold for zero.
ZERO = 1e-7

df.head
Out[ ]:
<bound method NDFrame.head of      1   67   68   73   78   65   63   67.1   60   63.1  ...   61.2   56.1  \
0    1   75   74   71   71   62   58     70   64     71  ...     66     62   
1    1   83   64   66   67   67   74     74   72     64  ...     67     64   
2    1   72   66   65   65   64   61     71   78     73  ...     69     68   
3    1   62   60   69   61   63   63     70   68     70  ...     66     66   
4    1   68   63   67   67   65   72     74   72     70  ...     70     70   
..  ..  ...  ...  ...  ...  ...  ...    ...  ...    ...  ...    ...    ...   
181  0   74   69   75   70   70   74     77   77     65  ...     66     67   
182  0   72   61   64   66   64   59     68   66     76  ...     69     64   
183  0   75   73   72   77   68   67     76   73     67  ...     70     67   
184  0   59   62   72   74   66   66     74   76     63  ...     65     71   
185  0   64   66   68   71   62   64     74   73     63  ...     70     69   

      76.3   75.1   74.1   77   76.4   74.2   59.1   68.3  
0       68     69     69   66     64     58     57     52  
1       69     63     68   54     65     64     43     42  
2       68     63     71   72     65     63     58     60  
3       58     56     72   73     71     64     49     42  
4       70     67     77   71     77     72     68     59  
..     ...    ...    ...  ...    ...    ...    ...    ...  
181     63     61     71   68     66     65     54     57  
182     67     71     69   68     65     73     56     52  
183     72     71     79   75     77     75     67     71  
184     67     69     77   78     77     76     70     70  
185     68     65     75   72     62     64     57     54  

[186 rows x 45 columns]>
In [ ]:
dft=np.array(df)
In [ ]:
dft
Out[ ]:
array([[ 1, 75, 74, ..., 58, 57, 52],
       [ 1, 83, 64, ..., 64, 43, 42],
       [ 1, 72, 66, ..., 63, 58, 60],
       ...,
       [ 0, 75, 73, ..., 75, 67, 71],
       [ 0, 59, 62, ..., 76, 70, 70],
       [ 0, 64, 66, ..., 64, 57, 54]])
In [ ]:
dat = df.iloc[:,1:].values
In [ ]:
dat
Out[ ]:
array([[75, 74, 71, ..., 58, 57, 52],
       [83, 64, 66, ..., 64, 43, 42],
       [72, 66, 65, ..., 63, 58, 60],
       ...,
       [75, 73, 72, ..., 75, 67, 71],
       [59, 62, 72, ..., 76, 70, 70],
       [64, 66, 68, ..., 64, 57, 54]])
In [ ]:
labels = df.iloc[:,0].values
In [ ]:
labels
Out[ ]:
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

Defining the Data Points and Labels
if a string of alphas is given to the plotting function, then it will also label all support vectors with their corresponding alpha values. Just to recall support vectors are those points for which .
In [ ]:
def plot_x(x, t, alpha=[], C=0):
    sns.scatterplot(dat[:,0], dat[:, 1], style=labels,
    hue=labels, markers=['s', 'P'],
    palette=['magenta', 'green'])
    if len(alpha) > 0:
        alpha_str = np.char.mod('%.1f', np.round(alpha, 1))
        ind_sv = np.where(alpha > ZERO)[0]
        for i in ind_sv:   
            plt.gca().text(dat[i,0], dat[i, 1]-.25, alpha_str[i] )
 
                  
plot_x(dat, labels)

 


Defining the Objective Function
Our objective function is Ld defined above, which has to be maximized. As we are using the minimize() function, we have to multiply by Ld (-1) to maximize it. Its implementation is given below. The first parameter for the objective function is the variable w.r.t. which the optimization takes place. We also need the training points and the corresponding labels as additional arguments.
In [ ]:
# Objective function
def lagrange_dual(alpha, x, t):
    result = 0
    ind_sv = np.where(alpha > ZERO)[0]
    for i in ind_sv:
        for k in ind_sv:
            result = result + alpha[i]*alpha[k]*t[i]*t[k]*np.dot(x[i, :], x[k, :]) 
    result = 0.5*result - sum(alpha)     
    return result
Defining the Bounds The bounds on alpha are defined using the Bounds() method. All alphas are constrained to lie between 0 and C . Here is an example for C=10 .
In [ ]:
linear_constraint = LinearConstraint(labels, [0], [0])
print(linear_constraint)
<scipy.optimize._constraints.LinearConstraint object at 0x7fb95b4d08d0>
In [ ]:
bounds_alpha = Bounds(np.zeros(dat.shape[0]), np.full(dat.shape[0], 10))
print(bounds_alpha)
Bounds(array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))
Defining the Function to Find Alphas
find the optimal values of alpha when given the parameters x, t, and C. The objective function requires the additional arguments x and t, which are passed via args in minimize().

def optimize_alpha(x, t, C):
    m, n = x.shape
    np.random.seed(1)
    # Initialize alphas to random values
    alpha_0 = np.random.rand(m)*C
    # Define the constraint
    linear_constraint = LinearConstraint(t, [0], [0])
    # Define the bounds
    bounds_alpha = Bounds(np.zeros(m), np.full(m, C))
    # Find the optimal value of alpha
    result = minimize(lagrange_dual, alpha_0, args = (x, t), method='trust-constr', 
                      hess=BFGS(), constraints=[linear_constraint],
                      bounds=bounds_alpha)
    # The optimized value of alpha lies in result.x
    alpha = result.x
    return alpha
Determining the Hyperplane
The expression for the hyperplane is given by: W t X + W0 = 0
def get_w(alpha, t, x):
    m = len(x)
    # Get all support vectors
    w = np.zeros(x.shape[1])
    for i in range(m):
        w = w + alpha[i]*t[i]*x[i, :]        
    return w
 
def get_w0(alpha, t, x, w, C):
    C_numeric = C-ZERO
    # Indices of support vectors with alpha<C
    ind_sv = np.where((alpha > ZERO)&(alpha < C_numeric))[0]
    w0 = 0.0
    for s in ind_sv:
        w0 = w0 + t[s] - np.dot(x[s, :], w)
    # Take the average    
    w0 = w0 / len(ind_sv)
    return w0
In [ ]:
#Classifying Test Points
def classify_points(x_test, w, w0):
    # get y(x_test)
    predicted_labels = np.sum(x_test*w, axis=1) + w0
    predicted_labels = np.sign(predicted_labels)
    # Assign a label arbitrarily a +1 if it is zero
    predicted_labels[predicted_labels==0] = 1
    return predicted_labels
 
def misclassification_rate(labels, predictions):
    total = len(labels)
    errors = sum(labels != predictions)
    return errors/total*100
In [ ]:
#Plotting the Margin and Hyperplane

def plot_hyperplane(w, w0):
    x_coord = np.array(plt.gca().get_xlim())
    y_coord = -w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, y_coord, color='red')
 
def plot_margin(w, w0):
    x_coord = np.array(plt.gca().get_xlim())
    ypos_coord = 1/w[1] - w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, ypos_coord, '--', color='green') 
    yneg_coord = -1/w[1] - w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, yneg_coord, '--', color='magenta')

Powering Up The SVM
It’s now time to run the SVM. The function display_SVM_result() will help us visualize everything. We’ll initialize alpha to random values, define C and find the best values of alpha in this function. We’ll also plot the hyperplane, the margin and the data points. The support vectors would also be labelled by their corresponding alpha value. The title of the plot would be the percentage of errors and number of support vectors.
In [ ]:
def display_SVM_result(x, t, C):
    # Get the alphas
    alpha = optimize_alpha(x, t, C)   
    # Get the weights
    w = get_w(alpha, t, x)
    w0 = get_w0(alpha, t, x, w, C)
    plot_x(x, t, alpha, C)
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plot_hyperplane(w, w0)
    plot_margin(w, w0)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # Get the misclassification error and display it as title
    predictions = classify_points(x, w, w0)
    err = misclassification_rate(t, predictions)
    title = 'C = ' + str(C) + ',  Errors: ' + '{:.1f}'.format(err) + '%'
    title = title + ',  total SV = ' + str(len(alpha[alpha > ZERO]))
    plt.title(title)
    
display_SVM_result(dat, labels, 10)    
plt.show()

 

The Effect of C
If you change the value of C to infinite, then the soft margin turns into a hard margin, with no toleration for errors. The problem we defined above is not solvable in this case. Let’s generate an artificial set of points and look at the effect of C on classification. To understand the entire problem, we’ll use a simple dataset, where the positive and negative examples are separable.
In [ ]:
dat, labels = dt.make_blobs(n_samples=[40,40],
                           cluster_std=1,
                           random_state=0)
labels[labels==0] = -1
plot_x(dat, labels)


 
Now let’s define different values of C
fig = plt.figure(figsize=(8,25))
 
i=0
C_array = [1e-2, 900, 1e5]
 
for C in C_array:
    fig.add_subplot(311+i)    
    display_SVM_result(dat, labels, C)  
    i = i + 1


 

he above is a nice example, which shows that increasing , decreases the margin. A high value of adds a stricter penalty on errors. A smaller value allows a wider margin and more misclassification errors. Hence, defines a tradeoff between the maximization of margin and classification errors
In [22]:
fig = plt.figure(figsize=(8,25))
 
i=0
C_array = [1e-2, 100, 1e5]
 
for C in C_array:
    fig.add_subplot(311+i)    
    display_SVM_result(dat, labels, C)  
    i = i + 1


 

Second Testing using Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
https://raw.githubusercontent.com/Juanriver14/Datasets/main/SPECTF_train.csv
https://raw.githubusercontent.com/Juanriver14/Datasets/main/SPECTF_test.csv
In [ ]:
url = 'https://raw.githubusercontent.com/Juanriver14/Datasets/main/SPECTF_train.csv'
df = pd.read_csv(url)
In [ ]:
df.head
Out[ ]:
<bound method NDFrame.head of     1  59  52  70  67  73  66  72  61  58  ...  66.3  56.1  62  56.2  72.3  \
0   1  72  62  69  67  78  82  74  65  69  ...    65    71  63    60    69   
1   1  71  62  70  64  67  64  79  65  70  ...    73    70  66    65    64   
2   1  69  71  70  78  61  63  67  65  59  ...    61    61  66    65    72   
3   1  70  66  61  66  61  58  69  69  72  ...    67    69  70    66    70   
4   1  57  69  68  75  69  74  73  71  57  ...    63    58  69    67    79   
.. ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ...   ...   ...  ..   ...   ...   
74  0  70  75  72  72  67  71  71  78  63  ...    66    67  68    70    70   
75  0  59  57  67  71  66  68  68  70  56  ...    62    64  56    53    71   
76  0  67  64  73  75  77  77  74  70  65  ...    61    64  65    60    68   
77  0  68  65  72  72  47  74  76  74  67  ...    64    69  71    73    73   
78  0  66  54  69  66  69  69  75  72  63  ...    69    65  65    64    67   

    62.1  74.2  74.3  64.1  67.4  
0     73    67    71    56    58  
1     55    61    41    51    46  
2     73    68    68    59    63  
3     64    60    55    49    41  
4     77    72    70    61    65  
..   ...   ...   ...   ...   ...  
74    71    64    67    56    54  
75    68    64    63    56    56  
76    75    74    80    67    68  
77    75    68    56    58    44  
78    69    71    68    59    59  

[79 rows x 45 columns]>
In [ ]:
df.shape
Out[ ]:
(79, 45)

# Drop 1
target=df[['1']]
df=df.drop(labels='1',axis=1)
In [ ]:
column_head=[(lambda x,y: "F"+str(x)+y) (x,y) for x in range(1,23) for y in ['R','S']]
In [ ]:
df.columns=column_head
In [ ]:
url = 'https://raw.githubusercontent.com/Juanriver14/Datasets/main/SPECTF_test.csv'
dft = pd.read_csv(url)
In [ ]:
# Get a feel for test set
dft.head()
Out[ ]:
	1	67	68	73	78	65	63	67.1	60	63.1	...	61.2	56.1	76.3	75.1	74.1	77	76.4	74.2	59.1	68.3
0	1	75	74	71	71	62	58	70	64	71	...	66	62	68	69	69	66	64	58	57	52
1	1	83	64	66	67	67	74	74	72	64	...	67	64	69	63	68	54	65	64	43	42
2	1	72	66	65	65	64	61	71	78	73	...	69	68	68	63	71	72	65	63	58	60
3	1	62	60	69	61	63	63	70	68	70	...	66	66	58	56	72	73	71	64	49	42
4	1	68	63	67	67	65	72	74	72	70	...	70	70	70	67	77	71	77	72	68	59
dft.shape
Out[ ]:
(186, 45)
In [ ]:
# Use dropped 
test_target=dft[['1']]
dft=dft.drop(labels='1',axis=1)
column_head=[(lambda x,y: "F"+str(x)+y) (x,y) for x in range(1,23) for y in ['R','S']]
dft.columns=column_head
In [ ]:
dft.head(2)
Out[ ]:
	F1R	F1S	F2R	F2S	F3R	F3S	F4R	F4S	F5R	F5S	...	F18R	F18S	F19R	F19S	F20R	F20S	F21R	F21S	F22R	F22S
0	75	74	71	71	62	58	70	64	71	68	...	66	62	68	69	69	66	64	58	57	52
1	83	64	66	67	67	74	74	72	64	68	...	67	64	69	63	68	54	65	64	43	42
2 rows × 44 columns
In [ ]:
df.head(1)
Out[ ]:
	F1R	F1S	F2R	F2S	F3R	F3S	F4R	F4S	F5R	F5S	...	F18R	F18S	F19R	F19S	F20R	F20S	F21R	F21S	F22R	F22S
0	72	62	69	67	78	82	74	65	69	63	...	65	71	63	60	69	73	67	71	56	58
1 rows × 44 columns
from sklearn.decomposition import PCA
pca = PCA().fit(df)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()
 
In [ ]:
pc=PCA(n_components=34,svd_solver='randomized').fit(df)
In [ ]:
x_train=pca.transform(df)
In [ ]:
x_train
Out[ ]:
array([[-2.25153691e+01,  4.43400602e+00,  1.09025298e+01, ...,
         2.36547079e-01, -1.90600607e-01, -1.54656216e-01],
       [ 2.40190220e+01, -3.16216506e+01,  1.86551363e+01, ...,
        -8.22678816e-01, -1.57392166e+00, -1.01786125e+00],
       [-2.19734319e+00,  1.43263594e+01, -1.25115570e+01, ...,
         7.13284185e-01, -1.66991206e+00, -1.20327601e+00],
       ...,
       [-2.00323653e+01,  1.40177059e+01, -9.34358705e+00, ...,
        -7.95600589e-01,  5.07720674e-01,  2.11010169e-02],
       [ 6.43860906e+00, -1.74391901e+01, -6.30872656e+00, ...,
        -4.38786023e-01, -7.58102783e-01,  3.83601119e-01],
       [-1.04968178e+01,  8.34998664e+00,  3.96402167e+00, ...,
         7.85465149e-01,  2.21166304e-01, -4.88880228e-01]])
In [ ]:
x_test=pca.transform(dft)
x_test
# Transform Test with PCA
Out[ ]:
array([[ -1.17530963, -10.80210189,   1.69345964, ...,   0.6992736 ,
          1.08949346,  -0.57320507],
       [  8.4969187 ,  -7.7891098 ,  -1.81119436, ...,  -4.35426022,
          3.98596719,   0.70649879],
       [ -7.50081378,  -5.72300766,  16.19286928, ...,   3.68446802,
         -1.84276515,   0.83848757],
       ...,
       [-30.19021928,  -0.3435545 ,  -8.16100715, ...,  -1.46218476,
          0.79210118,   3.20569344],
       [-26.59534356,  10.57124998,  -7.1436282 , ...,  -2.58869222,
         -1.93432059,  -0.07608877],
       [-11.28520906,  -8.83121585,   9.72864556, ...,  -3.21410934,
          1.31835611,  -0.17280859]])
Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive=GaussianNB().fit(x_train,target)
naive_prediction=naive.predict(x_test)
accuracy_score(test_target,naive_prediction)

0.45698924731182794


from sklearn.metrics import confusion_matrix,plot_confusion_matrix
import matplotlib.pyplot as plt
#confusion_matrix(test_target, pred_val)
plot_confusion_matrix(naive,x_test,test_target)
#ConfusionMatrixDisplay.from_predictions(test_target, naive_prediction)

 

# See where everything falls (which class) as well as accuracy
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(test_target, naive_prediction, target_names=target_names))
              precision    recall  f1-score   support

     class 0       0.13      1.00      0.23        15
     class 1       1.00      0.41      0.58       171

    accuracy                           0.46       186
   macro avg       0.56      0.70      0.40       186
weighted avg       0.93      0.46      0.55       186

Support Vector Machines
In [ ]:
from sklearn import svm
Kernel: Linear Kernel ; C=10
In [ ]:
# Accuracy check
svc=svm.SVC(kernel='linear',C=10)
svc.fit(x_train,target)
pred_val=svc.predict(x_test)
accuracy_score(test_target,pred_val)

0.7258064516129032

from sklearn.metrics import confusion_matrix
#confusion_matrix(test_target, pred_val)
plot_confusion_matrix(svc,x_test,test_target)


 

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(test_target, pred_val, target_names=target_names))
# see class and accuracy score
              precision    recall  f1-score   support

     class 0       0.14      0.47      0.22        15
     class 1       0.94      0.75      0.83       171

    accuracy                           0.73       186
   macro avg       0.54      0.61      0.52       186
weighted avg       0.88      0.73      0.78       186

Kernel: RBF ; C=940 ; Gamma = 0.0034
In [ ]:
# Accuracy score has increased here 
svc=svm.SVC(kernel='rbf',gamma=0.0034,C=940)
svc.fit(df,target)
pred_val=svc.predict(dft)
accuracy_score(test_target,pred_val)

0.8118279569892473


SVM(2) 81%
In [ ]:
from sklearn.metrics import confusion_matrix
confusion_matrix(test_target, pred_val)
plot_confusion_matrix(svc,x_test,test_target)


 


from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(test_target, pred_val, target_names=target_names))
              precision    recall  f1-score   support

     class 0       0.24      0.67      0.36        15
     class 1       0.97      0.82      0.89       171

    accuracy                           0.81       186
   macro avg       0.60      0.74      0.62       186
weighted avg       0.91      0.81      0.84       186

Kernel: Gaussian Kernel ; C=940 ; Gamma = 0.004
In [ ]:
from sklearn.gaussian_process.kernels import RBF
gsvc=svm.SVC(kernel=RBF(),C=940,gamma=0.004).fit(x_train,target)
predict_gsvc=gsvc.predict(x_test)
accuracy_score(test_target,predict_gsvc)
0.08064516129032258
SVM (3) 0.08%
In [ ]:
from sklearn.metrics import confusion_matrix
#confusion_matrix(test_target, predict_gsvc)
plot_confusion_matrix(gsvc,x_test,test_target)
#ConfusionMatrixDisplay.from_predictions(test_target, predict_gsvc)

 
In [ ]:
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(test_target, predict_gsvc, target_names=target_names))
              precision    recall  f1-score   support

     class 0       0.08      1.00      0.15        15
     class 1       0.00      0.00      0.00       171

    accuracy                           0.08       186
   macro avg       0.04      0.50      0.07       186
weighted avg       0.01      0.08      0.01       186




