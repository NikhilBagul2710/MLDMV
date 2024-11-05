.. code:: ipython3

    #PCA dimensionality reduction
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Load your wine dataset
    df = pd.read_csv('wine.csv')
    
    # Separating features and target variable
    X = df.drop(columns=['Customer_Segment'])  # Features (measurements)
    y = df['Customer_Segment']  # Target variable (Type of wine)
    
    # Scatter plot before applying PCA
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 1]['Alcohol'], X[y == 1]['Malic_Acid'], label='Customer Segment 1', alpha=0.7)
    plt.scatter(X[y == 2]['Alcohol'], X[y == 2]['Malic_Acid'], label='Customer Segment 2', alpha=0.7)
    plt.scatter(X[y == 3]['Alcohol'], X[y == 3]['Malic_Acid'], label='Customer Segment 3', alpha=0.7)
    plt.xlabel('Alcohol')
    plt.ylabel('Malic Acid')
    plt.legend()
    plt.title('Scatter Plot (Original Data)')
    plt.show()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    
    # Create a new DataFrame with the first two principal components
    pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['Customer_Segment'] = y
    
    
    # Visualize the data using the first two principal components
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df[pca_df['Customer_Segment'] == 1]['Principal Component 1'], pca_df[pca_df['Customer_Segment'] == 1]['Principal Component 2'], label='Customer Segment 1', alpha=0.7)
    plt.scatter(pca_df[pca_df['Customer_Segment'] == 2]['Principal Component 1'], pca_df[pca_df['Customer_Segment'] == 2]['Principal Component 2'], label='Customer Segment 2', alpha=0.7)
    plt.scatter(pca_df[pca_df['Customer_Segment'] == 3]['Principal Component 1'], pca_df[pca_df['Customer_Segment'] == 3]['Principal Component 2'], label='Customer Segment 3', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA: Wine Dataset')
    plt.show()
    
    '''1. Standardizing the Data
    
    2. Calculating the Covariance Matrix
    High covariance b/w features indicates they vary together, while low covariance indicates they vary independently.
    
    3. Computing Eigenvalues and Eigenvectors
    Eigenvectors represent the directions of maximum variance (i.e., principal components).
    Eigenvalues represent the magnitude of variance along each eigenvector.
    
    4. Selecting Principal Components Based on Variance (Eigenvalues)
    PCA sorts the eigenvalues in descending order. This ordering prioritizes the directions (components) that capture the most variance.'''
    
    



.. code:: ipython3

    #bivariate nd multiplr regression analysis on UCI and PIMA indian diabetic dataset
    import pandas as pd
    import numpy as np
    from scipy import stats as st
    
    
    data =pd.read_csv(r"diabetes.csv")
    
    
    data.head()
    
    
    data.shape
    
    
    print(data['Glucose'].value_counts())
    
    
    data.describe()
    
    
    st.mode(data)
    
    
    st.skew(data)
    
    
    st.kurtosis(data)
    
    
    np.var(data)
    
    
    
    from sklearn.linear_model import LinearRegression 
    from sklearn.model_selection import train_test_split
    x = data.drop('Outcome',axis=1)
    y= data['Outcome']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
    lr = LinearRegression()
    temp = lr.fit(x_train,y_train)
    
    
    lr.score(x_test,y_test)
    
    
    from sklearn.linear_model import LogisticRegression 
    from sklearn.model_selection import train_test_split
    x = data.drop('Outcome',axis=1)
    y= data['Outcome']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
    lor = LogisticRegression()
    temp = lor.fit(x_train,y_train)
    
    
    lor.score(x_test,y_test)
    
    
    
    from sklearn.metrics import classification_report
    preds = lor.predict(x_test)
    print(classification_report(y_test,preds))
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    plt.bar(0,y.value_counts()[0],label="NO")
    plt.bar(1,y.value_counts()[1],label="YES")
    plt.legend()
    plt.xticks([0,1]);
    
    
    
    fig, axes = plt.subplots(figsize=(15,10))
    sns.countplot(x='BloodPressure', data=data, hue='Outcome',ax=axes)
    
    
    
    
    





.. code:: ipython3

    #svm for handwritten img classification 
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn import metrics
    
    
    # Load the digits dataset
    digits = datasets.load_digits()
    
    # Split the data into features (X) and labels (y)
    X = digits.data
    y = digits.target
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    # Create an SVM classifier (linear kernel)
    clf = svm.SVC(kernel='linear')
    
    
    # Fit the classifier on the training data
    clf.fit(X_train, y_train)
    
    
    # Predict on the test data
    y_pred = clf.predict(X_test)
    
    
    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy : ", accuracy)
    
    
    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix : ")
    print(confusion_matrix)
    
    
    # Classification report
    classification_report = metrics.classification_report(y_test, y_pred)
    print("Classification Report : ")
    print(classification_report)
    
    
    # Visualize some of the test images and their predicted labels
    plt.figure(figsize=(15, 8))
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_test[i].reshape(8, 8), cmap=plt.cm.gray_r)
        plt.title(f"Predicted : {y_pred[i]}, Actual : {y_test[i]}")
        plt.axis('on')
    
    
    








.. code:: ipython3

    #uber Linear , ridge , lasso
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.impute import SimpleImputer
    
    # Load the dataset
    df = pd.read_csv("uber.csv")
    
    # view dataset
    df.head()
    
    
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    # print(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    # print(df['hour'])
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    # print(df['day_of_week'])
    
    
    # check datasets for more columns we added 'hour' and 'day_of_week' column
    df.head()
    
    
    # Drop unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'key', 'pickup_datetime'])
    
    
    # check datasets for removal of columns we removed 'first_column with no name', 'key' and 'pickup_datetime' column
    df.head()
    
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    
    # Split the data into features (X) and target (y)
    X = df_imputed.drop(columns=['fare_amount'])  # create new dataset ignoring 'fare_amount' column
    y = df_imputed['fare_amount']  # create a series of only 'fare_amount' column
    
    
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    # Standardize the features (scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    # Implement Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    
    
    # Implement Ridge Regression
    ridge_model = Ridge(alpha=1.0)  # You can experiment with different alpha values
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    
    
    # Implement Lasso Regression
    lasso_model = Lasso(alpha=0.1)  # You can experiment with different alpha values
    lasso_model.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_model.predict(X_test_scaled)
    
    
    
    # Evaluate the models
    def evaluate_model(y_true, y_pred, model_name):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{model_name} - R2 Score: {r2:.4f}, RMSE: {rmse:.2f}")
    
    
    
    evaluate_model(y_test, y_pred_lr, "Linear Regression")
    evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
    evaluate_model(y_test, y_pred_lasso, "Lasso Regression")
    


.. parsed-literal::

    Linear Regression - R2 Score: 0.0007, RMSE: 10.31
    Ridge Regression - R2 Score: 0.0007, RMSE: 10.31
    Lasso Regression - R2 Score: 0.0003, RMSE: 10.31
    




.. code:: ipython3

    #knn Social Media
    import pandas as pd, numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    
    data = pd.read_excel(r"C:\Users\darsh\OneDrive\Documents\Desktop\CL_1 Prt\Datasets\Social_Network_Ads1.xlsx")
    data.head()
    
    
    data.shape
    
    
    data.isnull().sum()
    
    
    sd = StandardScaler()
    
    
    x = data.drop('Purchased',axis=1)
    y = data['Purchased']
    
    
    x.shape , y.shape
    
    
    x [['Age','EstimatedSalary']] = sd.fit_transform(x[['Age','EstimatedSalary']])
    
    
    enc = LabelEncoder()
    x['Gender'] = enc.fit_transform(x['Gender'])
    
    
    
    x = x.drop('User ID',axis=1)
    x.head()
    
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=1)
    
    
    
    knn = KNeighborsClassifier()
    
    
    knn.fit(x_train,y_train)
    
    
    knn.predict(x_test)
    
    
    accuracy = knn.score(x_test,y_test)
    print('Accuracy :',accuracy)
    
    
    y_pred = knn.predict(x_test)
    
    
    print(classification_report(y_test,y_pred))
    
    
    confusion_matrix(y_test,y_pred)
    
    
    error = 1 - accuracy
    print('Error_rate : ', error)
    
    
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='viridis')
    
    
    




.. code:: ipython3

    #kmeans for iris dataset
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    
    # Load the Iris dataset
    df = pd.read_csv("./datasets/iris.csv")
    
    
    # Select features (attributes) for clustering (e.g., sepal_length, sepal_width, petal_length, petal_width)
    X = df.iloc[:, 1:-1]  # Exclude the first column (id) and the last column (species)
    
    
    # Standardize the feature matrix (important for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    # Determine the optimal number of clusters using the elbow method
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    
    print(inertia)
    
    
    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid()
    plt.show()
    
    
    




.. code:: ipython3

    # kmeans for iris dataset
    
    import pandas as pd,numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    
    data = pd.read_csv(r"C:\Users\darsh\OneDrive\Documents\Desktop\CL_1 Prt\Datasets\Iris.csv")
    
    
    x = data.drop('Species',axis=1)
    y = data['Species']
    
    
    data.head()
    
    
    scaled = StandardScaler()
    x_scaled = scaled.fit_transform(x)
    
    
    
    sse = []
    for k in range (1,20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x_scaled)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1,20),sse,marker='o')
    plt.title('Elbow Method')
    plt.xlabel('No. of Clusters')
    plt.ylabel('SSE (sum of squared of distances)')
    plt.show();
    
    
    # k = 3
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(x_scaled)
    
    
    kmeans.labels_
    
    
    plt.scatter(data['SepalLengthCm'],data['PetalLengthCm'],c=kmeans.labels_)
    
    
    



.. code:: ipython3

    #random forest to predict car safety
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    
    # Load the car evaluation dataset
    data = pd.read_csv("./datasets/car_evaluation.csv")
    
    
    # Encoding all the string data
    data = data.apply(LabelEncoder().fit_transform)
    
    
    # Define the features (X) and the target variable (y)
    X = data.iloc[:, :-1]  # Features (all columns except the last one)
    y = data.iloc[:, -1]   # Target variable (last column)
    
    
    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    
    # Train the classifier on the training data
    rf_classifier.fit(X_train, y_train)
    
    
    # Make predictions on the test data
    y_pred = rf_classifier.predict(X_test)
    
    
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    
    
    print(f"Accuracy: {accuracy}")
    print("\nConfusion Matrix:\n", confusion)
    print("\nClassification Report:\n", classification_rep)
    
    
    



.. code:: ipython3

    #random forest to predict car safety
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OrdinalEncoder
    
    car = pd.read_csv(r"car_evaluation.csv")
    car.head(5)
    
    car.columns = ['buying','maint','doors','persons','lug_boot','safety','class']
    
    car.head(5)
    
    
    car.info()
    
    
    car.shape
    
    
    car.isnull().sum()
    
    oe = OrdinalEncoder()
    car[['buying','maint','doors','persons','lug_boot','safety','class']] =oe.fit_transform(car[['buying','maint','doors','persons','lug_boot','safety','class']])
    car.head()
    
    
    car['persons'].value_counts()
    
    
    x = car.drop('class',axis=1)
    y = car['class']
    
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
    
    
    rf = RandomForestClassifier()
    rf.fit(x_train,y_train)
    
    
    accuracy = rf.score(x_test,y_test)*100
    print("Accuracy score : ",accuracy)
    
    
    y_pred = rf.predict(x_test)
    
    
    print(classification_report(y_test,y_pred))




.. code:: ipython3

    #tic tac toe
    import numpy as np
    import random
    
    
    # Task a & b: Setting up the Tic-Tac-Toe environment
    class TicTacToeEnv:
        def __init__(self):
            self.reset()
        
        def reset(self):
            self.board = np.zeros((3, 3), dtype=int)
            self.done = False
            self.current_player = 1  # 1 for 'X', -1 for 'O'
            return tuple(self.board.flatten())
        
        def available_actions(self):
            return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
        
        def step(self, action):
            if self.done:
                return tuple(self.board.flatten()), 0, True  # Game is already over
            i, j = action
            self.board[i, j] = self.current_player
            reward = self.check_winner()
            self.done = reward != 0 or not self.available_actions()
            self.current_player *= -1
            return tuple(self.board.flatten()), reward, self.done
    
        def check_winner(self):
            for i in range(3):
                if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                    return 1 * self.current_player
            if abs(sum(self.board.diagonal())) == 3 or abs(sum(np.fliplr(self.board).diagonal())) == 3:
                return 1 * self.current_player
            return 0
    
    
    
    
    
    
    
    
    
    
    
    # Task c: Building the Q-learning model
    Q = {}
    
    
    
    
    
    def choose_action(state, epsilon=0.1):
        if state not in Q:
            Q[state] = {a: 0 for a in env.available_actions()}
        return random.choice(env.available_actions()) if random.random() < epsilon else max(Q[state], key=Q[state].get)
    
    def update_q(state, action, reward, next_state, alpha=0.1, gamma=0.95):
        if state not in Q:
            Q[state] = {a: 0 for a in env.available_actions()}
        if next_state not in Q:
            # Set Q[next_state] with a default value of 0 if no available actions
            Q[next_state] = {a: 0 for a in env.available_actions()} or {(0, 0): 0} 
        
        # Q-learning update rule with terminal state check
        max_future_q = max(Q[next_state].values()) if Q[next_state] else 0
        Q[state][action] += alpha * (reward + gamma * max_future_q - Q[state][action])
    
    
    
    
    
    # Task d: Training the model
    env = TicTacToeEnv()
    for episode in range(10000):
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state)
            next_state, reward, done = env.step(action)
            update_q(state, action, reward, next_state)
            state = next_state
    
    
    
    
    
    # Task e: Testing the model
    def test_model():
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state, epsilon=0)  # Always exploit
            next_state, reward, done = env.step(action)
            print(np.reshape(next_state, (3, 3)))
            state = next_state
            if done:
                if reward > 0:
                    print("AI won!")
                elif reward < 0:
                    print("AI lost!")
                else:
                    print("It's a draw!")
    
    
    
    
    
    # Run a test
    test_model()








.. code:: ipython3

    #sales data
    
    import pandas as pd,json
    import re
    
    dcsv = pd.read_csv(r"C:\Users\darsh\OneDrive\Documents\Desktop\CL_1 Prt\Datasets\customers.csv")
    djson = pd.read_json(r"C:\Users\darsh\OneDrive\Documents\Desktop\CL_1 Prt\Datasets\customers.json")
    dxlsx = pd.read_excel(r"C:\Users\darsh\OneDrive\Documents\Desktop\CL_1 Prt\Datasets\customers.xlsx")
    
    
    dcsv.head()
    
    
    djson.head()
    
    
    dxlsx.head()
    
    
    print(f"Missing values in csv\n{dcsv.isna().sum()}")
    print(f"\nMissing values in json\n{djson.isna().sum()}")
    print(f"\nMissing values in xlsx\n{dxlsx.isna().sum()}")
    
    
    
    print(f"info of csv")
    dcsv.info()
    print(f"\ninfo of json")
    djson.info()
    print(f"\ninfo of xlsx")
    dxlsx.info()
    
    
    dcsv.fillna(0,inplace=True)
    djson.fillna(0,inplace=True)
    dxlsx.fillna(0,inplace=True)
    
    
    dcsv[dcsv.duplicated()]
    
    
    djson[djson.duplicated()]
    
    dxlsx[dxlsx.duplicated()]
    
    
    
    
    dcsv.drop_duplicates(inplace=True)
    djson.drop_duplicates(inplace=True)
    dxlsx.drop_duplicates(inplace=True)
    
    
    uni_df = pd.concat([dcsv,djson,dxlsx],ignore_index=True)
    uni_df.shape
    
    
    dcsv['full name'] = dcsv['first_name'] + ' ' + dcsv['last_name']
    
    
    dcsv.head()
    
    
    def extract_pin_code(address):
        match = re.search(r'\b\d{5}\b', address)
        return match.group(0) if match else None
    dcsv['pin code'] = [extract_pin_code(add) for add in dcsv['address']]
    
    
    dcsv.head()
    
    
    uni_df.describe()
    
    
    
    uni_df.groupby('job').agg({'orders':'sum',
                              'spent':'mean'})
    
    
    
    uni_df['spent'].sum() #calc total sales
    
    
    uni_df['spent'].mean() 
    
    
    uni_df['job'].value_counts() 
    
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.barplot(x='job', y='spent', data=uni_df)
    plt.title('Sales by Product Category')
    plt.xlabel('Product Category')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=90)
    plt.show()
    
    # Create a pie chart to represent the distribution of product categories
    product_distribution = uni_df['job'].value_counts()
    plt.pie(product_distribution, labels=product_distribution.index, autopct='%1.2f%%', startangle=140)
    plt.title('job Category Distribution')
    plt.xticks(rotation=90)
    plt.show()
    
    # Create a box plot to visualize the distribution of order values
    sns.boxplot(x='job', y='spent', data=uni_df)
    plt.title('Order Value Distribution by Job')
    plt.xlabel('Job')
    plt.ylabel('Order Value')
    plt.xticks(rotation=90)
    plt.show()


.. code:: ipython3

    #sales data bakwas wala
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import json
    
    
    csv = pd.read_csv("sales_data_sample.csv", encoding="cp1252")
    
    
    ed = pd.read_excel("./datasets/Sample-Sales-Data.xlsx")
    
    
    with open("./datasets/customers.json", "r") as json_file:
        json_data = json.load(json_file)
    
    
    csv.tail()
    
    
    csv.info()
    
    
    csv.describe()
    
    
    csv.dropna()
    
    
    
    csv.drop_duplicates()
    
    
    ed.head()
    
    
    ed.tail()
    
    
    ed.info()
    
    
    ed.describe()
    
    
    unified_data = pd.concat([csv, ed], ignore_index=True)
    
    
    total_sales = unified_data['SALES'].sum()
    print("Total Sales:", total_sales)
    
    
    category_sales = unified_data.groupby('ORDERNUMBER')['SALES'].mean()
    
    
    category_counts = unified_data['SALES'].value_counts()
    category_counts.plot(kind='bar')
    plt.title('Product Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.show()
    
    
    





.. code:: ipython3

    #Open weather Map api 
    
    import requests
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    url = 'https://api.openweathermap.org/data/2.5/forecast?lat=44.34&lon=10.99&appid=ee9cfc2b8e9f8d695fc2e509dbf2659c' # <-- This is app id also known as api key, which you have to generate your own, and paste it here
    
    #vurl ='https://api.openweathermap.org/data/2.5/weather?lat=44.34&lon=10.99&appid=307d11bca480dd730d99187c926&city_name=London'
    response = requests.get(url)
    data = response.json()
    pretty_json = json.dumps(data,indent=4)
    print(pretty_json)
    
    
    date_time = data['list'][0]['dt_txt']
    date_time
    
    
    data_struct = []
    for record in data['list']:
        temp = record['main']['temp']
        humid = record['main']['humidity']
        wind_speed = record['wind']['speed']
        desp =record['weather'][0]['description']
        date_time = record['dt_txt']
    
        data_struct.append({'Temperature':temp,'Humidity':humid,'Wind Speed':wind_speed,'Weather Description':desp,'Date Time':date_time})
    
    data_df = pd.DataFrame(data_struct)
    data_df[['Date','Time']] = data_df['Date Time'].str.split(' ',expand=True)
    data_df.head(10)
    
    
    
    data_df.shape
    
    
    data_df.isnull().sum()
    
    
    data_df.info()
    
    
    data_df.describe()
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=data_df['Date Time'],
                    y=data_df['Temperature'],
                    hue=data_df['Weather Description'])
    plt.xticks(rotation=90);
    
    
    
    
    plt.figure(figsize=(10,5))
    plt.plot(data_df['Date Time'],data_df['Humidity'])
    plt.xticks(rotation=90);
    
    
    data_df.groupby('Date').agg({'Temperature' : 'mean','Wind Speed':'mean'})
    
    
    sns.heatmap(data_df[['Temperature','Humidity','Wind Speed']].corr(),annot=True,cmap='crest')
    
    
    


.. code:: ipython3

    #Open weather map 

.. code:: ipython3

    import requests 
    import pandas as pd
    import datetime
    
    
    # Set your OpenWeatherMap API key
    api_key = 'fb365aa6104829b44455572365ff3b4e' 
    
    
    
    # Set the location for which you want to retrieve weather data 
    lat = 18.184135
    lon = 74.610764
    
    
    
    # https://openweathermap.org/api/one-call-3
    # how	How to use api call 
    # Construct the API URL
    api_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"
    
    
    # Send a GET request to the API 
    response = requests.get(api_url)
    weather_data = response.json()
    weather_data.keys()
    len(weather_data['list'])
    weather_data['list'][0]['weather'][0]['description']
    
    
    
    # Getting the data from dictionary and taking into one variable 
    # Extract relevant weather attributes using list comprehension
    temperatures = [item['main']['temp'] for item in weather_data['list']] 
    
    # It will extract all values (40) and putting into one variable
    timestamps = [pd.to_datetime(item['dt'], unit='s') for item in weather_data['list']]
    temperature = [item['main']['temp'] for item in weather_data['list']]
    humidity = [item['main']['humidity'] for item in weather_data['list']]
    wind_speed = [item['wind']['speed'] for item in weather_data['list']]
    weather_description = [item['weather'][0]['description'] for item in weather_data['list']]
    
    
    
    # Create a pandas DataFrame with the extracted weather data
    weather_df = pd.DataFrame({'Timestamp': timestamps, 
                               'Temperature': temperatures, 
                               'humidity': humidity, 
                               'wind_speed':wind_speed,
                               'weather_description': weather_description})
    
    
    
    # Set the Timestamp column as the DataFrame's index
    weather_df.set_index('Timestamp', inplace=True)
    max_temp = weather_df['Temperature'].max()
    print(f"Maximum Temperature - {max_temp}")
    min_temp = weather_df['Temperature'].min()
    print(f"Minimum Temperature - {min_temp}")
    
    
    
    # Clean and preprocess the data # Handling missing values
    weather_df.fillna(0, inplace=True) # Replace missing values with 0 or appropriate value
    
    
    
    # Handling inconsistent format (if applicable)
    weather_df['Temperature'] = weather_df['Temperature'].apply(lambda x: x - 273.15 if isinstance(x, float)else x)
    
    
    
    # Convert temperature from Kelvin to Celsius
    # Print the cleaned and preprocessed data print(weather_df)
    weather_df.head()
    
    
    
    import matplotlib.pyplot as plt
    daily_mean_temp = weather_df['Temperature'].resample('D').mean()
    daily_mean_humidity = weather_df['humidity'].resample('D').mean()
    daily_mean_wind_speed = weather_df['wind_speed'].resample('D').mean()
    
    
    
    # Plot the mean daily temperature over time (Line plot)
    plt.figure(figsize=(10, 6))
    daily_mean_temp.plot(color='red', linestyle='-', marker='o')
    plt.title('Mean Daily Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.show()
    
    
    
    # Plot the mean daily humidity over time (Bar plot)
    plt.figure(figsize=(10, 6))
    daily_mean_humidity.plot(kind='bar', color='blue')
    plt.title('Mean Daily Humidity')
    plt.xlabel('Date')
    plt.ylabel('Humidity (%)')
    plt.grid(True)
    plt.show()
    
    
    
    # Plot the relationship between temperature and wind speed (Scatter plot)
    plt.figure(figsize=(10, 6))
    plt.scatter(weather_df['Temperature'], weather_df['wind_speed'], color='green')
    plt.title('Temperature vs. Wind Speed')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Wind Speed (m/s)')
    plt.grid(True)
    plt.show()
    
    
    
    # Heatmap
    import seaborn as sns
    heatmap_data = weather_df[['Temperature', 'humidity']]
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
    plt.title('Temperature vs Humidity Heatmap')
    plt.show()
    
    
    
    # Create a scatter plot to visualize the relationship between temperature and humidity
    plt.scatter(weather_df['Temperature'], weather_df['humidity'])
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Humidity (%)')
    plt.title('Temperature vs Humidity Scatter Plot')
    plt.show()
    
    
    





.. code:: ipython3

    #customer churn
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split 
    from sklearn import metrics
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    
    
    data = pd.read_csv("./datasets/Telcom_Customer_Churn.csv")
    print(data.index)
    
    
    data.head()
    
    
    print(data.columns)
    
    
    
    data.shape 
    
    
    data.nunique()
    
    
    data.isna().sum()
    
    
    data.isnull().sum()
    
    
    
    # Check the number of rows before removing duplicates 
    print("Number of rows before removing duplicates:", len(data))
    
    
    
    # Remove duplicate records
    data_cleaned = data.drop_duplicates()
    
    
    
    # Remove duplicate records
    data_cleaned = data.drop_duplicates()
    
    
    
    data.describe()
    
    
    
    # Measure of frequency destribution
    unique, counts = np.unique(data['tenure'], return_counts=True) 
    print(unique, counts)
    
    
    
    # Measure of frequency destribution
    unique, counts = np.unique(data['MonthlyCharges'], return_counts=True) 
    print(unique, counts)
    
    
    
    # Measure of frequency destribution
    unique, counts = np.unique(data['TotalCharges'], return_counts=True) 
    print(unique, counts)
    
    
    sns.pairplot(data)
    
    
    
    plt.boxplot(data['tenure'])
    plt.show()
    
    
    plt.boxplot(data['MonthlyCharges']) 
    plt.show()
    
    
    
    X = data.drop("Churn", axis=1) 
    y = data["Churn"]
    
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    X_train.shape 
    
    
    y_train.shape 
    
    
    X_test.shape 
    
    
    y_test.shape 
    
    
    # Export the cleaned dataset to a CSV file 
    data.to_csv("./datasets/Cleaned_Telecom_Customer_Churn.csv", index=False)
    
    
    
    


.. code:: ipython3

    
    #customer churn
    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    
    
    data = pd.read_csv(r"C:\Users\darsh\OneDrive\Documents\Desktop\CL_1 Prt\Datasets\Telco-Customer-Churn.csv")
    data.head()
    
    
    data.info()
    
    
    data.info()
    
    
    data.describe()
    
    
    data[data.duplicated(subset=['customerID'])].T
    
    
    data = data.drop_duplicates(subset=['customerID'])
    
    
    data.shape
    
    data['gender'].value_counts()
    
    
    data['gender'] = data['gender'].replace({'F':'Female','M ':'Male'})
    
    
    
    data.gender.value_counts()
    
    
    
    data.isnull().sum()
    
    
    categorical_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','Contract','StreamingMovies','PaymentMethod','Churn']
    numercial_cols = ['MonthlyCharges','tenure','TotalCharges']
    
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    for col in numercial_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric
        data[col].fillna(data[col].mean(), inplace=True)
    
    
    
    
    data.isna().sum()
    
    
    data.dtypes
    
    
    data[['tenure','MonthlyCharges','TotalCharges']] = data[['tenure','MonthlyCharges','TotalCharges']].astype('int64')
    
    
    data.dtypes
    
    
    # Detecting outliers
    outliers = data[(data["MonthlyCharges"] > 3.5 * data["MonthlyCharges"].std()) | (data["MonthlyCharges"] < -3.5 * data["MonthlyCharges"].std())]
    outliers
    
    
    # Removed outliers
    data = data[(data["MonthlyCharges"] < 3.5 * data["MonthlyCharges"].std()) & (data["MonthlyCharges"] > -3.5 * data["MonthlyCharges"].std())]
    
    
    
    data.shape
    
    
    data['early_churn'] = (data['tenure'] <= 12) & (data['Churn'] == 'Yes')
    
    
    
    data.head()
    
    
    
    from sklearn.preprocessing import Normalizer as n
    
    data[['MonthlyCharges','TotalCharges']] = n().fit_transform(data[['MonthlyCharges','TotalCharges']])
    
    
    data.head()
    
    
    from sklearn.model_selection import train_test_split
    
    X = data.drop('Churn',axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    
    
    X_train.shape
    
    
    data.shape
    
    
    
    X_train.to_csv('Xtrain.csv')
    X_test.to_csv('Xtest.csv')
    



.. code:: ipython3

    #reale estate 
    # Step 1: Import and clean column names
    import pandas as pd
    df = pd.read_csv(r"Real estate.csv")
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    
    # Step 2: Handle missing values
    missing_values = df.isnull().sum()
    if missing_values > 0:
        # Imputation or removal strategy
    
    # Step 3: Perform data merging (if necessary)
    if additional_data:
        df_merged = pd.merge(df, additional_data, on="property_id")
    
    # Step 4: Filter and subset the data
    filtered_df = df.loc[(df["sale_date"] >= "2020-01-01") & (df["property_type"] == "residential")]
    
    # Step 5: Handle categorical variables
    categorical_cols = ["property_type", "location"]
    for col in categorical_cols:
        df[col] = pd.get_dummies(df[col], drop_first=True)
    
    # Step 6: Aggregate the data
    aggregated_df = df.groupby("neighborhood").agg({"sale_price": "mean"})
    
    # Step 7: Identify and handle outliers
    outliers = df[(df["sale_price"] > 3 * df["sale_price"].std()) | (df["sale_price"] < -3 * df["sale_price"].std())]
    # if outliers:
        # Handle outliers
    
    
    
    
    
    
    
    
    
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv('../../Datasets/RealEstate_Prices.csv')
    additional = pd.read_csv('../../Datasets/additional_info.csv')
    df.columns,additional.columns
    
    
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    additional.columns = [col.strip().replace(" ", "_").lower() for col in additional.columns]
    
    
    
    df.columns,additional.columns
    
    
    merged_df = pd.merge(df,additional,on='property_id',how='inner')
    
    
    
    # checking for missing values
    merged_df.isna().sum()
    
    
    
    categorical_cols = ['bedrooms']
    numerical_cols = ['area_sq_ft','sale_price','demographics_population']
    
    for col in categorical_cols:
        merged_df[col] = merged_df[col].fillna(merged_df[col].mode()[0])
    
    for col in numerical_cols:
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
    
    
    merged_df.isna().sum()
    
    
    merged_df = merged_df.dropna()
    
    
    
    merged_df.isna().sum()
    
    
    
    filtered_df = merged_df[(merged_df["sale_date"] >= "2020-01-01") & (merged_df["property_type"] == "Apartment")]
    filtered_df
    
    
    
    from sklearn.preprocessing import LabelEncoder
    
    categorical_cols = ['property_type',
    'neighborhood',
    'crime_rate',
    'house_condition']
    
    enc = LabelEncoder()
    
    for col in categorical_cols:
        merged_df[col] = enc.fit_transform(merged_df[col])
    
    
    
    
    
    merged_df.head()
    
    
    
    aggregated_df = merged_df.groupby("neighborhood").agg({"sale_price": "mean"})
    aggregated_df
    
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    cols = [
    'area_sq_ft',
    'sale_price',
    'distance_to_amenities_mi',
    'demographics_population',
    'property_taxes'
    ]
    
    for col in cols:
        sns.boxplot(merged_df[col])
        plt.show()
    
    
    
    
    merged_df[(merged_df["sale_price"] > 3 * merged_df["sale_price"].std()) | (merged_df["sale_price"] < -3 * merged_df["sale_price"].std())]
    

.. code:: ipython3

    #real estate
    
    import pandas as pd 
    import numpy as np
    from matplotlib import pyplot as plt
    import warnings
    
    
    # Supressing update warnings
    warnings.filterwarnings('ignore')
    
    
    
    df1 = pd.read_csv("./datasets/Bengaluru_House_Data.csv") 
    
    
    
    df1.head()
    
    
    df1.shape 
    
    
    df1.columns
    
    
    df1['area_type']
    
    
    df1['area_type'].unique()
    
    
    
    df1['area_type'].value_counts()
    
    
    df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns') 
    
    
    
    df2.shape
    
    
    
    df2.isnull().sum()
    
    
    
    df2.shape 
    
    
    
    
    df3 = df2.dropna() 
    df3.isnull().sum()
    
    
    
    df3.shape 
    
    
    df3['size'].unique()
    
    
    
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
    
    
    
    
    df3.head()
    
    
    
    df3.bhk.unique()
    
    
    
    df3[df3.bhk>20]
    
    
    df3.total_sqft.unique()
    
    
    
    def is_float(x):
        try:
            float(x) 
            return True
        except(ValueError, TypeError):
            return False 
    
    
    
    
    df3[~df3['total_sqft'].apply(is_float)].head(10)
    
    
    
    def convert_sqft_to_num(x): 
        tokens = x.split('-')
        if len(tokens) == 2:
            try:
                return (float(tokens[0])+float(tokens[1]))/2
            except ValueError:
                return None
        try:
            return float(x) 
        except ValueError:
            return None 
        
    result = convert_sqft_to_num('2100 - 2850')
    print(result)
    
    
    
    
    convert_sqft_to_num('34.46Sq. Meter') 
    df4 = df3.copy()
    df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num) 
    df4
    
    
    
    df4 = df4[df4.total_sqft.notnull()] 
    df4
    
    
    
    df4.loc[30]
    
    
    
    df5 = df4.copy()
    df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft'] 
    df5.head()
    
    
    
    df5_stats = df5['price_per_sqft'].describe() 
    df5_stats
    
    
    
    
    
    
    
    df5.to_csv("./datasets/bhp.csv",index=False)
    
    
    
    df5.location = df5.location.apply(lambda x: x.strip()) 
    location_stats = df5['location'].value_counts(ascending=False) 
    location_stats
    
    
    
    len(location_stats[location_stats>10])
    
    
    len(location_stats) 
    
    
    
    len(location_stats) 
    
    
    
    location_stats_less_than_10 = location_stats[location_stats<=10] 
    location_stats_less_than_10
    
    
    
    len(df5.location.unique())
    
    
    
    df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x) 
    len(df5.location.unique())
    
    
    
    
    df5.head(10)
    
    
    
    df5[df5.total_sqft/df5.bhk<300].head()
    
    
    
    
    df5.shape 
    
    
    
    
    df6 = df5[~(df5.total_sqft/df5.bhk<300)] 
    df6.shape
    
    
    
    df6.columns
    
    
    
    df6.columns
    
    
    
    plt.boxplot(df6['total_sqft']) 
    plt.show()
    
    
    
    
    
    Q1 = np.percentile(df6['total_sqft'], 25.) # 25th percentile of the data of the given feature 
    Q3 = np.percentile(df6['total_sqft'], 75.) # 75th percentile of the data of the given feature 
    IQR = Q3-Q1 #Interquartile Range
    ll = Q1 - (1.5*IQR) 
    ul = Q3 + (1.5*IQR)
    upper_outliers = df6[df6['total_sqft'] > ul].index.tolist() 
    lower_outliers = df6[df6['total_sqft'] < ll].index.tolist() 
    bad_indices = list(set(upper_outliers + lower_outliers)) 
    drop = True
    if drop:
        df6.drop(bad_indices, inplace = True, errors = 'ignore')
    
    plt.boxplot(df6['bath']) 
    plt.show()
    
    
    
    
    
    Q1 = np.percentile(df6['bath'], 25.) # 25th percentile of the data of the given feature 
    Q3 = np.percentile(df6['bath'], 75.) # 75th percentile of the data of the given feature 
    IQR = Q3-Q1 #Interquartile Range
    ll = Q1 - (1.5*IQR) 
    ul = Q3 + (1.5*IQR)
    upper_outliers = df6[df6['bath'] > ul].index.tolist() 
    lower_outliers = df6[df6['bath'] < ll].index.tolist() 
    bad_indices = list(set(upper_outliers + lower_outliers)) 
    drop = True
    if drop:
        df6.drop(bad_indices, inplace = True, errors = 'ignore')
    plt.boxplot(df6['price']) 
    plt.show()
    
    
    
    
    
    
    
    
    
    
    Q1 = np.percentile(df6['price'], 25.) # 25th percentile of the data of the given feature 
    Q3 = np.percentile(df6['price'], 75.) # 75th percentile of the data of the given feature 
    IQR = Q3-Q1 #Interquartile Range
    ll = Q1 - (1.5*IQR) 
    ul = Q3 + (1.5*IQR)
     
    upper_outliers = df6[df6['price'] > ul].index.tolist() 
    lower_outliers = df6[df6['price'] < ll].index.tolist() 
    bad_indices = list(set(upper_outliers + lower_outliers)) 
    drop = True
    if drop:
        df6.drop(bad_indices, inplace = True, errors = 'ignore')
    
    plt.boxplot(df6['bhk']) 
    plt.show()
    
    
    
    
    
    
    
    
    
    Q1 = np.percentile(df6['bhk'], 25.) # 25th percentile of the data of the given feature 
    Q3 = np.percentile(df6['bhk'], 75.) # 75th percentile of the data of the given feature 
    IQR = Q3-Q1 #Interquartile Range
    ll = Q1 - (1.5*IQR) 
    ul = Q3 + (1.5*IQR)
    upper_outliers = df6[df6['bhk'] > ul].index.tolist() 
    lower_outliers = df6[df6['bhk'] < ll].index.tolist() 
    bad_indices = list(set(upper_outliers + lower_outliers)) 
    drop = True
    if drop:
        df6.drop(bad_indices, inplace = True, errors = 'ignore')
    
    plt.boxplot(df6['price_per_sqft']) 
    plt.show()
    
    
    
    
    Q1 = np.percentile(df6['price_per_sqft'], 25.) # 25th percentile of the data of the given feature 
    Q3 = np.percentile(df6['price_per_sqft'], 75.) # 75th percentile of the data of the given feature 
    IQR = Q3-Q1 #Interquartile Range
    ll = Q1 - (1.5*IQR) 
    ul = Q3 + (1.5*IQR)
    upper_outliers = df6[df6['price_per_sqft'] > ul].index.tolist() 
    lower_outliers = df6[df6['price_per_sqft'] < ll].index.tolist() 
    bad_indices = list(set(upper_outliers + lower_outliers))
    drop = True 
    if drop:
        df6.drop(bad_indices, inplace = True, errors = 'ignore')
    
    plt.boxplot(df6['price_per_sqft']) 
    plt.show()
    
    
    
    
    
    
    
    
    df6.shape
    
    
    
    
    
    
    X = df6.drop(['price'],axis='columns') 
    X.head(3)
    
    
    
    
    
    X.shape 
    
    
    
    y = df6.price 
    y.head(3)
    
    
    
    len(y)
    
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    
    X_train.shape 
    
    
    
    
    
    y_train.shape 
    
    
    
    X_test.shape 
    
    
    
    
    y_test.shape 
    
    
    
    y_test.shape 
    
    




.. code:: ipython3

    #AQI
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Step 1: Import the dataset
    data = pd.read_csv(r"C:\Users\darsh\OneDrive\Documents\Desktop\CL_1 Prt\Datasets\City_Air_Quality.csv", sep=';')
    
    # Step 2: Explore the dataset
    print("Dataset Head:\n", data.head())
    print("\nDataset Info:\n", data.info())
    
    # Step 3: Identify relevant variables and clean column names
    data.columns = data.columns.str.strip()  # Remove any leading/trailing spaces from column names
    
    # Correct the time format
    data['Time'] = data['Time'].str.replace('.', ':', regex=False)  # Ensure replacement is done correctly
    
    # Combine Date and Time into a single datetime column
    data['Date'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    
    # Check for NaT values and print rows with missing dates
    missing_dates = data[data['Date'].isna()]
    if not missing_dates.empty:
        print("\nRows with missing Date values:\n", missing_dates)
    
    # Drop rows with NaT values in 'Date'
    data.dropna(subset=['Date'], inplace=True)
    
    # Step 4: Convert pollutant columns to numeric, forcing errors to NaN
    pollutants = ['NO2(GT)', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)']
    for pollutant in pollutants:
        data[pollutant] = pd.to_numeric(data[pollutant], errors='coerce')
    
    # Check for any NaN values in pollutant columns after conversion
    print("\nNaN values in pollutant columns after conversion:\n", data[pollutants].isna().sum())
    
    # Step 5: Line plot for overall AQI trend over time
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['NO2(GT)'], color='blue', label='AQI (NO2)', linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("NO2(GT) Levels (µg/m³)")
    plt.title("NO2 Trends Over Time")
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-ticks for better readability
    plt.tight_layout()  # Adjust layout
    plt.show()
    
    # Step 6: Line plots for individual pollutants
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['CO(GT)'], color='red', label='CO', linewidth=2)
    plt.plot(data['Date'], data['NMHC(GT)'], color='green', label='NMHC', linewidth=2)
    plt.plot(data['Date'], data['C6H6(GT)'], color='purple', label='C6H6', linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Pollutant Levels (µg/m³)")
    plt.title("Trends of CO, NMHC, and C6H6 Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Step 7: Bar plot for average AQI values by month
    data['Month'] = data['Date'].dt.month_name()  # Get month names for better labeling
    monthly_aqi = data.groupby('Month')['NO2(GT)'].mean().reindex([
        'January', 'February', 'March', 'April', 'May', 
        'June', 'July', 'August', 'September', 'October', 
        'November', 'December'
    ])  # Ensure months are in order
    
    plt.figure(figsize=(10, 5))
    plt.bar(monthly_aqi.index, monthly_aqi.values, color='orange')
    plt.xlabel("Month")
    plt.ylabel("Average NO2(GT) (µg/m³)")
    plt.title("Average Monthly NO2 Levels")
    plt.xticks(rotation=45)  # Rotate month labels for better readability
    plt.tight_layout()
    plt.show()
    
    # Step 8: Box plot for pollutant level distribution
    plt.figure(figsize=(8, 5))
    plt.boxplot([data['CO(GT)'].dropna(), data['NMHC(GT)'].dropna(), data['C6H6(GT)'].dropna()], 
                labels=['CO', 'NMHC', 'C6H6'], 
                patch_artist=True)  # Add color to box plots
    plt.xlabel("Pollutants")
    plt.ylabel("Level (µg/m³)")
    plt.title("Pollutant Level Distribution")
    plt.tight_layout()
    plt.show()
    
    # Step 9: Scatter plot for NO2 vs. Pollutant Levels
    plt.figure(figsize=(8, 5))
    plt.scatter(data['CO(GT)'], data['NO2(GT)'], color='red', label='CO vs NO2', alpha=0.5)
    plt.scatter(data['NMHC(GT)'], data['NO2(GT)'], color='green', label='NMHC vs NO2', alpha=0.5)
    plt.scatter(data['C6H6(GT)'], data['NO2(GT)'], color='purple', label='C6H6 vs NO2', alpha=0.5)
    plt.xlabel("Pollutant Level (µg/m³)")
    plt.ylabel("NO2(GT) Levels (µg/m³)")
    plt.title("Relationship Between NO2 and Other Pollutant Levels")
    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    


.. code:: ipython3

    #AQI
    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt 
    import seaborn as sns
    from sklearn.impute import SimpleImputer
    import warnings 
    
    
    # Supressing update warnings
    warnings.filterwarnings('ignore') 
    
    
    
    data = pd.read_csv("./datasets/data.csv", encoding="cp1252") 
    data
    
    
    
    
    
    
    data.info()
    
    
    
    
    # Cleaning up name changes
    data.state = data.state.replace({'Uttaranchal':'Uttarakhand'}) 
    data.state[data.location == "Jamshedpur"] = data.state[data.location == 'Jamshedpur'].replace({"Bihar":"Jharkhand"})
    
    
    
    
    # Changing types to uniform format
    types = {
        "Residential": "R", 
        "Residential and others": "RO",
        "Residential, Rural and other Areas": "RRO", 
        "Industrial Area": "I",
        "Industrial Areas": "I", 
        "Industrial": "I", 
        "Sensitive Area": "S", 
        "Sensitive Areas": "S", 
        "Sensitive": "S", 
        np.nan: "RRO"
    }
    
    data.type = data.type.replace(types) 
    data.head()
    
    
    
    
    # defining columns of importance, which shall be used reguarly 
    VALUE_COLS = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']
    
    
    
    # invoking SimpleImputer to fill missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
    data[VALUE_COLS] = imputer.fit_transform(data[VALUE_COLS])
    
    
    
    
    
    # checking to see if the dataset has any null values left over and the format 
    print(data.isnull().sum())
    data.tail()
    
    
    
    
    
    
    
    
    # Plotting highest and lowest ranking states
    # defining a function to find and plot the top 10 and bottom 10 states for a given indicator (defaults to SO2) 
    def top_and_bottom_10_states(indicator="so2"):
        fig, ax = plt.subplots(2,1, figsize=(20, 12))
        ind = data[[indicator, 'state']].groupby('state', as_index=False).median().sort_values(by=indicator,ascending=False)
        top10 = sns.barplot(x='state', y=indicator, data=ind[:10], ax=ax[0], color='red') 
        top10.set_title("Top 10 states by {} (1991-2016)".format(indicator)) 
        top10.set_ylabel("so2 (µg/m3)")
        top10.set_xlabel("State")
        bottom10 = sns.barplot(x='state', y=indicator, data=ind[-10:], ax=ax[1], color='green') 
        bottom10.set_title("Bottom 10 states by {} (1991-2016)".format(indicator)) 
        bottom10.set_ylabel("so2 (µg/m3)")
        bottom10.set_xlabel("State") 
    
    top_and_bottom_10_states("so2") 
    top_and_bottom_10_states("no2")
    
    
    
    
    
    
    # Plotting the highest ever recorded levels
    # defining a function to find the highest ever recorded levels for a given indicator (defaults to SO2) by state 
    # sidenote: mostly outliers
    def highest_levels_recorded(indicator="so2"): 
        plt.figure(figsize=(20,10))
        ind = data[[indicator, 'location', 'state', 'date']].groupby('state', as_index=False).max() 
        highest = sns.barplot(x='state', y=indicator, data=ind)
        highest.set_title("Highest ever {} levels recorded by state".format(indicator))
        plt.xticks(rotation=90) 
        
    highest_levels_recorded("no2") 
    highest_levels_recorded("rspm")
    
    
    
    
    
    
    # Plotting pollutant average by type
    # defining a function to plot pollutant averages by type for a given indicator 
    def type_avg(indicator=""):
        type_avg = data[VALUE_COLS + ['type', 'date']].groupby("type").mean() 
        if not indicator:
            t = type_avg[indicator].plot(kind='bar') 
            plt.xticks(rotation = 0)
            plt.title("Pollutant average by type for {}".format(indicator)) 
        else:
            t = type_avg.plot(kind='bar') 
            plt.xticks(rotation = 0) 
            plt.title("Pollutant average by type")
    
    type_avg('so2')
    
    
    
    
    
    
    
    # Plotting pollutant averages by locations/state
    # defining a function to plot pollutant averages for a given indicator (defaults to SO2) by locations in a given state 
    def location_avgs(state, indicator="so2"):
        locs = data[VALUE_COLS + ['state', 'location', 'date']].groupby(['state', 'location']).mean() 
        state_avgs = locs.loc[state].reset_index()
        sns.barplot(x='location', y=indicator, data=state_avgs) 
        plt.title("Location-wise average for {} in {}".format(indicator, state)) 
        plt.xticks(rotation = 90)
    
    location_avgs("Bihar", "no2")
    
    
    
    
    
    
    


.. code:: ipython3

    #retailsales
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Step 1: Load the dataset
    data = pd.read_csv(r"C:\Users\darsh\OneDrive\Documents\Desktop\CL_1 Prt\Datasets\retail_sales_dataset.csv")
    
    # Step 2: Display the first few rows of the dataset
    print("Dataset Head:\n", data.head())
    
    # Step 3: Display dataset info
    print("\nDataset Info:\n", data.info())
    
    # Step 4: Group by Product Category and calculate total sales amount
    product_sales = data.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)
    print("\nTotal Sales by Product Category:\n", product_sales)
    
    # Step 5: Visualize sales distribution by product category using a bar plot
    plt.figure(figsize=(10, 6))
    product_sales.plot(kind='bar', color='skyblue')
    plt.title('Total Sales by Product Category', fontsize=16)
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Total Sales Amount', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    

.. code:: ipython3

    #retail sales
    import pandas as pd
    import matplotlib.pyplot as plt
    
    
    
    df = pd.read_csv("./datasets/customer_shopping_data.csv") 
    df.head()
    
    df.tail()
    
    
    # To check the count of records grouped by region/branch of the mall
    df.groupby("shopping_mall").count()
    
    
    # To check the count of records grouped by the product categories
    df.groupby("category").count()
    
    
    
    
    # total sales for each mall branch
    branch_sales = df.groupby("shopping_mall").sum()
    branch_sales
    
    
    
    
    
    # total sales for each category of product
    category_sales = df.groupby("category").sum()
    category_sales
    
    
    
    # to get the top performing branches
    branch_sales.sort_values(by = "price", ascending = False)
    
    
    
    
    
    # to get the top selling categories
    category_sales.sort_values(by = "price", ascending = False)
    
    
    
    # to get total sales for each combination of branch and product_category
    combined_branch_category_sales = df.groupby(["shopping_mall", "category"]).sum()
    combined_branch_category_sales
    
    
    
    
    
    
    # pie chart for sales by branch
    plt.pie(branch_sales["price"], labels = branch_sales.index) 
    plt.show()
    
    
    
    
    
    
    # pie chart for sales by product category
    plt.pie(category_sales["price"], labels = category_sales.index) 
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    combined_pivot = df.pivot_table(index="shopping_mall", columns="category", values="price", aggfunc="sum") 
    
    
    
    
    # grouped bar chart for sales of different categories at different branches
    combined_pivot.plot(kind="bar", figsize=(10, 6)) 
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
