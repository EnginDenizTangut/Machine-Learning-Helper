import pandas as pd
import tkinter as tk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler,RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor,IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,mean_squared_error,r2_score,accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.impute import KNNImputer

class Buddy:
    def __init__(self):
        self.dataset = None
        self.file_path = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None



    def run(self):
        self.upload_file()
        self.load_dataset()
        self.handle_missing_values()
        self.categorized()
        #self.outliers_find()
        self.remove_outliers()
        #self.outliers_find()
        self.split_data()
        self.scaling()
        self.anomaly_detection()
        self.select_model()



    def upload_file(self):
        root = tk.Tk()
        root.withdraw()
        self.file_path = filedialog.askopenfilename(
            title="Bir dosya seçin",
            filetypes=(("CSV dosyaları", "*.csv"), ("Tüm Dosyalar", "*.*"))
        )

    def load_dataset(self):
        if self.file_path:
            self.dataset = pd.read_csv(self.file_path)
        else:
            print("Dosya seçilmedi.")
            exit()

    
        

    def handle_missing_values(self):
        while True:
            try:
                selection = int(input("Which type of way you want?\n1) dropna()\n2) fillna()\n3) KNN Imputer\n"))
                
                if selection == 1:
                    if self.dataset.isnull().values.any():
                        print("Eksik veriler bulundu. Siliniyor...")
                        self.dataset = self.dataset.dropna()
                    else:
                        print("Eksik veri bulunamadı.")
                    break  # dropna işleminden sonra döngüden çıkılır
                elif selection == 2:
                    method_selection = int(input("Which method you want?\n1) mean\n2) median\n3) 0\n"))
                    if method_selection == 1:
                        print("Eksik veriler bulundu. Ortalama ile dolduruluyor...")
                        # Yalnızca sayısal sütunları işleme al
                        numeric_columns = self.dataset.select_dtypes(include=['float64', 'int64']).columns
                        self.dataset[numeric_columns] = self.dataset[numeric_columns].fillna(self.dataset[numeric_columns].mean())
                    elif method_selection == 2:
                        print("Eksik veriler bulundu. Medyan ile dolduruluyor...")
                        # Yalnızca sayısal sütunları işleme al
                        numeric_columns = self.dataset.select_dtypes(include=['float64', 'int64']).columns
                        self.dataset[numeric_columns] = self.dataset[numeric_columns].fillna(self.dataset[numeric_columns].median())
                    elif method_selection == 3:
                        print("Eksik veriler bulundu. 0 ile dolduruluyor...")
                        self.dataset = self.dataset.fillna(0)
                    else:
                        print("Geçersiz seçim. Lütfen 1, 2 veya 3 girin.")
                        continue  # Geçersiz seçenek durumunda tekrar sorulur
                    break  # fillna işleminden sonra döngüden çıkılır
                elif selection == 3:
                    print("Eksik veriler bulundu. KNN ile dolduruluyor...")
                    # Yalnızca sayısal sütunları işleme al
                    numeric_columns = self.dataset.select_dtypes(include=['float64', 'int64']).columns
                    if not numeric_columns.empty:
                        imputer = KNNImputer(n_neighbors=5)  # KNN imputer kullanarak eksik verileri dolduruyoruz
                        self.dataset[numeric_columns] = imputer.fit_transform(self.dataset[numeric_columns])
                        print("Eksik veriler KNN ile dolduruldu.")
                    else:
                        print("Sayısal sütun bulunamadı. KNN imputer sadece sayısal verilerle çalışır.")
                        continue
                    break  # KNN imputer işleminden sonra döngüden çıkılır
                else:
                    print("Geçersiz seçim. Lütfen 1, 2 veya 3 girin.")
            except ValueError:
                print("Geçersiz giriş. Lütfen sayı girin.")

    def categorized(self):
        for column in self.dataset.columns:
            if len(self.dataset[column].unique()) > 2 and self.dataset[column].dtype == 'object':
                self.dataset = pd.get_dummies(self.dataset, columns=[column], drop_first=True)
            else:
                le = LabelEncoder()
                self.dataset[column] = le.fit_transform(self.dataset[column])

        for column in self.dataset.columns:
            if self.dataset[column].dtype == 'bool':
                le = LabelEncoder()
                self.dataset[column] = le.fit_transform(self.dataset[column])

    def outliers_find(self):
        numeric_columns = self.dataset.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:
            plt.boxplot(self.dataset[column])
            plt.title(f"Boxplot for {column}")
            plt.show()

    def remove_outliers(self):
        outliers_way = int(input("Which type of outliers way you want?\n1)IQR\n2)Z-Score\n"))
        if outliers_way == 1:
            numeric_columns = self.dataset.select_dtypes(include=['float64', 'int64']).columns
            for column in numeric_columns:
                Q1 = self.dataset[column].quantile(0.25)
                Q3 = self.dataset[column].quantile(0.75)
                IQR = Q3 - Q1
                self.dataset = self.dataset[(self.dataset[column] >= (Q1 - 1.5 * IQR)) & (self.dataset[column] <= (Q3 + 1.5 * IQR))]
        else:
            numeric_columns = self.dataset.select_dtypes(include=['float64', 'int64'])
            z_scores = np.abs(stats.zscore(numeric_columns))
            threshold = 3
            self.dataset = self.dataset[(z_scores < threshold).all(axis=1)]

    def after_clean_outliers(self):
        numeric_columns = self.dataset.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:
            plt.boxplot(self.dataset[column])
            plt.title(f"Boxplot for {column}")
            plt.show()

    def split_data(self):
        print(self.dataset)
        target_data = str(input("Which Column Should Be Target Column: "))
        X = self.dataset.drop(f'{target_data}', axis=1)
        y = self.dataset[target_data]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
                

    def scaling(self):
        scaling_type = int(input("Which type of scaling you want?\n1)StandardScaler\n2)MinMaxScaler\n3)RobustScaler\n"))
        if scaling_type == 1:
            scaler = StandardScaler()
            self.X_train_scaled = scaler.fit_transform(self.X_train)
            self.X_test_scaled = scaler.transform(self.X_test)
            print("X_train_scaled\n")
            print(pd.DataFrame(self.X_train_scaled))
        elif scaling_type == 2:
            scaler = MinMaxScaler(feature_range=(0,1))
            self.X_train_scaled = scaler.fit_transform(self.X_train)
            self.X_test_scaled = scaler.transform(self.X_test)
            print("X_train_scaled\n")
            print(pd.DataFrame(self.X_train_scaled))
        else:
            scaler = RobustScaler()
            self.X_train_scaled = scaler.fit_transform(self.X_train)
            self.X_test_scaled = scaler.transform(self.X_test)
            print("X_train_scaled\n")
            print(pd.DataFrame(self.X_train_scaled))

    def explain_with_shap(self, model):
        try:
            explainer = shap.Explainer(model, self.X_train_scaled)
            shap_values = explainer(self.X_test_scaled)

            print("SHAP Feature Importance (Bar Plot)")
            shap.summary_plot(shap_values, self.X_test, plot_type="bar")

            print("SHAP Summary Plot")
            shap.summary_plot(shap_values, self.X_test)
        except Exception as e:
            print("SHAP açıklama sırasında hata oluştu:", e)

    def anomaly_detection(self):
        model = IsolationForest(contamination=0.5)
        model.fit(self.dataset)  # Tüm veri setini kullanıyoruz
        predictions = model.predict(self.dataset)
        
        # Görselleştirme (sadece ilk 2 özelliği kullanıyoruz görselleştirme için)
        plt.scatter(self.dataset.iloc[:, 0], self.dataset.iloc[:, 1], c=predictions, cmap='coolwarm')
        plt.title("Isolation Forest Anomali Tespiti")
        plt.xlabel(self.dataset.columns[0])  # İlk özelliği göster
        plt.ylabel(self.dataset.columns[1])  # İkinci özelliği göster
        plt.show()


    def select_model(self):
        selection = int(input("Which type of model you want:\n1)Tree\n2)Regression\n3)NeuralNetwork\n4)Classifier\n"))
        if selection == 1:
            tree_types = int(input("Which type of tree you want:\n1)DecisionTreeRegression\n2)RandomForestRegressor\n3)XGBoost\n"))
            if tree_types == 1:
                dtr = DecisionTreeRegressor()
                dtr.fit(self.X_train_scaled, self.y_train)
                train_score = dtr.score(self.X_train_scaled, self.y_train)
                test_score = dtr.score(self.X_test_scaled, self.y_test)
                y_pred = dtr.predict(self.X_test_scaled)
                print(f"Train Score: {train_score}")
                print(f"Test Score: {test_score}")
                print("MSE: ", mean_squared_error(self.y_test,y_pred))
                print("R2: ", r2_score(self.y_test,y_pred))                
            elif tree_types == 2:
                rfr = RandomForestRegressor()
                rfr.fit(self.X_train_scaled, self.y_train)
                train_score = rfr.score(self.X_train_scaled, self.y_train)
                test_score = rfr.score(self.X_test_scaled, self.y_test)
                print(f"Train Score: {train_score}")
                print(f"Test Score: {test_score}")
            elif tree_types == 3:
                xgb_model = xgb.XGBRegressor(objective='reg:squarederror',random_state=42)
                param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': range(1,10),
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
                }
                grid_search = GridSearchCV(estimator=xgb_model,param_grid=param_grid,cv=5,verbose=2,n_jobs=-1)
                grid_search.fit(self.X_train_scaled,self.y_train)
                print("Best Parameters: ", grid_search.best_params_)
                best_xgb_model = grid_search.best_estimator_
                y_pred = best_xgb_model.predict(self.X_test_scaled)
                self.explain_with_shap(best_xgb_model)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                plt.figure(figsize=(8, 6))
                plt.scatter(self.y_test, y_pred, color='blue')
                plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', linewidth=2)
                plt.title('Predicted vs Actual')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.show()
        elif selection == 2:
            regression_types = int(input("Which type of regression you want:\n1)Logistic\n2)Linear\n3)CatBoostRegressor"))
            if regression_types == 1:
                lr = LogisticRegression()
                lr.fit(self.X_train_scaled, self.y_train)
                train_score = lr.score(self.X_train_scaled, self.y_train)
                test_score = lr.score(self.X_test_scaled, self.y_test)
                print(f"Train Score: {train_score}")
                print(f"Test Score: {test_score}")
            elif regression_types == 2:
                linr = LinearRegression()
                linr.fit(self.X_train_scaled, self.y_train)
                train_score = linr.score(self.X_train_scaled, self.y_train)
                test_score = linr.score(self.X_test_scaled, self.y_test)
                print(f"Train Score: {train_score}")
                print(f"Test Score: {test_score}")
            elif regression_types == 3:
                model = CatBoostRegressor(iterations=500,
                                        learning_rate=0.1,
                                        depth=6,
                                        eval_metric="RMSE",
                                        verbose=100,
                                        random_state=42)
                model.fit(self.X_train_scaled,self.y_train,eval_set=(self.X_test_scaled,self.y_test),use_best_model=True)
                y_pred = model.predict(self.X_test_scaled)
                print("CatBoost RMSE:", np.sqrt(mean_squared_error(self.y_test, y_pred)))
                self.explain_with_shap(model)
        elif selection == 3:
            self.mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu')  # 'self.mlp' olarak kaydedildi
            self.mlp.fit(self.X_train_scaled, self.y_train)
            y_pred = self.mlp.predict(self.X_test_scaled)
            print(f"Train Score: {self.mlp.score(self.X_train_scaled, self.y_train)}")
            print(f"Test Score: {self.mlp.score(self.X_test_scaled, self.y_test)}")
            print("MSE: ", mean_squared_error(self.y_test, y_pred))
            print("R2: ", r2_score(self.y_test, y_pred))

            self.visualize_results(y_pred)
        elif selection == 4:
            cluster_type = int(input("Which type of cluster you want:\n1)KMEANS\n2)KNN\n3)SVM\n4)DBSCAN\n5)CatBoostClassifier"))
            if cluster_type == 1:
                #Dirsek degeri ile n_cluster bulmak
                wcss = []
                for i in range(1,11):
                    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
                    kmeans.fit(self.X_train_scaled)
                    wcss.append(kmeans.inertia_)

                plt.plot(range(1, 11), wcss)
                plt.title('Elbow Method')
                plt.xlabel('Küme Sayısı (k)')
                plt.ylabel('WCSS')
                plt.show()

                while True:
                    try:
                        self.n_cluster_value = int(input("n_cluster_value (or type 'q' to quit): "))
                        kmeans = KMeans(n_clusters=self.n_cluster_value, random_state=42)
                        kmeans.fit(self.X_train_scaled)
                        train_score = kmeans.score(self.X_train_scaled)
                        test_score = kmeans.score(self.X_test_scaled)
                        print("Train Score: ", train_score)
                        print("Test Score: ", test_score)

                        y_kmeans = kmeans.predict(self.X_train_scaled)
                        plt.figure(figsize=(8, 6))
                        plt.scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
                        centers = kmeans.cluster_centers_
                        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
                        plt.title('KMeans Kümeleme Sonuçları')
                        plt.xlabel('Özellik 1')
                        plt.ylabel('Özellik 2')
                        plt.show()

                    except ValueError:
                        user_input = input("You entered a non-numeric value. Type 'q' to quit or retry: ")
                        if user_input == 'q':
                            break
            elif cluster_type == 2:
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(self.X_train_scaled,self.y_train)
                score = knn.score(self.X_test_scaled,self.y_test)
                print("KNN SCORE: ",score)
                y_pred = knn.predict(self.X_test_scaled)
                accuracy = accuracy_score(self.y_test,y_pred)

                conf_matrix = confusion_matrix(self.y_test,y_pred)
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Tahmin Edilen')
                plt.ylabel('Gerçek')
                plt.title('Confusion Matrix')
                plt.show()
            elif cluster_type == 3:
                kernels = ['rbf', 'linear', 'poly']
                for kernel in kernels:
                    svm = SVC(kernel=kernel)
                    svm.fit(self.X_train_scaled, self.y_train)
                    y_pred = svm.predict(self.X_test_scaled)  # self.X_test yerine self.X_test_scaled olmalı
                    print(f"Accuracy for {kernel} kernel: ", accuracy_score(self.y_test, y_pred))
                
                for kernel in kernels:
                    pca = PCA(n_components = 2)
                    X_train_2d = pca.fit_transform(self.X_train_scaled)
                    X_test_2d = pca.transform(self.X_test_scaled)
                    svm = SVC(kernel=kernel)
                    svm.fit(X_train_2d,self.y_train)
                    y_pred = svm.predict(X_test_2d)
                    accuracy = accuracy_score(self.y_test,y_pred)
                    print(f"Accuracy for rbf {kernel}: {accuracy}")
                    
                    plt.figure(figsize=(12, 6))
                    plot_decision_regions(X=X_train_2d, y=self.y_train.values, clf=svm, legend=2)
                    plt.title(f'SVM with RBF Kernel (2D PCA)')
                    plt.xlabel('Principal Component 1')
                    plt.ylabel('Principal Component 2')
                    plt.tight_layout()
                    plt.show()
                    """
                    if self.X_train_scaled.shape[1] == 2:  # Eğer sadece 2 özellik varsa
                        plt.figure(figsize=(12, 6))
                        plt.subplot(1, 2, 1)
                        plot_decision_regions(X=self.X_train_scaled, y=self.y_train, clf=svm, legend=2)
                        plt.title(f'SVM with {kernel.capitalize()} Kernel')
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                        plt.tight_layout()
                        plt.show()
                    """
            elif cluster_type == 4:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(self.X_train)  # self.X üzerinde normalizasyon yapılıyor
                
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                y_dbscan = dbscan.fit_predict(X_scaled)
                
                plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap='viridis')
                plt.title("DBSCAN Clustering")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.show()
            elif cluster_type == 5:
                model = CatBoostClassifier(iterations=1000,learning_rate=0.1,depth=6,verbose=1,random_state=42)
                model.fit(self.X_train_scaled, self.y_train, eval_set=(self.X_test_scaled, self.y_test), use_best_model=True)
                y_pred = model.predict(self.X_test_scaled)
                print("CatBoost RMSE:", np.sqrt(mean_squared_error(self.y_test, y_pred)))
                self.explain_with_shap(model)
            else:
                print(f"Cannot plot decision region for {kernel} kernel because the data is not 2D.")



    def visualize_results(self, y_pred):
        # Gerçek değerler ve tahminler arasındaki farkı görselleştirme
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, y_pred)
        plt.title("Gerçek vs Tahmin Edilen")
        plt.xlabel("Gerçek Değerler")
        plt.ylabel("Tahmin Edilen Değerler")
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color='red', linewidth=2)

        # Kayıp (Loss) grafiği
        plt.subplot(1, 2, 2)
        plt.plot(self.mlp.loss_curve_)  # 'self.mlp' kullanılıyor
        plt.title("Model Kayıp Grafiği")
        plt.xlabel("Eğitim Iterasyonu")
        plt.ylabel("Kayıp")

        plt.tight_layout()
        plt.show()


# Kullanım
buddy = Buddy()
buddy.run()
