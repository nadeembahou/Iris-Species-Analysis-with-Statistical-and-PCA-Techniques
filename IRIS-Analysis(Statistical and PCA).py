import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis

print("\n#1A: statistics of each feature and class using the test statistics:\n ")

# Load the data from the CSV file
iris_data = pd.read_csv("iris3.csv")

# Define the class labels
class_labels = ['setosa', 'versicolor', 'virginica']

# Test Statistics Functions
def calculate_minimum(x):
    return np.min(x)

def calculate_maximum(x):
    return np.max(x)

def calculate_mean(x):
    return np.mean(x)

def calculate_trimmed_mean(x, p):
    n = len(x)
    trim_size = int(n * p)
    trimmed_data = np.sort(x)[trim_size:-trim_size]
    return np.mean(trimmed_data)

def calculate_standard_deviation(x):
    return np.std(x)

def calculate_skewness(x):
    return skew(x)

def calculate_kurtosis(x):
    return kurtosis(x)

# Analysis Function
class_labels_column = 'species'

def feature_class_analysis(data, feature_name, class_labels):
    grouped_data = data.groupby(class_labels_column)

    results = {'Feature': feature_name}
    
    for label in class_labels:
        class_data = grouped_data.get_group(label)[feature_name]
        results[f'{label.capitalize()} Minimum'] = calculate_minimum(class_data)
        results[f'{label.capitalize()} Maximum'] = calculate_maximum(class_data)
        results[f'{label.capitalize()} Mean'] = calculate_mean(class_data)
        results[f'{label.capitalize()} Trimmed Mean (10%)'] = calculate_trimmed_mean(class_data, 0.1)
        results[f'{label.capitalize()} Standard Deviation'] = calculate_standard_deviation(class_data)
        results[f'{label.capitalize()} Skewness'] = calculate_skewness(class_data)
        results[f'{label.capitalize()} Kurtosis'] = calculate_kurtosis(class_data)

    return results

# Perform Analysis
analysis_results = []
for feature_name in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    analysis_results.append(feature_class_analysis(iris_data, feature_name, class_labels))

# Display the results in a table
result_df = pd.DataFrame(analysis_results)

# Transpose the DataFrame
result_df = result_df.transpose()

# Print the transposed table
print(result_df)


print("\n1B - Analysis, explanations & Conclusions: \n")

print(
    """Sepal Length Analysis:

Range: Setosa: 4.3-5.8, Versicolor: 4.9-7.0, Virginica: 4.9-7.9.               
Mean: Setosa: ~5.006, Versicolor: ~5.936, Virginica: ~6.588.                     
Trimmed Mean (10%): Setosa: ~5.0025, Versicolor: ~5.9375, Virginica: ~6.5725. 
Standard Deviation: Setosa: ~0.349, Versicolor: ~0.511, Virginica: ~0.629.
Skewness: Setosa: +0.116, Versicolor: +0.102, Virginica: +0.114.
Kurtosis: Setosa: -0.346, Versicolor: -0.599, Virginica: -0.088.

Sepal Width Analysis:

Range: Setosa: 2.3-4.4, Versicolor: 2.0-3.4, Virginica: 2.2-3.8.
Mean: Setosa: ~3.418, Versicolor: ~2.77, Virginica: ~2.974.
Trimmed Mean (10%): Setosa: ~3.4025, Versicolor: ~2.78, Virginica: ~2.9625.
Standard Deviation: Setosa: ~0.377, Versicolor: ~0.311, Virginica: ~0.319.
Skewness: Setosa: +0.104, Versicolor: -0.352, Virginica: +0.355.
Kurtosis: Setosa: +0.685, Versicolor: -0.448, Virginica: +0.520.
Petal Length Analysis:

Petal Length Analysis:

Range: Setosa: 1.0-1.9, Versicolor: 3.0-5.1, Virginica: 4.5-6.9.
Mean: Setosa: ~1.464, Versicolor: ~4.26, Virginica: ~5.552.
Trimmed Mean (10%): Setosa: ~1.4625, Versicolor: ~4.2925, Virginica: ~5.51.
Standard Deviation: Setosa: ~0.1718, Versicolor: ~0.4652, Virginica: ~0.5463.
Skewness: Setosa: +0.0697, Versicolor: -0.5882, Virginica: +0.5328.
Kurtosis: Setosa: +0.8137, Versicolor: -0.0744, Virginica: -0.2565.

Petal Width Analysis:

Range: Setosa: 0.1-0.6, Versicolor: 1.0-1.8, Virginica: 1.4-2.5.
Mean: Setosa: ~0.244, Versicolor: ~1.326, Virginica: ~2.026.
Trimmed Mean (10%): Setosa: ~0.235, Versicolor: ~1.325, Virginica: ~2.0325.
Standard Deviation: Setosa: ~0.1061, Versicolor: ~0.1958, Virginica: ~0.2719.
Skewness: Setosa: +0.1610, Versicolor: -0.0302, Virginica: -0.1256.
Kurtosis: Setosa: +0.6851, Versicolor: -0.4878, Virginica: -0.6613.
""")

print("""
Summary:

1. Sepal Length:
   - Virginica generally has longer sepals than Versicolor and Setosa.
   - Setosa shows the least variability in sepal length.
   - All species exhibit a slight right skew in sepal length distributions.

2. Sepal Width:
   - Setosa tends to have wider sepals compared to Versicolor and Virginica.
   - Setosa's sepal width distribution is the most variable.
   - Versicolor's sepal width distribution is slightly left-skewed.

3. Petal Length:
   - Virginica typically has the longest petals among the three species.
   - Setosa's petal length distribution is the most peaked.
   - All species exhibit a slight right skew in petal length.

4. Petal Width:
   - Virginica generally has wider petals compared to Versicolor and Setosa.
   - Virginica's petal width distribution shows the highest variability.
   - Setosa's petal width distribution is the most positively skewed.
      
In conclusion, Setosa exhibits greater variability in both sepal length and width compared to the other two species. Additionally, its petal length and width distributions tend to be more 
peaked and skewed to the right, indicating a more distinct morphology. On the other hand, Versicolor and Virginica show more similar distributions in terms of skewness and kurtosis, 
suggesting a closer resemblance in their morphological features. However, slight differences in mean and standard deviation indicate subtle distinctions between the two species.
""")

#2a
print("\n2a - Visually see two sets of features and the class they belong to --> Printed") 
# Load the Iris dataset
iris_data = pd.read_csv('iris3.csv')

sns.set(style="whitegrid")

# Create a scatter plot for Sepal Length vs Sepal Width
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris_data)
plt.title('Sepal Length vs Sepal Width')
plt.savefig('sepal_length_vs_sepal_width.png')  
plt.show()

# Create a scatter plot for Petal Length vs Petal Width
plt.figure(figsize=(10, 6))
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=iris_data)
plt.title('Petal Length vs Petal Width')
plt.savefig('petal_length_vs_petal_width.png') 
plt.show()

# 2b part i: Algorithm (pseudocode) to sort the four features in the dataset
print("""\n2b - part B-i: An algorithm (pseudocode) to sort the four features in the dataset.:

1. Input: Dataset with features (sepal_length, sepal_width, petal_length, petal_width)
2. For each feature in the dataset:
    a. Initialize an empty list to store the values of the current feature.
    b. Iterate through each row in the dataset and append the value of the current feature to the list.
    c. Sort the list in ascending order using a suitable sorting algorithm --> Merge Sort.
    d. Update the column in the dataset with the sorted values.
3. Output: Dataset with each feature sorted.""")

# 2b part ii: Time Complexity Analysis
print("\n2b part ii. Time Complexity Analysis:")
print("""\nThe algorithm utilizes Merge Sort. It has an average-case time complexity of O(n log n), 
   making it suitable for handling large datasets. Sorting each of the four features
  independently results in a total time complexity of O(n log n) for the entire dataset, 
  showcasing the algorithm's effectiveness in handling sorting operations efficiently.\n""")

# 2b part iii: Implementing the algorithm - no built in function

# Load the dataset
new_dataset = pd.read_csv('iris3.csv')

# Function to perform merge sort
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    return merge(left_half, right_half)

# Function to merge two sorted arrays
def merge(left, right):
    result = []
    left_index, right_index = 0, 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1
    result.extend(left[left_index:])
    result.extend(right[right_index:])
    return result

# Function to sort each column separately
def sort_dataset(dataset):
    sorted_dataset = dataset.copy()
    for column in sorted_dataset.columns:
        sorted_dataset[column] = merge_sort(sorted_dataset[column].values)
    return sorted_dataset

# Sort the dataset
sorted_dataset = sort_dataset(new_dataset)

# Display the sorted dataset
print("\nSorted Dataset:\n")
print(sorted_dataset)


print("\n2B-IV. Determine if any of the four features can separate the three plant species:")

print(""" Visually inspecting the data, it appears that the "petal_length" feature has the potential
to separate the three plant species to some extent. This is because the range of petal lengths for
setosa flowers is noticeably different from the ranges for versicolor and virginica. However, there
is some overlap between versicolor and virginica, indicating that petal length alone may not be
sufficient for perfect separation. The other features (sepal_length, sepal_width, and petal_width)
show considerable overlap among the species, making them less effective for species separation.""")

print("""\n2B-V. Provide an explanation of the results:

# A. Among the four features, petal length exhibits the most distinct distribution among the three
# plant species. Setosa flowers tend to have shorter petal lengths compared to versicolor and virginica,
# which have longer and more varied petal lengths. This feature's ability to partially separate the species
# is due to its significant differences in range and distribution across the species. However, overlap
# between versicolor and virginica suggests that petal length alone may not be entirely reliable for
# species separation.

# B. The metric used to determine separation was visual inspection of the data distribution for each feature
# across the three species. This method allows for qualitative assessment of the potential of each feature
# to separate the species based on their distributions. Visual inspection is chosen for its simplicity and
# ease of interpretation, providing a quick overview of feature distributions and their potential for separation.\n""")

#part 2c: for each feature in the dataset:

print("\n2c: Algorithm to normalize the Iris data by feature:")
def normalize_feature(feature_values):
    min_val = np.min(feature_values)
    max_val = np.max(feature_values)
    normalized_values = (feature_values - min_val) / (max_val - min_val)
    return normalized_values

# Print a message before normalization
print("Before normalization:")
print(iris_data.head())

# Normalize each feature in the dataset
for feature_name in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    iris_data[feature_name] = normalize_feature(iris_data[feature_name])

# Print a message after normalization
print("\nAfter normalization:")
print(iris_data.head())

#2d - i
print("""\n 2D- i - For each class (species) in the dataset:
a. Calculate the mean of each feature (sepal length, sepal width, petal length, petal width) for that class.
b. For each observation in the class:
      i. Calculate the Euclidean distance between the observation and the mean of the class.
c. Sort the observations in the class based on their distances from the mean in descending order.
d. Remove the furthest observation from the mean (outlier).
e. Repeat steps b-d until the desired number of outliers are removed or until convergence criteria are met.\n""")

#2d - ii
print("""\n 2d - ii n = no. of observations in the dataset, m =  no. of features, and k = no .of classes.
mean for each class: O(n * m * k)  / Distance for each observation: O(n * m * k) / Sorting observations by distance: O(n * log(n))

Total Running Time: O(n * m * k + n * log(n))
Assumption: Based on the small dataset and the no. of classes is small compared to the number of observations.\n""")

#2d - iii
print("\n 2d- iii: Class created!\n")

class OutlierRemoval:
    def __init__(self):
        self.data = None
    
    def initialize_data(self, input_data):
        self.data = input_data
    
    def remove_outliers(self, threshold=2):
        if self.data is None:
            raise ValueError("No data initialized. Please initialize data first.")
        
        cleaned_data = pd.DataFrame()
        outliers_list = []  # Initialize an empty list to store outliers
        
        # Group data by species
        grouped_data = self.data.groupby('species')
        
        # Iterate over groups
        for species, group in grouped_data:
            cleaned_group = group.copy()
            
            # Calculate z-score for each feature
            z_scores = np.abs((group.iloc[:, :-1] - group.iloc[:, :-1].mean()) / group.iloc[:, :-1].std())
            
            # Identify outliers based on threshold
            outliers_mask = z_scores > threshold
            
            # Append outliers to the list
            outliers_list.append(group[outliers_mask.any(axis=1)].copy())
            
            # Remove outliers
            cleaned_group = cleaned_group[~outliers_mask.any(axis=1)]
            
            # Append cleaned group to final cleaned data
            cleaned_data = pd.concat([cleaned_data, cleaned_group])
        
        # Convert outliers list to DataFrame
        if len(outliers_list) > 0:
            outliers = pd.concat(outliers_list)
        else:
            outliers = pd.DataFrame()
        
        return cleaned_data, outliers

# Use the OutlierRemoval class to remove outliers
outlier_remover = OutlierRemoval()
outlier_remover.initialize_data(iris_data)
cleaned_data, outliers = outlier_remover.remove_outliers()

#2d - iv
print("\n 2d- iv: The cleaned data without the outliers has been plotted\n")

# Function to plot each class individually
def plot_class_outliers(outlier_remover):
    # Get the cleaned data from the outlier remover
    cleaned_data = outlier_remover.data
    
    # Get the list of unique classes (species)
    classes = cleaned_data['species'].unique()
    
    # Iterate over each class
    for class_name in classes:
        class_data = cleaned_data[cleaned_data['species'] == class_name]
        features = class_data.columns[:-1] 
        
        # Plot each combination of two features
        plotted_combinations = set()  # Track plotted combinations to avoid duplicates
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                # Check if the combination has been plotted already
                combination = (features[i], features[j])
                if combination in plotted_combinations:
                    continue  # Skip plotting if the combination has been plotted
                else:
                    plotted_combinations.add(combination)
                
                # Create a scatter plot for the current feature combination
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=features[i], y=features[j], hue='species', data=class_data)
                plt.title(f'{features[i]} vs {features[j]} for {class_name}')
                plt.xlabel(features[i])
                plt.ylabel(features[j])
                plt.legend()
                plt.show()
                print(f'Plotted: {features[i]} vs {features[j]} for {class_name}')  # Print the plotted combination

# Call the function to plot cleaned data without outliers
plot_class_outliers(outlier_remover)

# Analyze the results
print("\nCleaned Data:")
print(cleaned_data.head())

print("\nOutliers:")
print(outliers.head())

#2D-V -A & B
print("\n2D-V- parts A & B:")

print("""\n A - Yes, there were outliers detected in the dataset. The outliers were identified using the z-score method, which calculates the deviation of
each data point from the mean in terms of standard deviations. If a data point's z-score exceeds a certain threshold (typically 2 or 3 
standard deviations), it is considered an outlier. In this case, outliers were determined by comparing the z-scores of each feature within 
each class to a predefined threshold. Outliers were then removed based on this criterion.

B- The z-score method was chosen for outlier detection because it provides a standardized way to detect outliers across different features 
and allows for comparison of the magnitude of deviations regardless of the scale of the feature. Additionally, it offers a more objective 
and standardized approach compared to visual inspection, ensuring consistency and reliability in identifying outliers across various 
datasets and analysis scenarios.\n""")

#2E-i
print("\n 2E - i: Rank four features in dataset - Pseudocode:")

print("""Algorithm RankFeatures(IrisDataset):
    Initialize an empty dictionary FeatureScores
    
    For each feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
        Calculate the variability of the feature across different species
        
        Store the feature score in the FeatureScores dictionary
        
    Rank the features based on their scores in descending order
    
    Return the ranked features \n """)

#2E-ii
print("\n2E - ii: Total and running time of algorithm in O-notation andT (n). :")

print(""" 
    - Calculating variability for each feature across species: Time complexity of O(n), where n = data points.
    - Storing feature scores in a dictionary: Constant time complexity, denoted as O(1).
    - Ranking features: Constant time complexity as well, O(1).
    - Total running time complexity: Mostly from calculating variablity so most likely O(n).\n""")

#2E-iii
print("2E - iii: Implement the design, with a class for future use:")

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
class FeatureRanker:
    def __init__(self, iris_dataset):
        self.iris_dataset = iris_dataset
    
    def rank_features(self):
        feature_scores = {}
        
        for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
            # Calculate variability of the feature across different species
            variability = self.calculate_variability(feature)
            feature_scores[feature] = variability
        
        # Rank features based on their scores
        ranked_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_features
    
    def calculate_variability(self, feature):
        # Calculate the variability of the feature across different species
        variability = self.iris_dataset.groupby('species')[feature].std().mean()
        return variability

#Calling in the function to rank features. 
ranker = FeatureRanker(iris_data)
ranked_features = ranker.rank_features()

# Printing each feature and its score on a separate line
print("\nRanked features:")
for feature, score in ranked_features:
    print(f"{feature}: {score}")

#2E-iv
print("\n2E - iv: Determine if any of the four features can separate the three plant types: \n")
# Visualize data with scatter plots
sns.pairplot(iris_data, hue='species', diag_kind='hist', markers=["o", "s", "D"])
plt.show()

print("""After inspecting the plots, more particulary on the diagonals, the features that most effectively separates the three plant 
types is petal length and petal width. These features exhibit distinct clustering for each plant species when plotted.""")

#2E-V:

print("\n2E - V - A: Was there any feature that could separate the data by plant species; if so why, if not why not? ")

print("""Petal length and petal width could effectively separate the data by plant species(not 100 percent fully but almost), they have 
significantly different values across the three species, resulting in clear separation when visualized.\n""")

print("\n2E - V - B: If a feature could not separate the plant types; what conclusion can drawn from this feature?\n")

print("""If a feature could not effectively separate the plant types, it implies that this feature does not exhibit significant differences 
across the species to reliably distinguish between them.\n""")

print("\n2E - V - C: Can a metric be developed to complement the ranking method? Explain why or why not: \n")

print("""Yes, we could create an additional metric to enhance the ranking method. This metric would assess how much the distributions of 
each feature overlap between different species. It could help identify the most useful features for classifying plant species. \n""")

#2F
print("\n#2F")
print("# Separate features and labels")
X = iris_data.drop(columns=['species'])  # Features
y = iris_data['species']  # Labels

print(" i. Use the built-in PCA to perform analysis of the Iris data set using all species (classes):\n")
pca_all = PCA()
X_pca_all = pca_all.fit_transform(X)

print("ii. Using the built-in PCA to perform analysis of the Iris data set by specie (class)")
pca_by_class = {}

for species in y.unique():
    X_species = X[y == species]
    pca_species = PCA()
    X_pca_species = pca_species.fit_transform(X_species)
    pca_by_class[species] = {
        'pca': pca_species,
        'X_pca': X_pca_species,
        'explained_variance_ratio': pca_species.explained_variance_ratio_
    }

# iii. Provide an explanation of the results:
print("""\nF-iii - A. What is the difference between using all the data and using the data by specie (class)?
When utilizing all the data, PCA aims to identify principal components that capture the maximum variance across all species combined.
On the other hand, when analyzing the data by species, PCA focuses on finding principal components that explain the variance within 
each species separately. This distinction allows for a more nuanced understanding of the unique patterns and variations within 
individual species, which may not be apparent when considering all species together.""")

# Print the percentage explained variance for each principal component for all species
print("\nF- iii - B. Percentage explained variance for each principal component (all species):")
for i, ratio in enumerate(pca_all.explained_variance_ratio_):
    print(f"Principal Component {i + 1}: {ratio * 100:.2f}%")

print("""\nPrincipal Component 1 captures the majority of the variance, indicating that it is the most significant component in reducing
the dimensionality of the data.\n""")

# Print the percentage explained variance for each principal component for each species
print("\nPercentage explained variance for each principal component by species:")
for species, data in pca_by_class.items():
    print(f"\nSpecies: {species}")
    explained_variance_ratio = data['explained_variance_ratio']
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"Principal Component {i + 1}: {ratio * 100:.2f}%")       

print("""\nThe percentage explained for each principal component highlights how much of the species-specific variability is captured 
by each component. For setosa, PC1 dominates with 83.95%, followed by PC2 at 9.18%. Versicolor sees PC1 at 72.36% and PC2 at 
18.11%, while virginica shows PC1 at 65.36% and PC2 at 22.56%. These results represent the importance of the first principal 
component in capturing the majority of species-specific variation, with subsequent components contributing to a lesser extent.\n""")

print("""\n2F - iii - C. How many principal components should you keep?
Based on the results I keep the first two principal components as they capture the majority of the variance for all species. For 
setosa, PC1 and PC2 together explain around 93.13% of the variance. Similarly, for versicolor and virginica, PC1 and PC2 account for 
approximately 90.47% and 87.92% of the variance, respectively. Retaining these two principal components would preserve most of the 
important information while reducing the dimensionality of the data.\n""")

# Visualize the explained variance ratio for all data
plt.plot(np.cumsum(pca_all.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Ratio for All Data')
plt.grid(True)
plt.show()


 
