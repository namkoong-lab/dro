def list_functions():
    functions_metadata = [
        {
            'name': 'classification_basic',
            'description': 'Generate basic classification data where each class is sampled from a different center, with the data points spread radially from the center.',
            'parameters': [
                ('d', 'int', 'Dimension of the covariates (default: 2)'),
                ('k', 'int', 'Number of classes (default: 2)'),
                ('num_samples', 'int', 'Total number of samples (default: 500)'),
                ('radius', 'float', 'Radius of the ball to sample the class center (default: 5.0)'),
                ('seed', 'int', 'Random seed (default: 42)'),
                ('visualize', 'bool', 'Whether to visualize the data (default: False)'),
                ('save_dir', 'str', 'The save path of the visualization figure (default: "./visualization.png")')
            ],
            'returns': 'X and y as numpy arrays, where X is the covariate data and y is the target data'
        },
        {
            'name': 'classification_DN21',
            'description': 'Generate classification data using a linear decision boundary with label flipping as per the DN21 setting from the paper "Learning Models with Uniform Performance via Distributionally Robust Optimization".',
            'parameters': [
                ('d', 'int', 'Dimension of the covariates'),
                ('flip_ratio', 'float', 'The ratio of labels to be flipped (default: 0.1)'),
                ('num_samples', 'int', 'Total number of samples (default: 100)'),
                ('seed', 'int', 'Random seed (default: 42)'),
                ('visualize', 'bool', 'Whether to visualize the data (default: False)'),
                ('save_dir', 'str', 'The save path of the visualization figure (default: "./visualization.png")')
            ],
            'returns': 'X (covariate data) and y_noisy (target data with noise from flipped labels) as numpy arrays'
        },
        {
            'name': 'classification_SNVD20',
            'description': 'Generate classification data by applying a filter based on the Euclidean norm of points sampled from a 2D normal distribution, with additional filtering as per the SNVD20 setting from the paper "Certifying Some Distributional Robustness with Principled Adversarial Training".',
            'parameters': [
                ('num_samples', 'int', 'Total number of samples (default: 500)'),
                ('seed', 'int', 'Random seed (default: 42)'),
                ('visualize', 'bool', 'Whether to visualize the data (default: False)'),
                ('save_dir', 'str', 'The save path of the visualization figure (default: "./visualization.png")')
            ],
            'returns': 'X_filtered (filtered covariate data) and y_filtered (filtered target data) as numpy arrays'
        },
        {
            'name': 'classification_LWLC',
            'description': 'Generate high-dimensional classification data with features scrambled and a Gaussian distribution applied to the feature set, based on the LWLC model from the paper "Distributionally Robust Optimization with Data Geometry".',
            'parameters': [
                ('num_samples', 'int', 'Total number of samples (default: 10000)'),
                ('d', 'int', 'Dimension of feature sets S and V (default: 5)'),
                ('bias', 'float', 'Bias ratio for class labels (default: 0.5)'),
                ('scramble', 'bool', 'Whether to scramble features (default: True)'),
                ('sigma_s', 'float', 'Variance of the Gaussian distribution for feature S (default: 3.0)'),
                ('sigma_v', 'float', 'Variance of the Gaussian distribution for feature V (default: 0.3)'),
                ('high_dimension', 'int', 'The final dimension of X (default: 300)'),
                ('seed', 'int', 'Random seed (default: 42)'),
                ('visualize', 'bool', 'Whether to visualize the data (default: False)'),
                ('save_dir', 'str', 'The save path of the visualization figure (default: "./visualization.png")')
            ],
            'returns': 'X (high-dimensional covariate data) and y (target data) as numpy arrays'
        },
        {
            'name': 'regression_basic',
            'description': 'Generates a basic regression dataset with a specified number of samples and dimensions, with Gaussian noise added to the target variable.',
            'parameters': [
                ('num_samples', 'int', 'The number of samples'),
                ('d', 'int', 'The dimension of covariates'),
                ('noise', 'float', 'The variance of the noise term'),
                ('seed', 'int', 'Random seed')
            ],
            'returns': 'X (numpy.ndarray), y (numpy.ndarray): The generated covariate and target data'
        },
        {
            'name': 'regression_DN20_1',
            'description': 'Generates a regression dataset following Section 3.1.2 of "Learning Models with Uniform Performance via Distributionally Robust Optimization", where the target variable has Gaussian noise and a noisy label is added based on the value of the first covariate.',
            'parameters': [
                ('num_samples', 'int', 'The number of samples'),
                ('d', 'int', 'The dimension of covariates'),
                ('noise', 'float', 'The variance of the noise term'),
                ('seed', 'int', 'Random seed')
            ],
            'returns': 'X (numpy.ndarray), y_noisy (numpy.ndarray): The generated covariate and noisy target data'
        },
        {
            'name': 'regression_DN20_2',
            'description': 'Generates a regression dataset following Section 3.1.3 of "Learning Models with Uniform Performance via Distributionally Robust Optimization", with two different linear models based on a minority group ratio and Gaussian noise.',
            'parameters': [
                ('num_samples', 'int', 'The number of samples'),
                ('prob', 'float', 'The minority group ratio'),
                ('noise', 'float', 'The variance of the noise term'),
                ('seed', 'int', 'Random seed')
            ],
            'returns': 'X (numpy.ndarray), y (numpy.ndarray): The generated covariate and target data'
        },
        {
            'name': 'regression_DN20_3',
            'description': 'Generates a regression dataset based on the UCI Communities and Crime dataset, as described in Section 3.3 of "Learning Models with Uniform Performance via Distributionally Robust Optimization", with options to download or load the data.',
            'parameters': [
                ('save_dir', 'str', 'The path to save the data'),
                ('download', 'bool', 'Whether to download the data or load from the saved directory')
            ],
            'returns': 'X (numpy.ndarray), y (numpy.ndarray): The generated covariate and target data'
        },
        {
            'name': 'regression_LWLC',
            'description': 'Generates a regression dataset with controlled spurious correlation, following Section 4.1 of "Distributionally Robust Optimization with Data Geometry", with options to scramble features and control spurious correlations.',
            'parameters': [
                ('n1', 'int', 'The total number of samples in the pool'),
                ('n2', 'int', 'The number of samples required'),
                ('ps', 'int', 'The dimension of feature S'),
                ('pvb', 'int', 'The dimension of feature Vb'),
                ('pv', 'int', 'The dimension of other features in V (except for Vb)'),
                ('r', 'float', 'The adjustment parameter controlling spurious correlation'),
                ('scramble', 'bool', 'Whether to mix S and V')
            ],
            'returns': 'X (numpy.ndarray), y (numpy.ndarray): The generated covariate and target data'
        }
    ]
    
    for idx, func in enumerate(functions_metadata):
        print(f"Data Generation Mechanism {idx+1}: {func['name']}")
        print(f"  Description: {func['description']}")
        print(f"  Parameters:")
        for param in func['parameters']:
            print(f"    - {param[0]} ({param[1]}): {param[2]}")
        print(f"  Returns: {func['returns']}\n")


if __name__ == "__main__":
    list_functions()