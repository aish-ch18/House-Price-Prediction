# Installation

To run this project, you need Python 3.x and the following libraries:

- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib
- seaborn

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn

After installing the dependencies, clone this repository:

git clone https://github.com/your-username/california-housing-prediction.git
cd california-housing-prediction


---

### 4. Usage

**`USAGE.md`**

```markdown
# Usage

To run the project, execute the main script:

```bash
python main.py

This script will perform the following steps:

Load and preprocess the California Housing dataset.
Visualize the dataset using pair plots.
Train a Linear Regression model.
Define, build, and train a Neural Network model using a custom KerasRegressor wrapper.
Evaluate the models and compare their performance.
Tune the hyperparameters of the Neural Network model using GridSearchCV.
Visualize the results of the hyperparameter tuning.
