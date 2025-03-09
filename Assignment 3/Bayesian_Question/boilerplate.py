#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    
    train_data = pd.read_csv("train_data.csv")
    val_data = pd.read_csv("validation_data.csv")
    return train_data, val_data

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    
    edges = []
    features = df.columns.tolist()
    # Defining the DAG as a list of directed edges (covering all 6C2 combinations)
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            edges.append((features[i], features[j]))
    
    dag = bn.make_DAG(edges)

    bn.plot(dag)
        
    model = bn.parameter_learning.fit(dag, df)
    
    return model

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""

    edges = []
    features = df.columns.tolist()
    # Defining the DAG as a list of directed edges (covering all 6C2 combinations)
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            edges.append((features[i], features[j]))

    # prune these edges
    redundant_edges = [('Start_Stop_Id', 'End_Stop_ID'), ('Start_Stop_ID', 'Fare_Category'), ('Start_Stop_ID', 'Route_Type'),
                    ('End_Stop_ID', 'Route_Type'), ('End_Stop_ID', 'Fare_Category'), ('Distance', 'Zones_Crossed')]
    
    # remove these edges from the original edges
    pruned_edges = [edge for edge in edges if edge not in redundant_edges]

    # create a new DAG with the pruned edges
    pruned_dag = bn.make_DAG(pruned_edges)

    # bn.plot(pruned_dag)

    pruned_model = bn.parameter_learning.fit(pruned_dag, df)

    return pruned_model

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    
    # Created an initial network (same as Task 1)
    edges = []
    features = df.columns.tolist()
    # Defining the DAG as a list of directed edges (covering all 6C2 combinations)
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            edges.append((features[i], features[j]))
    
    dag = bn.make_DAG(edges)

    # Hill Climbing optimization to improve the structure
    optimized_dag = bn.structure_learning.fit(df, methodtype='hc')

    # bn.plot(optimized_dag)

    optimized_model = bn.parameter_learning.fit(optimized_dag, df)

    return optimized_model

def save_model(fname, model):
    """Save the model to a file using pickle."""

    pkl_file = open(fname, 'wb')
    pickle.dump(model, pkl_file)
    pkl_file.close()
  
def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)

    # # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

