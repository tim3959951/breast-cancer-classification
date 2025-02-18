import data_processing
import model_training
import model_evaluation
import shap_analysis

if __name__ == "__main__":
    print("Starting Breast Cancer Classification Pipeline...")
    
    # Run each step
    data_processing
    model_training
    model_evaluation
    shap_analysis
    
    print("Pipeline completed. Check the visualizations and results!")
