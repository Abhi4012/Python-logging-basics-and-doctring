from utils.all_utils import prepare_data, save_plot
from utils.model import perceptron
import pandas as pd


def main(data, modelName, plotName, eta, epochs):
    df_AND = pd.DataFrame(data) 
    X, y = prepare_data(df_AND)


    # Create a Perceptron model for the OR gate with the specified learning rate and epochs.
    model = perceptron(eta=eta, epochs=epochs)

    # Fit (train) the model with the input features (X) and target labels (y) using the defined learning rate and epochs.
    model.fit(X,y)

    # Calculate and display the total loss for the trained model.
    _ = model.total_loss()

    model.save(filename=modelName,model_dir="model")
    save_plot(df_AND, model, filename=plotName )

if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1]
    } 
    ETA = 0.3
    EPOCHS = 10
    main(data=AND, modelName="and.model", plotName="and.png",eta=ETA, epochs=EPOCHS)







