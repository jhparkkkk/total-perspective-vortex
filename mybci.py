

if __name__ == "__main__":
    # Load the data
    data = load_data()
    # Preprocess the data
    data = preprocess_data(data)
    # Train the model
    model = train_model(data)
    # Evaluate the model
    evaluate_model(model, data)
