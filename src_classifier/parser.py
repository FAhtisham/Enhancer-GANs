import argparse

def parameter_parser():

	 parser = argparse.ArgumentParser(description = "Tweet Classification")

	 parser.add_argument("--epochs",
								dest = "epochs",
								type = int,
								default = 100,
						 help = "Number of gradient descent iterations. Default is 200.")
     
	 parser.add_argument("--data",
								dest = "data",
								type = str,
								default = "permissive_enhancers",
						 help = "File to be processed")

	 parser.add_argument("--learning_rate",
								dest = "learning_rate",
								type = float,
								default = 0.01,
						 help = "Gradient descent learning rate. Default is 0.01.")

	 parser.add_argument("--hidden_dim",
								dest = "hidden_dim",
								type = int,
								default = 128,
						 help = "Number of neurons by hidden layer. Default is 128.")
						 
	 parser.add_argument("--lstm_layers",
								dest = "lstm_layers",
								type = int,
								default = 5,
					 help = "Number of LSTM layers")
					 
	 parser.add_argument("--batch_size",
									dest = "batch_size",
									type = int,
									default = 256,
							 help = "Batch size")

	 parser.add_argument("--test_size",
								dest = "test_size",
								type = float,
								default = 0.20,
						 help = "Size of test dataset. Default is 10%.")
						 
	 parser.add_argument("--max_len",
								dest = "max_len",
								type = int,
								default = 300,
						 help = "Maximum sequence length per tweet")
						 
	 parser.add_argument("--vocab_size",
								dest = "vocab_size",
								type = float,
								default = 5,
						 help = "Maximum number of words in the dictionary")					 
	 return parser.parse_args()