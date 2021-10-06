import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description = "Enhancer GAN ")
    parser.add_argument("--epochs",
								dest = "epochs",
								type = int,
								default = 100 ,
						 help = "Number of gradient descent iterations..")
    parser.add_argument("--data",
								dest = "data",
								type = str,
								default = "permissive_enhancers",
						 help = "File to be processed")
    parser.add_argument("--learning_rate_d",
								dest = "learning_rate_d",
								type = float,
								default = 0.00001,
						 help = "Gradient descent learning rate. Default is 0.01.")
    parser.add_argument("--learning_rate_g",
								dest = "learning_rate_g",
								type = float,
								default = 0.0001,
						 help = "Gradient descent learning rate. Default is 0.01.")						 
    parser.add_argument("--hidden_dim",
								dest = "hidden_dim",
								type = int,
								default = 128,
						 help = "Number of neurons by hidden layer. Default is 128.")
    
    parser.add_argument("--batch_size",
									dest = "batch_size",
									type = int,
									default = 2048,
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
						 help = "Maximum sequence length per seq")
    parser.add_argument("--vocab_size",
								dest = "vocab_size",
								type = float,
								default = 5,
						 help = "Maximum number of words in the dictionary")	 
    parser.add_argument("--Discriminator steps",
                        dest = 'discriminator_steps',
                        type = float,
                        default = 5,
                        help = " number of Discriminator steps in training")
    parser.add_argument("--Generator steps",
                        dest = 'generator_steps',
                        type = float,
                        default = 1,
                        help = " number of Generator steps in training")

    parser.add_argument("--lambda",
                        dest = "lam",
                        type = float,
                        default = 1,
                        help = " lambda")
    parser.add_argument("--directory of file",
                        dest = "directory",
                        type = str,
                        default = "permissive_enhancers",
                        help = " path to the file")

    return parser.parse_args()
