import argparse
import logger
from echo import echo_state_network
from dataprep import create_dataset
from keras_echo import keras_esn

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", help = "Hidden layer activation function", choices = ['keras', 'mantas'], type = str, default = 'mantas')
    parser.add_argument("--data", help = "Hidden layer activation function", choices = ['mackey', 'wind', 'weather'], type = str, default = 'mackey')
    return parser.parse_args()
    
log = logger.setup_logger(__name__)

def main():
    config = arg_parse()
    log.info("Starting...")
    log.info("An echo state network will be run on the following dataset:")
    log.info(config.data)

    input_data = create_dataset(config.data)

    if config.network == 'mantas':
        echo_state_network(input_data)

    else:
        keras_esn(input_data, input_data)


if __name__ == "__main__":
    main()