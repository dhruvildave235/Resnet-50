import flwr as fl
from flwr.server.strategy import FedAvg

def main():
    strategy=FedAvg(min_fit_clients=2,min_evaluate_clients=2,min_available_clients=2)
    fl.server.start_server(server_address="localhost:8080",strategy=strategy)

if __name__=="__main__":
    main()
    