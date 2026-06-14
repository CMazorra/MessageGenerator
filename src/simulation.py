import simpy
import random
import pandas as pd
import statistics

ARRIVAL_RATE_MINUTES = 1.5 
LAMBDA_RATE = ARRIVAL_RATE_MINUTES / 60.0
SIMULATION_TIME = 3600
NUM_REPLICAS = 30

MODELS_LATENCY = {
    "Mistral_BeamSearch": (45.04, 5.0),
    "Mistral_MCTS": (38.55, 6.0),
    "Llama3_BeamSearch": (25.90, 4.0),   
    "Llama3_MCTS": (77.32, 10.0),
    "Phi3_BeamSearch": (19.93, 3.0),
    "Phi3_MCTS": (417.37, 50.0)
}

class TextGenerationServer:
    def __init__(self, env, num_workers, model_name):
        self.env = env
        self.machine = simpy.Resource(env, capacity=num_workers)
        self.mu, self.sigma = MODELS_LATENCY[model_name]

    def generate_text(self):
        service_time = max(1.0, random.gauss(self.mu, self.sigma))
        yield self.env.timeout(service_time)

def user_request(env, server, metrics):
    arrival_time = env.now
    with server.machine.request() as request:
        yield request
        wait_time = env.now - arrival_time
        start_service = env.now
        
        yield env.process(server.generate_text())
        
        metrics.append({
            "wait_time_sec": wait_time,
            "service_time_sec": env.now - start_service
        })

def setup_environment(env, num_workers, model_name, metrics):
    server = TextGenerationServer(env, num_workers, model_name)
    while True:
        inter_arrival_time = random.expovariate(LAMBDA_RATE)
        yield env.timeout(inter_arrival_time)
        env.process(user_request(env, server, metrics))

def run_single_replica(model_name, num_workers, seed):
    """Ejecuta una sola réplica de 1 hora de simulación."""
    random.seed(seed)
    env = simpy.Environment()
    metrics = []
    env.process(setup_environment(env, num_workers, model_name, metrics))
    env.run(until=SIMULATION_TIME)
    
    if not metrics:
        return 0, 0
        
    df = pd.DataFrame(metrics)
    avg_wait = df['wait_time_sec'].mean()
    utilization = (df['service_time_sec'].sum() / (SIMULATION_TIME * num_workers)) * 100
    return avg_wait, utilization

def run_statistical_experiment(model_name, num_workers):
    """Ejecuta múltiples réplicas y aplica estadística formal."""
    wait_times = []
    utilizations = []
    
    for i in range(NUM_REPLICAS):
        wait, util = run_single_replica(model_name, num_workers, seed=42+i)
        wait_times.append(wait)
        utilizations.append(util)
    
    # AQUÍ SE VE EL USO DEL MÓDULO STATISTICS
    mean_wait = statistics.mean(wait_times)
    std_wait = statistics.stdev(wait_times) if len(wait_times) > 1 else 0
    mean_util = statistics.mean(utilizations)
    
    return mean_wait, std_wait, mean_util

if __name__ == "__main__":
    print(f"Iniciando Módulo de Simulación de Eventos Discretos (DES)")
    print(f"Tráfico: {ARRIVAL_RATE_MINUTES} req/min | Réplicas por escenario: {NUM_REPLICAS}\n")
    
    results = []
    workers_to_test = [1, 2, 3] # Probaremos con 1, 2 y 3 servidores
    
    for model in MODELS_LATENCY.keys():
        for w in workers_to_test:
            mean_wait, std_wait, mean_util = run_statistical_experiment(model, w)
            results.append({
                "Modelo": model,
                "Workers": w,
                "Espera_Media_Sec": round(mean_wait, 2),
                "Espera_StdDev": round(std_wait, 2),
                "Utilización_%": round(mean_util, 2)
            })
            
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))