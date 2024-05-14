import json
import matplotlib.pyplot as plt

def load_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['episodes'], data['returns'], data

# Load results from different runs
episodes1, returns1, config1 = load_results('scores/results_target_1.json')
#episodes2, returns2, config2 = load_results('scores/results_target_2.json')
#episodes3, returns3, config3 = load_results('scores/results_target_3.json')

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(episodes1, returns1, label=f'Target Update Frequency {config1["target_update_frequency"]}', marker='o')
#plt.plot(episodes2, returns2, label=f'Target Update Frequency {config2["target_update_frequency"]}', marker='x')
#plt.plot(episodes3, returns3, label=f'Target Update Frequency {config3["target_update_frequency"]}', marker='s')
plt.xlabel('Episode')
plt.ylabel('Mean Return')
plt.title('Training Performance Comparison')
plt.legend()
plt.grid(True)
plt.savefig('combined_results.png')
plt.show()
