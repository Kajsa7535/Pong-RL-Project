import json
import matplotlib.pyplot as plt

def load_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['episodes'], data['returns'], data

# Load results from different runs


episodes1, returns1, config1 = load_results('scores/results_target_6.json') #5
episodes2, returns2, config2 = load_results('scores/results_target_1.json') # 10
episodes3, returns3, config3 = load_results('scores/results_target_3.json') # 50
episodes4, returns4, config4 = load_results('scores/results_target_4.json') #100
episodes5, returns5, config5 = load_results('scores/results_target_5.json') # 200
episodes6, returns6, config6 = load_results('scores/results_target_7.json') # 500
episodes8, returns8, config8 = load_results('scores/results_target_8.json') #1000
episodes9, returns9, config9 = load_results('scores/results_target_9.json') #100




# Plot results
plt.figure(figsize=(10, 6))
plt.plot(episodes1, returns1, label=f'Target Update Frequency {config1["target_update_frequency"]}')
plt.plot(episodes2, returns2, label=f'Target Update Frequency {config2["target_update_frequency"]}')
plt.plot(episodes3, returns3, label=f'Target Update Frequency {config3["target_update_frequency"]}')
plt.plot(episodes9, returns9, label=f'Target Update Frequency {config9["target_update_frequency"]}')
plt.plot(episodes5, returns5, label=f'Target Update Frequency {config5["target_update_frequency"]}')
plt.plot(episodes6, returns6, label=f'Target Update Frequency {config6["target_update_frequency"]}')
plt.plot(episodes8, returns8, label=f'Target Update Frequency {config8["target_update_frequency"]}')
plt.xlabel('Episode')
plt.ylabel('Mean Return')
plt.title('Training Performance Comparison')
plt.legend()
plt.grid(True)
plt.savefig('combined_results.png')
plt.show()
