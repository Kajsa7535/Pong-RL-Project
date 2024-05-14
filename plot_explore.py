import json
import matplotlib.pyplot as plt

def load_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['episodes'], data['returns'], data

# Load results from different runs


episodes1, returns1, config1 = load_results('scores/results_explore_1.json')
episodes2, returns2, config2 = load_results('scores/results_explore_2.json')
#episodes3, returns3, config3 = load_results('scores/results_explore_3.json')
episodes4, returns4, config4 = load_results('scores/results_explore_4.json')
#episodes5, returns5, config5 = load_results('scores/results_explore_5.json')
episodes6, returns6, config6 = load_results('scores/results_explore_6.json')
episodes7, returns7, config7 = load_results('scores/results_target_lower_learning_rate.json')




# Plot results
plt.figure(figsize=(10, 6))
plt.plot(episodes1, returns1, label=f'eps_start: {config1["eps_start"]}, eps_end: {config1["eps_end"]}, anneal_length: {config1["anneal_length"]}')
plt.plot(episodes2, returns2, label=f'eps_start: {config2["eps_start"]}, eps_end: {config2["eps_end"]}, anneal_length: {config2["anneal_length"]}')
#plt.plot(episodes3, returns3, label=f'eps_start: {config3["eps_start"]}, eps_end: {config3["eps_end"]}, anneal_length: {config3["anneal_length"]}')
plt.plot(episodes4, returns4, label=f'eps_start: {config4["eps_start"]}, eps_end: {config4["eps_end"]}, anneal_length: {config4["anneal_length"]}')
#plt.plot(episodes5, returns5, label=f'eps_start: {config5["eps_start"]}, eps_end: {config5["eps_end"]}, anneal_length: {config5["anneal_length"]}')
plt.plot(episodes6, returns6, label=f'eps_start: {config6["eps_start"]}, eps_end: {config6["eps_end"]}, anneal_length: {config6["anneal_length"]}')
plt.plot(episodes7, returns7, label=f'eps_start: {config7["eps_start"]}, eps_end: {config7["eps_end"]}, anneal_length: {config7["anneal_length"]}')

plt.xlabel('Episode')
plt.ylabel('Mean Return')
plt.title('Training Performance Comparison')
plt.legend()
plt.grid(True)
plt.savefig('combined_results.png')
plt.show()
