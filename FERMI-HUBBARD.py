import cirq
import numpy as np
import matplotlib.pyplot as plt


q0, q1 = cirq.LineQubit.range(2)

def simulate_hopping(total_time, steps, hopping_strength):
    dt = total_time / steps
    circuit = cirq.Circuit(cirq.X(q0))
    
    probs_site_0 = []
    times = np.linspace(0, total_time, steps)

    for _ in range(steps):
        #(using riswap for t)
        exponent = (2 * hopping_strength * dt) / np.pi
        circuit.append(cirq.riswap(exponent).on(q0, q1))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        
        # State vector index 2 is |10 index 1 is |01
        prob = np.abs(result.state_vector())**2
        probs_site_0.append(prob[2])

    return times, probs_site_0

t_strength = 1.0
time_axis, occupancy = simulate_hopping(total_time=10, steps=100, hopping_strength=t_strength)


plt.figure(figsize=(8, 5))
plt.plot(time_axis, occupancy, label='Site 0 Occupancy', color='#1f77b4', linewidth=2)
plt.plot(time_axis, 1 - np.array(occupancy), label='Site 1 Occupancy', color='#ff7f0e', linestyle='--')

plt.title(f"Quantum Simulation: Electron Hopping (t={t_strength})", fontsize=14)
plt.xlabel("Time (Arb. Units)", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-0.1, 1.1)

plt.show()