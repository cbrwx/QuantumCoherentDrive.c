import numpy as np
import matplotlib.pyplot as plt

class QuantumCoherenceDrive:
    
    def __init__(self, n, omega, energy_storage_capacity):
        self.n = n
        self.omega = omega
        self.alpha = np.sqrt(n)
        self.energy_storage = 0
        self.storage_capacity = energy_storage_capacity
        self.external_factors = ['radiation', 'cosmic_event', 'system_noise']
    
    def energy(self):
        """Calculate the energy associated with the current coherent state."""
        return self.omega * (self.n - abs(self.alpha)**2)
    
    def optimize_coherence(self):
        """Restore coherence."""
        self.alpha = np.sqrt(self.n)
        return self.alpha
    
    def undergo_external_factor(self):
        """Simulate the effects of external factors causing decoherence."""
        factor = np.random.choice(self.external_factors)
        if factor == 'radiation':
            self.alpha *= 0.8
        elif factor == 'cosmic_event':
            self.alpha *= 0.5
        elif factor == 'system_noise':
            self.alpha *= 0.9
        return factor
    
    def energy_differential(self):
        """Calculate the energy differential after an external factor event."""
        E_before = self.energy()
        factor = self.undergo_external_factor()
        E_after = self.energy()
        delta_E = E_before - E_after
        
        # Store the energy generated
        self.energy_storage += delta_E
        if self.energy_storage > self.storage_capacity:
            self.energy_storage = self.storage_capacity
        
        return delta_E, factor
    
    def restore_coherence(self, energy_required):
        """Attempt to restore coherence using stored energy."""
        if self.energy_storage >= energy_required:
            self.energy_storage -= energy_required
            self.optimize_coherence()
            return True
        return False
    
    def generate_safety_field(self):
        """Generate a safety field using stored energy."""
        energy_needed = 0.05 * self.storage_capacity
        if self.energy_storage > energy_needed:
            self.energy_storage -= energy_needed
            return True
        return False
    
    def quantum_communication_relay(self):
        """Attempt to communicate with another QCD to restore coherence."""
        # Simulated as a random chance for demonstration
        return np.random.rand() > 0.7
    
    def quantum_field_optimization(self):
        """Optimize the quantum field using stored energy."""
        energy_needed = 0.1 * self.storage_capacity
        if self.energy_storage > energy_needed:
            self.energy_storage -= energy_needed
            return True
        return False
    
    def simulate_drive(self, steps=10, restoration_energy=0.1):
        """Simulate the QCD for a number of steps."""
        results = []
        for _ in range(steps):
            delta_E, factor = self.energy_differential()
            
            restored = False
            qfo_applied = self.quantum_field_optimization()
            safety_field_generated = self.generate_safety_field()
            qcr_used = self.quantum_communication_relay()

            if not self.restore_coherence(restoration_energy):
                if qfo_applied:
                    restored = self.restore_coherence(restoration_energy * 0.8)
                elif safety_field_generated:
                    restored = self.restore_coherence(restoration_energy * 0.9)
                elif qcr_used:
                    restored = self.restore_coherence(restoration_energy)
            
            results.append({
                'Energy Differential': delta_E,
                'Decoherence Factor': factor,
                'Restored Coherence': restored,
                'Energy Storage': self.energy_storage,
                'QFO Applied': qfo_applied,
                'Safety Field Generated': safety_field_generated,
                'QCR Used': qcr_used
            })
        return results

def plot_simulation_results(results):
    steps = len(results)
    energy_diff = [r['Energy Differential'] for r in results]
    energy_storage = [r['Energy Storage'] for r in results]
    coherence_restored = [1 if r['Restored Coherence'] else 0 for r in results]
    qfo_applied = [1 if r['QFO Applied'] else 0 for r in results]
    safety_field_generated = [1 if r['Safety Field Generated'] else 0 for r in results]
    qcr_used = [1 if r['QCR Used'] else 0 for r in results]

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(range(steps), energy_diff, label='Energy Differential', color='blue')
    plt.plot(range(steps), energy_storage, label='Energy Storage', color='green')
    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.title('Energy Dynamics over Time')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(steps), coherence_restored, label='Coherence Restored', color='red', linestyle='--', marker='o')
    plt.plot(range(steps), qfo_applied, label='QFO Applied', color='purple', linestyle=':', marker='x')
    plt.plot(range(steps), safety_field_generated, label='Safety Field Generated', color='yellow', linestyle='-.', marker='s')
    plt.xlabel('Steps')
    plt.ylabel('Actions (1=True, 0=False)')
    plt.title('Actions over Time')
    plt.yticks([0, 1])
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(steps), qcr_used, label='QCR Used', color='cyan', linestyle='-', marker='d')
    plt.xlabel('Steps')
    plt.ylabel('QCR Used (1=True, 0=False)')
    plt.title('Quantum Communication Relay Usage')
    plt.yticks([0, 1])
    plt.legend()

    plt.tight_layout()
    plt.show()

# Initialize the Quantum Coherence Drive with an energy storage capacity of 10 units
qcd = QuantumCoherenceDrive(n=10, omega=2.0 * np.pi, energy_storage_capacity=10)

# Simulate the QCD for 10 steps
simulation_results = qcd.simulate_drive(steps=10)

# Plot the results
plot_simulation_results(simulation_results)

# Display results
for i, result in enumerate(simulation_results):
    print(f"Step {i + 1}:")
    for key, value in result.items():
        print(f"{key}: {value}")
    print('-' * 50)
