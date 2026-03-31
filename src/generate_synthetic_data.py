import pandas as pd
import numpy as np

def generate_decay_data(num_satellites=10, days_tracked=100):
    print(f"Simulating orbital decay for {num_satellites} satellites...")
    np.random.seed(42)
    data = []
    
    for sat_id in range(1, num_satellites + 1):
        # Assign random starting characteristics for each satellite
        start_sma = np.random.uniform(6700, 6900) # Starting altitude
        base_bstar = np.random.uniform(0.00001, 0.00005) # Starting drag
        
        # Simulate a countdown to zero
        for day in range(days_tracked, 0, -1):
            progress = (days_tracked - day) / float(days_tracked) # 0.0 to 1.0
            
            # Physics Simulation: Altitude drops exponentially faster at the end
            current_sma = start_sma - (progress ** 4) * 400 
            # Add Gaussian noise to simulate TLE sensor inaccuracies
            noisy_sma = current_sma + np.random.normal(0, 2.5) 
            
            # Physics Simulation: Atmospheric drag (Bstar) spikes as altitude drops
            current_bstar = base_bstar + (progress ** 5) * 0.008
            noisy_bstar = abs(current_bstar + np.random.normal(0, 0.0002))
            
            data.append({
                "satellite_id": f"SAT-{sat_id}",
                "days_to_decay": day,
                "semi_major_axis_km": noisy_sma,
                "bstar": noisy_bstar
            })

    df = pd.DataFrame(data)
    df.to_csv("orbital_features.csv", index=False)
    print(f"Success! Saved {len(df)} rows to orbital_features.csv")

if __name__ == "__main__":
    generate_decay_data()