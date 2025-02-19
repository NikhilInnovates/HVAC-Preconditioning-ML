import numpy as np
import pandas as pd
import math
import os

# Parameters
num_samples = 1000  # Number of time steps (for ~6 GB dataset)
time_step = 60  # Time step in seconds (1 minute)
k = 0.01  # Heat transfer coefficient
hvac_power = 5  # HVAC heating/cooling power (degrees per minute)
thermal_mass = 50  # Building's thermal inertia
hvac_response_time = 5  # Minutes to reach full effect
day_length = 1440  # Minutes in a day (24h)

# File to save data
output_file = 'synthetic_data.csv'
chunk_size = 1_000_000  # Number of rows per chunk

# Function to generate a chunk of data
def generate_chunk(start_idx, end_idx, initial_inside_temp, initial_set_point):
    timestamps = []
    inside_temps = []
    outside_temps = []
    set_points = []
    heat_flows = []
    heating_times = []

    heating_duration = 0  # Track heating/cooling duration
    heat_flow = 0
    inside_temp = initial_inside_temp  # Initialize inside temperature
    set_point = initial_set_point  # Initialize set point

    for t in range(start_idx, end_idx):
        # Simulate daily and seasonal temperature variations
        seasonal_variation = 10 * math.sin(2 * math.pi * t / (day_length * 365))  # Seasonal effect
        daily_variation = 10 * math.sin(2 * math.pi * t / day_length)  # Daily effect
        outside_temp = 15 + daily_variation + seasonal_variation  # Base temperature + variations
        outside_temp += np.random.normal(0, 1)  # Random fluctuations (±1°C)
        outside_temp = np.clip(outside_temp, -10, 40)  # Clip to realistic range

        # Simulate dynamic set point changes
        if t % 300 == 0:  # Change set point every 5 hours
            set_point = np.random.uniform(18, 25)  # Random set point between 18°C and 25°C
        set_point += np.random.normal(0, 0.1)  # Small random variations

        # HVAC control with gradual power application
        if inside_temp < set_point - 0.5:
            heat_flow = min(hvac_power, heat_flow + hvac_power / hvac_response_time)
        elif inside_temp > set_point + 0.5:
            heat_flow = max(-hvac_power, heat_flow - hvac_power / hvac_response_time)
        else:
            heat_flow = 0

        # Introduce HVAC inefficiencies
        heat_flow *= np.random.uniform(0.8, 1.2)  # 80%-120% efficiency variation

        # Heat transfer with thermal mass
        dT_dt = k * (outside_temp - inside_temp) + heat_flow
        inside_temp += (dT_dt * time_step / 60) / thermal_mass  # Adjusted with thermal inertia

        # Simulate sensor noise
        inside_temp += np.random.normal(0, 0.2)  # ±0.2°C sensor error

        # Track heating/cooling duration
        if abs(inside_temp - set_point) < 0.5:
            heating_duration += time_step  # Increment duration
        else:
            heating_duration = 0  # Reset if out of range

        # Store data
        timestamps.append(t * time_step)
        inside_temps.append(inside_temp)
        outside_temps.append(outside_temp)
        set_points.append(set_point)
        heat_flows.append(heat_flow)
        heating_times.append(heating_duration)

    # Create DataFrame for the chunk
    chunk_data = pd.DataFrame({
        'Timestamp': timestamps,
        'Inside_Temperature': inside_temps,
        'Outside_Temperature': outside_temps,
        'Set_Point_Temperature': set_points,
        'Heat_Flow': heat_flows,
        'Heating_Cooling_Time': heating_times
    })
    return chunk_data, inside_temp, set_point  # Return the final inside_temp and set_point for the next chunk

# Generate data in chunks and save to CSV
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='') as f:
        # Write header
        pd.DataFrame(columns=[
            'Timestamp', 'Inside_Temperature', 'Outside_Temperature',
            'Set_Point_Temperature', 'Heat_Flow', 'Heating_Cooling_Time'
        ]).to_csv(f, index=False)

    # Initialize inside temperature and set point
    current_inside_temp = 25  # Starting inside temperature (°C)
    current_set_point = 22  # Initial set point temperature (°C)

    # Generate and save chunks
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        print(f"Generating chunk {start_idx // chunk_size + 1}...")
        chunk_data, current_inside_temp, current_set_point = generate_chunk(start_idx, end_idx, current_inside_temp, current_set_point)
        chunk_data.to_csv(output_file, mode='a', header=False, index=False)

    print(f"Realistic synthetic data generated and saved to '{output_file}'")
else:
    print(f"File '{output_file}' already exists.")