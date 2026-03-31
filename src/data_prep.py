from skyfield.api import EarthSatellite, load
import math

def parse_tle(name, line1, line2):
    """Parses a TLE string into numerical orbital features."""
    ts = load.timescale()
    satellite = EarthSatellite(line1, line2, name, ts)
    
    # Extract B*(drag coefficient) directly from the SGP4 model
    bstar = satellite.model.bstar
    
    # Extract Mean Motion (radians per minute)
    mean_motion_rad_per_min = satellite.model.no_kozai 
    
    # Convert to standard Revolutions Per Day
    mean_motion_rev_per_day = mean_motion_rad_per_min * (1440.0 / (2 * math.pi))
    
    # Calculate Semi-Major Axis (km) using Kepler's 3rd Law
    mu = 398600.4418 # Earth's gravitational parameter (km^3/s^2)
    n_rad_per_sec = mean_motion_rad_per_min / 60.0
    semi_major_axis_km = (mu / (n_rad_per_sec ** 2)) ** (1.0 / 3.0)
    
    # Extract the exact Epoch (timestamp) of this TLE
    epoch = satellite.epoch.utc_datetime()
    
    return {
        "name": name,
        "epoch": epoch,
        "bstar": bstar,
        "mean_motion_rev_day": mean_motion_rev_per_day,
        "semi_major_axis_km": semi_major_axis_km
    }


# Real historical TLE from Tiangong-1 right before it decayed
line1 = "1 37820U 11053A   18089.47167512  .00492822  11123-4  36400-3 0  9997"
line2 = "2 37820  42.7486 288.5284 0011855 106.6713 253.5186 16.33129487373516"

features = parse_tle("TIANGONG 1", line1, line2)

print("--- extracted orbital features ---")
for key, value in features.items():
    print(f"{key}: {value}")

import pandas as pd
from datetime import datetime, timezone

# The historical decay date of Tiangong-1 (Target zero-point)
DECAY_DATE = datetime(2018, 4, 2, 0, 16, tzinfo=timezone.utc)

# A small historical dataset of Tiangong-1 TLEs leading up to its decay
tle_history = [
    # 13 days out
    ("1 37820U 11053A   18079.52627019  .00095861  00000-0  10664-3 0  9997",
     "2 37820  42.7601 336.1969 0013446 250.3957 109.5298 16.03753176371900"),
    # 8 days out
    ("1 37820U 11053A   18084.45341257  .00166662  00000-0  14590-3 0  9998",
     "2 37820  42.7569 312.6366 0012579 261.3283  98.6361 16.14389977372697"),
    # 3 days out
    ("1 37820U 11053A   18089.47167512  .00492822  11123-4  36400-3 0  9997",
     "2 37820  42.7486 288.5284 0011855 106.6713 253.5186 16.33129487373516"),
    # 1 day out
    ("1 37820U 11053A   18091.24838686  .01353272  11287-4  71181-3 0  9998",
     "2 37820  42.7410 280.0039 0012470 102.2471 258.0772 16.51261329373809"),
    # Hours out
    ("1 37820U 11053A   18091.75845012  .02604060  11467-4  10769-2 0  9993",
     "2 37820  42.7303 277.5681 0012521 113.1251 247.2403 16.63414967373898")
]

def build_dataset(tle_list, decay_date):
    """Processes multiple TLEs into a Pandas DataFrame for Machine Learning."""
    data = []
    for line1, line2 in tle_list:
        # Reusing the parse_tle function from Step 1
        features = parse_tle("TIANGONG-1", line1, line2)
        
        # Calculate the Target Variable (Y) - How many days until decay
        time_diff = decay_date - features["epoch"]
        days_to_decay = time_diff.total_seconds() / (24 * 3600)
        
        # Add the target to our feature dictionary
        features["days_to_decay"] = days_to_decay
        data.append(features)
        
    return pd.DataFrame(data)

# Generate the DataFrame
df = build_dataset(tle_history, DECAY_DATE)

print("\n--- training data (features n target) ---")
print(df[["semi_major_axis_km", "bstar", "days_to_decay"]].to_string(index=False))