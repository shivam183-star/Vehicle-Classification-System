import pandas as pd
import numpy as np

np.random.seed(42)
rows_per_class = 110
data = []

brands = {
    "Sports/Hyper": ["Ferrari", "Lamborghini", "Porsche", "McLaren"],
    "SUV/MPV": ["Toyota", "Hyundai", "Kia", "Tata", "Mahindra"],
    "Sedan/Hatchback": ["Maruti", "Hyundai", "Honda", "Toyota", "Skoda"],
    "Truck": ["Tata", "Ashok Leyland", "Volvo"],
    "Off-Roader": ["Jeep", "Land Rover", "Toyota", "Ford"]
}

def noisy(val, std, low=1):
    return max(low, int(val + np.random.normal(0, std)))

# ---------------- GENERATORS ---------------- #

def sports_hyper():
    power = noisy(np.random.randint(320, 700), 25)
    weight = noisy(np.random.randint(1400, 1900), 80)
    torque = noisy(power * np.random.uniform(1.0, 1.4), 40)

    return {
        "brand": np.random.choice(brands["Sports/Hyper"]),
        "engine_power_hp": power,
        "torque_nm": torque,
        "weight_kg": weight,
        "ground_clearance_mm": np.random.randint(110, 170),  # overlap
        "top_speed_kmph": noisy(170 + power * 0.22, 12),
        "zero_to_100_sec": round(4.0 + (weight / power) , 2)/2,
        "seating_capacity": np.random.choice([2, 4]),
        "drivetrain": np.random.choice(["RWD", "AWD"]),
        "vehicle_class": "Sports/Hyper"
    }

def suv_mpv():
    power = noisy(np.random.randint(140, 260), 20)
    weight = noisy(np.random.randint(1600, 2500), 120)
    torque = noisy(power * np.random.uniform(1.4, 1.9), 50)

    return {
        "brand": np.random.choice(brands["SUV/MPV"]),
        "engine_power_hp": power,
        "torque_nm": torque,
        "weight_kg": weight,
        "ground_clearance_mm": np.random.randint(160, 230),
        "top_speed_kmph": noisy(140 + power * 0.24, 10),
        "zero_to_100_sec": round(8.5 + (weight / power), 2)/2,
        "seating_capacity": np.random.choice([5, 7]),
        "drivetrain": np.random.choice(["FWD", "AWD", "RWD"]),
        "vehicle_class": "SUV/MPV"
    }

def sedan_hatch():
    power = noisy(np.random.randint(90, 190), 20)
    weight = noisy(np.random.randint(1000, 1700), 100)
    torque = noisy(power * np.random.uniform(1.2, 1.6), 40)

    return {
        "brand": np.random.choice(brands["Sedan/Hatchback"]),
        "engine_power_hp": power,
        "torque_nm": torque,
        "weight_kg": weight,
        "ground_clearance_mm": np.random.randint(140, 190),
        "top_speed_kmph": noisy(135 + power * 0.27, 10),
        "zero_to_100_sec": round(8.8 + (weight / power), 2)/2,
        "seating_capacity": np.random.choice([4, 5]),
        "drivetrain": np.random.choice(["FWD", "RWD"]),
        "vehicle_class": "Sedan/Hatchback"
    }

def truck():
    power = noisy(np.random.randint(220, 420), 30)
    weight = noisy(np.random.randint(4500, 10500), 400)
    torque = noisy(np.random.randint(650, 1200), 100)

    return {
        "brand": np.random.choice(brands["Truck"]),
        "engine_power_hp": power,
        "torque_nm": torque,
        "weight_kg": weight,
        "ground_clearance_mm": np.random.randint(200, 280),
        "top_speed_kmph": noisy(85 + power * 0.14, 8),
        "zero_to_100_sec": round(18 + (weight / torque), 2)/2,
        "seating_capacity": np.random.choice([2, 3]),
        "drivetrain": np.random.choice(["RWD", "AWD"]),
        "vehicle_class": "Truck"
    }

def off_roader():
    power = noisy(np.random.randint(160, 320), 25)
    weight = noisy(np.random.randint(1800, 2800), 130)
    torque = noisy(power * np.random.uniform(1.6, 2.2), 50)

    return {
        "brand": np.random.choice(brands["Off-Roader"]),
        "engine_power_hp": power,
        "torque_nm": torque,
        "weight_kg": weight,
        "ground_clearance_mm": np.random.randint(190, 290),
        "top_speed_kmph": noisy(130 + power * 0.22, 10),
        "zero_to_100_sec": round(9.5 + (weight / power), 2)/2,
        "seating_capacity": np.random.choice([4, 5]),
        "drivetrain": np.random.choice(["AWD", "RWD"]),
        "vehicle_class": "Off-Roader"
    }

generators = [sports_hyper, suv_mpv, sedan_hatch, truck, off_roader]

for gen in generators:
    for _ in range(rows_per_class):
        data.append(gen())

df = pd.DataFrame(data)

# ---------------- LABEL NOISE (7%) ---------------- #
noise_ratio = 0.07
n_noise = int(len(df) * noise_ratio)
idx = np.random.choice(df.index, n_noise, replace=False)

df.loc[idx, "vehicle_class"] = np.random.choice(
    df["vehicle_class"].unique(), size=n_noise
)

df.to_csv("data/Cars_Data.csv", index=False)

print("Dataset created:", df.shape)
