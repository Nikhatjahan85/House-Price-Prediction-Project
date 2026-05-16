import pandas as pd
import numpy as np
import os

# =====================================================
# CREATE DATA FOLDER
# =====================================================

os.makedirs("data", exist_ok=True)

# =====================================================
# RANDOM DATA GENERATION
# =====================================================

np.random.seed(42)

num_samples = 1000

area = np.random.randint(500, 5000, num_samples)

bedrooms = np.random.randint(1, 6, num_samples)

bathrooms = np.random.randint(1, 5, num_samples)

parking = np.random.randint(0, 3, num_samples)

age = np.random.randint(1, 30, num_samples)

furnishing = np.random.choice(
    ['Furnished', 'Semi-Furnished', 'Unfurnished'],
    num_samples
)

location = np.random.choice(
    ['Urban', 'Semi-Urban', 'Rural'],
    num_samples
)

# =====================================================
# PRICE CALCULATION LOGIC
# =====================================================

price = (
    area * 3000
    + bedrooms * 500000
    + bathrooms * 300000
    + parking * 200000
    - age * 10000
)

# Add location effect
location_effect = []

for loc in location:

    if loc == 'Urban':
        location_effect.append(2000000)

    elif loc == 'Semi-Urban':
        location_effect.append(1000000)

    else:
        location_effect.append(300000)

price += location_effect

# Add furnishing effect
furnishing_effect = []

for furn in furnishing:

    if furn == 'Furnished':
        furnishing_effect.append(500000)

    elif furn == 'Semi-Furnished':
        furnishing_effect.append(250000)

    else:
        furnishing_effect.append(100000)

price += furnishing_effect

# Random noise
price += np.random.randint(-200000, 200000, num_samples)

# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'parking': parking,
    'age': age,
    'furnishing': furnishing,
    'location': location,
    'price': price
})

# =====================================================
# SAVE DATASET
# =====================================================

df.to_csv("data/housing.csv", index=False)

print("\nDataset Generated Successfully")
print("\nSaved at: data/housing.csv")

print("\nFirst 5 Rows:\n")
print(df.head())