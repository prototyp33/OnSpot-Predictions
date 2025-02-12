# OnSpot Predictive Model

**OnSpot** is a smart parking prediction system that integrates data from **OpenStreetMap (OSM)** and **Barcelona SIU** to forecast real-time parking availability. This project leverages historical parking data, spatial analysis, and machine learning to improve urban mobility by helping users find available parking spots efficiently.

---

## Features

- **Data Integration**: Combines OSM and SIU datasets for comprehensive parking data.
- **Predictive Modeling**: Utilizes machine learning algorithms to predict parking availability.
- **Spatial Visualization**: Interactive maps for visualizing parking zones and prediction results.
- **Data Cleaning and Feature Engineering**: Preprocessing steps for accurate and efficient predictions.

---

## Repository Structure

```
OnSpot_Predictive_Model/
│
├── data/                       # Raw, cleaned, and feature-engineered data
│   ├── cleaned_OSM-parking_data.csv
│   ├── feature_engineered_data.csv
│   └── validated_merged_parking_data.csv
│
├── scripts/                    # Python scripts for data processing and modeling
│   ├── mergedata1.py           # Script to merge OSM and SIU data
│   ├── data_utils.py           # Utility functions for data handling
│   ├── define_features.py      # Feature engineering scripts
│   └── load_data.py            # Script to load and preprocess data
│
├── notebooks/                  # Jupyter notebooks for EDA and analysis
│   ├── EDA_Exploration.ipynb   # Exploratory Data Analysis
│   └── merge_analysis.ipynb    # Analysis of merged data
│
├── results/                    # Output files like visualizations and reports
│   └── parking_zones_map.html  # Interactive map of parking zones
│
├── README.md                   # Project documentation
├── requirements.txt            # List of dependencies
└── .gitignore                  # Files and folders to be ignored by Git
```

---

## Getting Started

### **1. Clone the Repository**

```bash
git clone https://github.com/prototyp33/OnSpot-Predictions.git
cd OnSpot-Predictive_Model
```

### **2. Set Up the Environment**

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### **3. Run the Data Merging Script**

Merge the OSM and SIU datasets:

```bash
python scripts/mergedata1.py
```

This will generate the merged data file:
- `data/validated_merged_parking_data.csv`

### **4. Visualize the Results**

Open the interactive map to visualize parking zones:

```bash
open results/parking_zones_map.html
```

---

## Data Sources

- **OpenStreetMap (OSM)**: Provides geographical data including parking locations and restrictions.
- **Barcelona SIU (Open Data BCN)**: Historical parking data from the city of Barcelona, used for occupancy and turnover predictions.

---

## Key Scripts

- `mergedata1.py`: Merges OSM and SIU data, handles spatial matching, and validates merged data.
- `data_utils.py`: Utility functions for loading, cleaning, and preprocessing data.
- `define_features.py`: Script for creating and engineering features for the predictive model.
- `load_data.py`: Loads datasets into the predictive pipeline.

---

## Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository.
2. **Create a new branch**:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Commit your changes**:
   ```bash
   git commit -m "Add your message here"
   ```
4. **Push to the branch**:
   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open a Pull Request**.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or suggestions, please reach out via [GitHub Issues](https://github.com/prototyp33/OnSpot-Predictions/issues).

---

## Acknowledgments

- **OpenStreetMap** contributors for providing detailed geographical data.
- **Barcelona Open Data BCN** for offering valuable parking datasets.
- **Folium** for interactive map visualizations.
- **Scikit-learn** for machine learning tools and techniques.
