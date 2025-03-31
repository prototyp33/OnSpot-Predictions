"""Module for computing spatial features."""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import geopy.distance
from shapely.geometry import Point, Polygon
import folium

def compute_spatial_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute spatial features from latitude and longitude data.
    
    Args:
        data: DataFrame containing 'latitude' and 'longitude' columns
        
    Returns:
        DataFrame with spatial features
    """
    if 'latitude' not in data.columns or 'longitude' not in data.columns:
        raise ValueError("Data must contain 'latitude' and 'longitude' columns")
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Compute distance from city center (example coordinates for a city center)
    city_center = (40.7128, -74.0060)  # New York City coordinates
    result['distance_to_center'] = data.apply(
        lambda row: geopy.distance.distance(
            (row['latitude'], row['longitude']),
            city_center
        ).kilometers,
        axis=1
    )
    
    # Compute spatial clusters
    coords = data[['latitude', 'longitude']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=0.3, min_samples=5).fit(coords_scaled)
    result['spatial_cluster'] = clustering.labels_
    
    # Compute density features
    result['point_density'] = compute_point_density(
        data['latitude'],
        data['longitude'],
        radius_km=1.0
    )
    
    return result

def compute_point_density(
    latitudes: Union[pd.Series, np.ndarray],
    longitudes: Union[pd.Series, np.ndarray],
    radius_km: float = 1.0
) -> pd.Series:
    """Compute point density within a radius.
    
    Args:
        latitudes: Series or array of latitudes
        longitudes: Series or array of longitudes
        radius_km: Radius in kilometers
        
    Returns:
        Series with point densities
    """
    points = list(zip(latitudes, longitudes))
    densities = []
    
    for lat, lon in points:
        count = 0
        for other_lat, other_lon in points:
            distance = geopy.distance.distance(
                (lat, lon),
                (other_lat, other_lon)
            ).kilometers
            if distance <= radius_km:
                count += 1
        densities.append(count / (np.pi * radius_km ** 2))
    
    return pd.Series(densities, index=latitudes.index)

def create_geofence(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    num_points: int = 32
) -> Polygon:
    """Create a circular geofence around a point.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Radius in kilometers
        num_points: Number of points to approximate circle
        
    Returns:
        Shapely Polygon representing the geofence
    """
    points = []
    for i in range(num_points):
        angle = (2 * np.pi * i) / num_points
        point = geopy.distance.distance(kilometers=radius_km).destination(
            (center_lat, center_lon),
            bearing=(angle * 180 / np.pi)
        )
        points.append((point.longitude, point.latitude))
    points.append(points[0])  # Close the polygon
    
    return Polygon(points)

def check_point_in_geofence(
    lat: float,
    lon: float,
    geofence: Polygon
) -> bool:
    """Check if a point is within a geofence.
    
    Args:
        lat: Point latitude
        lon: Point longitude
        geofence: Shapely Polygon representing the geofence
        
    Returns:
        True if point is within geofence, False otherwise
    """
    point = Point(lon, lat)
    return geofence.contains(point)

def compute_distance_matrix(
    points: List[Tuple[float, float]]
) -> np.ndarray:
    """Compute distance matrix between points.
    
    Args:
        points: List of (latitude, longitude) tuples
        
    Returns:
        2D array of distances in kilometers
    """
    n = len(points)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            distance = geopy.distance.distance(
                points[i],
                points[j]
            ).kilometers
            distances[i, j] = distance
            distances[j, i] = distance
    
    return distances

def find_nearest_points(
    reference_point: Tuple[float, float],
    points: List[Tuple[float, float]],
    k: int = 5
) -> List[Tuple[int, float]]:
    """Find k nearest points to a reference point.
    
    Args:
        reference_point: (latitude, longitude) tuple
        points: List of (latitude, longitude) tuples
        k: Number of nearest points to find
        
    Returns:
        List of (point_index, distance) tuples
    """
    distances = []
    for i, point in enumerate(points):
        distance = geopy.distance.distance(
            reference_point,
            point
        ).kilometers
        distances.append((i, distance))
    
    return sorted(distances, key=lambda x: x[1])[:k]

def create_heatmap(
    latitudes: Union[pd.Series, np.ndarray],
    longitudes: Union[pd.Series, np.ndarray],
    values: Optional[Union[pd.Series, np.ndarray]] = None,
    zoom_start: int = 13
) -> folium.Map:
    """Create a heatmap visualization of spatial data.
    
    Args:
        latitudes: Series or array of latitudes
        longitudes: Series or array of longitudes
        values: Optional series or array of values for weighting
        zoom_start: Initial zoom level
        
    Returns:
        Folium Map object with heatmap
    """
    # Calculate center point
    center_lat = np.mean(latitudes)
    center_lon = np.mean(longitudes)
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start
    )
    
    # Prepare data for heatmap
    points = list(zip(latitudes, longitudes))
    if values is not None:
        weights = values
    else:
        weights = [1] * len(points)
    
    # Add heatmap layer
    folium.plugins.HeatMap(
        data=[[lat, lon, weight] for (lat, lon), weight in zip(points, weights)]
    ).add_to(m)
    
    return m

def compute_grid_statistics(
    latitudes: Union[pd.Series, np.ndarray],
    longitudes: Union[pd.Series, np.ndarray],
    values: Union[pd.Series, np.ndarray],
    grid_size_km: float = 1.0
) -> pd.DataFrame:
    """Compute statistics for spatial grid cells.
    
    Args:
        latitudes: Series or array of latitudes
        longitudes: Series or array of longitudes
        values: Series or array of values
        grid_size_km: Size of grid cells in kilometers
        
    Returns:
        DataFrame with grid cell statistics
    """
    # Calculate grid boundaries
    min_lat, max_lat = np.min(latitudes), np.max(latitudes)
    min_lon, max_lon = np.min(longitudes), np.max(longitudes)
    
    # Calculate number of grid cells
    lat_dist = geopy.distance.distance(
        (min_lat, min_lon),
        (max_lat, min_lon)
    ).kilometers
    lon_dist = geopy.distance.distance(
        (min_lat, min_lon),
        (min_lat, max_lon)
    ).kilometers
    
    n_lat = int(np.ceil(lat_dist / grid_size_km))
    n_lon = int(np.ceil(lon_dist / grid_size_km))
    
    # Create grid cells
    lat_edges = np.linspace(min_lat, max_lat, n_lat + 1)
    lon_edges = np.linspace(min_lon, max_lon, n_lon + 1)
    
    # Initialize results
    grid_stats = []
    
    # Compute statistics for each grid cell
    for i in range(n_lat):
        for j in range(n_lon):
            # Get points in current cell
            mask = (
                (latitudes >= lat_edges[i]) &
                (latitudes < lat_edges[i + 1]) &
                (longitudes >= lon_edges[j]) &
                (longitudes < lon_edges[j + 1])
            )
            cell_values = values[mask]
            
            if len(cell_values) > 0:
                stats = {
                    'grid_lat': (lat_edges[i] + lat_edges[i + 1]) / 2,
                    'grid_lon': (lon_edges[j] + lon_edges[j + 1]) / 2,
                    'count': len(cell_values),
                    'mean': np.mean(cell_values),
                    'std': np.std(cell_values),
                    'min': np.min(cell_values),
                    'max': np.max(cell_values)
                }
                grid_stats.append(stats)
    
    return pd.DataFrame(grid_stats) 