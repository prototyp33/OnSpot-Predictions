"""Location management domain model."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Set
from uuid import UUID, uuid4
from enum import Enum

@dataclass(frozen=True)
class Coordinates:
    """Value object representing geographical coordinates."""
    latitude: float
    longitude: float
    
    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError("Invalid latitude")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Invalid longitude")

@dataclass(frozen=True)
class Capacity:
    """Value object representing parking capacity."""
    total_spots: int
    reserved_spots: int = 0
    
    def __post_init__(self):
        if self.total_spots < 0:
            raise ValueError("Total spots cannot be negative")
        if self.reserved_spots < 0:
            raise ValueError("Reserved spots cannot be negative")
        if self.reserved_spots > self.total_spots:
            raise ValueError("Reserved spots cannot exceed total spots")
    
    @property
    def available_spots(self) -> int:
        return self.total_spots - self.reserved_spots

class ZoneType(Enum):
    """Value object representing zone types."""
    PUBLIC = "public"
    RESIDENTIAL = "residential"
    MIXED = "mixed"
    BUSINESS = "business"
    TOURIST = "tourist"
    RESTRICTED = "restricted"

@dataclass
class ParkingSpot:
    """Entity representing a single parking spot."""
    id: UUID
    coordinates: Coordinates
    is_reserved: bool = False
    is_accessible: bool = False
    is_active: bool = True

class Zone:
    """Entity representing a parking zone."""
    
    def __init__(self, name: str, zone_type: ZoneType, coordinates: List[Coordinates]):
        self.id = uuid4()
        self.name = name
        self.zone_type = zone_type
        self.coordinates = coordinates  # Polygon coordinates
        self._spots: Dict[UUID, ParkingSpot] = {}
        self._events = []
    
    def add_spot(self, coordinates: Coordinates, is_reserved: bool = False,
                 is_accessible: bool = False) -> UUID:
        """Add a new parking spot to the zone."""
        spot = ParkingSpot(
            id=uuid4(),
            coordinates=coordinates,
            is_reserved=is_reserved,
            is_accessible=is_accessible
        )
        self._spots[spot.id] = spot
        return spot.id
    
    def remove_spot(self, spot_id: UUID) -> None:
        """Remove a parking spot from the zone."""
        if spot_id in self._spots:
            del self._spots[spot_id]
    
    def get_capacity(self) -> Capacity:
        """Get zone capacity."""
        total = len(self._spots)
        reserved = sum(1 for spot in self._spots.values() if spot.is_reserved)
        return Capacity(total_spots=total, reserved_spots=reserved)
    
    def get_active_spots(self) -> List[ParkingSpot]:
        """Get all active parking spots."""
        return [spot for spot in self._spots.values() if spot.is_active]

class ParkingZone:
    """Aggregate root for parking zones."""
    
    def __init__(self, district_id: str):
        self.id = uuid4()
        self.district_id = district_id
        self._zones: Dict[UUID, Zone] = {}
        self._events = []
    
    def create_zone(self, name: str, zone_type: ZoneType,
                   coordinates: List[Coordinates]) -> UUID:
        """Create a new zone."""
        zone = Zone(name=name, zone_type=zone_type, coordinates=coordinates)
        self._zones[zone.id] = zone
        
        self._events.append({
            'type': 'ZoneCreated',
            'zone_id': zone.id,
            'district_id': self.district_id,
            'timestamp': datetime.now()
        })
        
        return zone.id
    
    def get_zone(self, zone_id: UUID) -> Optional[Zone]:
        """Get a zone by ID."""
        return self._zones.get(zone_id)
    
    def get_total_capacity(self) -> Capacity:
        """Get total capacity across all zones."""
        total = 0
        reserved = 0
        for zone in self._zones.values():
            capacity = zone.get_capacity()
            total += capacity.total_spots
            reserved += capacity.reserved_spots
        return Capacity(total_spots=total, reserved_spots=reserved)
    
    def get_zones_by_type(self, zone_type: ZoneType) -> List[Zone]:
        """Get all zones of a specific type."""
        return [zone for zone in self._zones.values() if zone.zone_type == zone_type]
    
    @property
    def events(self) -> List[Dict]:
        """Get accumulated domain events."""
        return self._events.copy()
    
    def clear_events(self) -> None:
        """Clear accumulated events after they've been processed."""
        self._events.clear()

class LocationService:
    """Application service for managing parking locations."""
    
    def __init__(self):
        self._parking_zones: Dict[str, ParkingZone] = {}
    
    def create_parking_zone(self, district_id: str) -> UUID:
        """Create a new parking zone aggregate."""
        if district_id not in self._parking_zones:
            self._parking_zones[district_id] = ParkingZone(district_id)
        return self._parking_zones[district_id].id
    
    def add_zone(self, district_id: str, name: str, zone_type: ZoneType,
                 coordinates: List[Coordinates]) -> UUID:
        """Add a new zone to a parking zone aggregate."""
        if district_id not in self._parking_zones:
            self.create_parking_zone(district_id)
        
        return self._parking_zones[district_id].create_zone(
            name=name,
            zone_type=zone_type,
            coordinates=coordinates
        )
    
    def get_district_capacity(self, district_id: str) -> Optional[Capacity]:
        """Get total capacity for a district."""
        if district_id not in self._parking_zones:
            return None
        return self._parking_zones[district_id].get_total_capacity()
    
    def get_zones_by_type(self, district_id: str, zone_type: ZoneType) -> List[Zone]:
        """Get all zones of a specific type in a district."""
        if district_id not in self._parking_zones:
            return []
        return self._parking_zones[district_id].get_zones_by_type(zone_type) 