import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import './Style.css';

// Fix for default marker icon issue
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

// Fix for default marker icon
const DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

const ParkingLotApp = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(true);
  const [username, setUsername] = useState('Guest');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showEditProfile, setShowEditProfile] = useState(false);
  const [userProfile, setUserProfile] = useState({
    name: 'Guest',
    email: '',
    address: ''
  });
  const [startingLocation, setStartingLocation] = useState('');
  const [recommendationPriority, setRecommendationPriority] = useState('occupancy');
  const [routes, setRoutes] = useState([]);
  const [isLoadingRoutes, setIsLoadingRoutes] = useState(false);
  const [visibleRoutes, setVisibleRoutes] = useState([]);
  const [directionsReady, setDirectionsReady] = useState(false);
  const profileRef = useRef(null);
  const mapRef = useRef(null);

  // Parking lots state with coordinates
  const [parkingLots, setParkingLots] = useState([
    {
      name: 'Perry Street',
      spotsAvailable: 0,
      isFull: false,
      location: [37.23086, -80.42565], // Perry Street Parking Garage exact coordinates
      address: 'Perry Street, Blacksburg, VA',
      color: '#3F8EF2'
    },
    {
      name: 'North End',
      spotsAvailable: 0,
      isFull: false,
      location: [37.233211, -80.420061], // North End Center Parking exact coordinates
      address: 'North End Center, Blacksburg, VA',
      color: '#FF9333'
    },
    {
      name: 'Kent Street',
      spotsAvailable: 0,
      isFull: false,
      location: [37.227811, -80.413660], // Kent Square Parking Garage exact coordinates
      address: '207 Draper Rd SW, Blacksburg, VA 24060',
      color: '#FF5E5E'
    }
  ]);

  // Fetch parking predictions
  // const fetchPredictions = async () => {
  //   try {
  //     const response = await axios.get('http://localhost:3001/api/predictions');
  //     const updatedLots = response.data;
      
  //     setParkingLots(prevLots => 
  //       prevLots.map(lot => {
  //         const updatedLot = updatedLots.find(u => u.name === lot.name);
  //         return updatedLot ? { ...lot, ...updatedLot } : lot;
  //       })
  //     );
  //   } catch (error) {
  //     console.error('Error fetching predictions:', error);
  //   }
  // };

  // // Fetch predictions on component mount and every minute
  // useEffect(() => {
  //   fetchPredictions();
  //   const interval = setInterval(fetchPredictions, 60000); // Update every minute
  //   return () => clearInterval(interval);
  // }, []);

  useEffect(() => {
    // Set initial theme
    document.body.classList.toggle('dark-mode', isDarkMode);
  }, [isDarkMode]);

  useEffect(() => {
    // Close profile menu when clicking outside
    const handleClickOutside = (event) => {
      if (profileRef.current && !profileRef.current.contains(event.target)) {
        setShowProfileMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
    document.body.classList.toggle('dark-mode', !isDarkMode);
  };

  const toggleProfileMenu = () => {
    setShowProfileMenu(!showProfileMenu);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUsername('');
    setPassword('');
    setShowProfileMenu(false);
  };

  const handleEditAccount = () => {
    setShowEditProfile(true);
    setShowProfileMenu(false);
  };

  const handleSaveProfile = () => {
    setShowEditProfile(false);
  };

  const handleCancelEdit = () => {
    setShowEditProfile(false);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setUserProfile(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const findRoutes = async () => {
    if (!startingLocation || !userProfile.address) {
      alert('Please select a starting location and ensure your address is set in your profile');
      return;
    }

    setIsLoadingRoutes(true);
    setRoutes([]);
    setVisibleRoutes([]);

    try {
      const formatAddress = (address) => {
        if (address.toLowerCase().includes('blacksburg') || 
            address.toLowerCase().includes('va') || 
            address.toLowerCase().includes('24061')) {
          return address;
        }
        return `${address}, Blacksburg, VA 24061`;
      };

      const origin = startingLocation === 'home' 
        ? formatAddress(userProfile.address) 
        : formatAddress(startingLocation);
      
      const directionsService = new window.google.maps.DirectionsService();

      const routePromises = parkingLots.map(async (lot) => {
        const request = {
          origin: origin,
          destination: formatAddress(lot.address),
          travelMode: window.google.maps.TravelMode.DRIVING
        };

        const result = await directionsService.route(request);
        
        const path = result.routes[0].overview_path.map(point => ({
          lat: point.lat(),
          lng: point.lng()
        }));

        // Get the exact endpoint coordinates
        const endPoint = path[path.length - 1];
        console.log(`Route endpoint for ${lot.name}:`, endPoint);

        return {
          path,
          color: lot.color,
          lotName: lot.name,
          duration: result.routes[0].legs[0].duration.text,
          distance: result.routes[0].legs[0].distance.text,
          travelTime: result.routes[0].legs[0].duration.value, // Travel time in seconds
          id: lot.name,
          endPoint: endPoint
        };
      });

      const results = await Promise.all(routePromises);
      
      // Prepare travel times and distances for prediction
      const travelTimes = {};
      const distances = {};
      results.forEach((route, index) => {
        travelTimes[`garage${index + 1}`] = route.travelTime;
        distances[`garage${index + 1}`] = parseFloat(route.distance.replace(" mi", ""));
      });

      // Get predictions based on travel times
      try {
        const predictionResponse = await axios.post(
          'http://localhost:3000/predict', 
          {
            travelTimes,
            distances,
            priority: recommendationPriority
          },
          {
            headers: {
              'Content-Type': 'application/json'
            }
          }
        );
        
        // Update parking lots with predictions
        const predictions = predictionResponse.data;
        console.log("Received predictions from backend:", predictionResponse.data);

        setParkingLots(prevLots =>
          prevLots.map(lot => {
            let garageKey;
        
            // Map lot names to the keys returned by backend
            if (lot.name === 'Perry Street') {
              garageKey = 'garage1';
            } else if (lot.name === 'North End') {
              garageKey = 'garage2';
            } else if (lot.name === 'Kent Street') {
              garageKey = 'garage3';
            }
        
            const prediction = predictions[garageKey];
        
            return prediction
              ? {
                  ...lot,
                  spotsAvailable: Math.round(prediction.expected_occupancy),
                  isFull: prediction.expected_occupancy <= 0
                }
              : lot;
          })
        );
      } catch (error) {
        console.error('Error getting predictions:', error);
      }

      setRoutes(results);
      setVisibleRoutes(results.map(route => route.id));
    } catch (error) {
      console.error('Error calculating routes:', error);
      alert('Error calculating routes. Please try again.');
    } finally {
      setIsLoadingRoutes(false);
      setDirectionsReady(true);
    }
  };

  const toggleRoute = (routeId) => {
    setVisibleRoutes(prev => 
      prev.includes(routeId)
        ? prev.filter(id => id !== routeId)
        : [...prev, routeId]
    );
  };

  // Get the recommended lot based on priority
  const getRecommendedLot = () => {
    if (!routes.length) return { name: 'No routes calculated', spotsAvailable: 0 };

    const lotsWithRoutes = parkingLots.map(lot => {
      const route = routes.find(r => r.lotName === lot.name);
      return {
        ...lot,
        duration: route ? route.travelTime : Infinity,
        distance: route ? parseFloat(route.distance.replace(' mi', '')) : Infinity
      };
    });

    switch (recommendationPriority) {
      case 'time':
        return lotsWithRoutes.reduce((prev, current) => 
          (prev.duration < current.duration) ? prev : current
        );
      case 'distance':
        return lotsWithRoutes.reduce((prev, current) => 
          (prev.distance < current.distance) ? prev : current
        );
      case 'occupancy':
      default:
        return lotsWithRoutes
          .filter(lot => !lot.isFull)
          .reduce((prev, current) => 
            (prev.spotsAvailable > current.spotsAvailable) ? prev : current
          , { name: 'No available lots', spotsAvailable: 0 });
    }
  };

  // Get the recommendation reason based on priority
  const getRecommendationReason = (lot) => {
    if (!routes.length) return '';
    const route = routes.find(r => r.lotName === lot.name);
    if (!route) return '';

    switch (recommendationPriority) {
      case 'time':
        return `Shortest travel time: ${route.duration}`;
      case 'distance':
        return `Shortest distance: ${route.distance}`;
      case 'occupancy':
        return `Most spots available: ${lot.spotsAvailable} spots`;
      default:
        return '';
    }
  };

  // Get the recommended lot
  const recommendedLot = getRecommendedLot();

  return (
    <div className="parking-app">
      <header className="app-header">
        <div className="logo-container">
          <img src="/vt-logo.png" alt="VT Logo" className="vt-logo" />
          <h1 className="app-title">Parking Predictor</h1>
        </div>
        <div className="header-right">
          <button className="theme-toggle" onClick={toggleTheme}>
            <i className={`fas fa-${isDarkMode ? 'sun' : 'moon'}`}></i>
            {isDarkMode ? 'Light Mode' : 'Dark Mode'}
          </button>
          <div className="profile-container" ref={profileRef}>
            <button className="profile-btn" onClick={toggleProfileMenu}>
              <i className="fas fa-user"></i> Profile
            </button>
            <div className={`profile-popup ${showProfileMenu ? 'show' : ''}`}>
              <button onClick={handleEditAccount}>
                <i className="fas fa-edit"></i> Edit Account
              </button>
              <button onClick={handleLogout}>
                <i className="fas fa-sign-out-alt"></i> Log Out
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="app-content">
        <MapContainer 
          center={[37.2295, -80.4139]} 
          zoom={15} 
          className="map-container"
          ref={mapRef}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          {parkingLots.map((lot) => (
            <Marker key={lot.name} position={lot.location}>
              <Popup>
                {lot.name} Lot
                <br />
                {lot.isFull 
                  ? 'Lot is Full' 
                  : `${lot.spotsAvailable} Spots Available`
                }
              </Popup>
            </Marker>
          ))}
          {routes
            .filter(route => visibleRoutes.includes(route.id))
            .map((route) => (
              <Polyline
                key={route.id}
                positions={route.path}
                color={route.color}
                weight={3}
                opacity={0.7}
              />
            ))}
        </MapContainer>

        <div className="sidebar">
          <div className="location-selector">
            <div className="input-box">
              <label>Starting Location</label>
              <select 
                value={startingLocation}
                onChange={(e) => setStartingLocation(e.target.value)}
              >
                <option value="">Select Location</option>
                <option value="home">My Home</option>
                <option value="campus-center">Campus Center</option>
                <option value="library">Library</option>
              </select>
            </div>

            <div className="input-box">
              <label>Recommendation Priority</label>
              <select
                value={recommendationPriority}
                onChange={(e) => setRecommendationPriority(e.target.value)}
              >
                <option value="no-priority">No Priority</option>
                <option value="occupancy">Available Spots</option>
                <option value="time">Travel Time</option>
                <option value="distance">Distance</option>
              </select>
            </div>

            <button 
              className="find-routes-btn"
              onClick={findRoutes}
              disabled={isLoadingRoutes}
            >
              {isLoadingRoutes ? 'Finding Routes...' : 'Find Routes!'}
            </button>

            {routes.length > 0 && (
              <div className="route-info">
                {routes.map((route) => (
                  <div 
                    key={route.id} 
                    className={`route-detail ${visibleRoutes.includes(route.id) ? 'active' : 'inactive'}`}
                    style={{ color: route.color }}
                    onClick={() => toggleRoute(route.id)}
                  >
                    <h4>{route.lotName}</h4>
                    <p>Distance: {route.distance}</p>
                    <p>Duration: {route.duration}</p>
                    <div className="route-toggle">
                      {visibleRoutes.includes(route.id) ? 'Hide Route' : 'Show Route'}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {recommendedLot && (
              <div className="recommended-lot">
                <h3>Recommended Lot</h3>
                <div className="lot-details">
                  <h4>{recommendedLot.name}</h4>
                  {recommendedLot.spotsAvailable > 0 && (
                    <>
                      <p>{getRecommendationReason(recommendedLot)}</p>
                      <p>{recommendedLot.spotsAvailable} spots available</p>
                    </>
                  )}
                  <button
                    hidden={!directionsReady}
                    className="gmaps-btn"
                    onClick={() => {
                      const originEncoded = encodeURIComponent(startingLocation === 'home' ? userProfile.address : startingLocation);
                      const destinationEncoded = encodeURIComponent(recommendedLot.address);
                      window.open(`https://www.google.com/maps/dir/?api=1&origin=${originEncoded}&destination=${destinationEncoded}&travelmode=driving`, '_blank');
                    }}
                  >
                    Open in Google Maps
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {showEditProfile && (
        <>
          <div className="dialog-overlay" onClick={handleCancelEdit} />
          <div className="edit-profile-dialog">
            <h2>Edit Profile</h2>
            <form className="edit-profile-form">
              <div className="form-group">
                <label htmlFor="name">Name</label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={userProfile.name}
                  onChange={handleInputChange}
                  placeholder="Enter your name"
                />
              </div>
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={userProfile.email}
                  onChange={handleInputChange}
                  placeholder="Enter your email"
                />
              </div>
              <div className="form-group">
                <label htmlFor="address">Street Address</label>
                <input
                  type="text"
                  id="address"
                  name="address"
                  value={userProfile.address}
                  onChange={handleInputChange}
                  placeholder="Enter your street address in Blacksburg"
                />
              </div>
              <div className="dialog-buttons">
                <button
                  type="button"
                  className="cancel-button"
                  onClick={handleCancelEdit}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="save-button"
                  onClick={handleSaveProfile}
                >
                  Save Changes
                </button>
              </div>
            </form>
          </div>
        </>
      )}
    </div>
  );
};

export default ParkingLotApp;