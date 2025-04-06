import React, { useState } from 'react';
import axios from 'axios';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import './ParkingLotApp.css';

// Fix for default marker icon issue
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

const ParkingLotApp = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  // Parking lots state
  const [parkingLots, setParkingLots] = useState([
    {
      name: 'Perry Street',
      spotsAvailable: 0,
      isFull: false,
      location: [37.2295, -80.4139]
    },
    {
      name: 'North End',
      spotsAvailable: 0,
      isFull: false,
      location: [37.2280, -80.4230]
    },
    {
      name: 'Kent Street',
      spotsAvailable: 0,
      isFull: false,
      location: [37.2310, -80.4180]
    }
  ]);

  const [startingLocation, setStartingLocation] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');

    try {
      const response = await axios.post('http://localhost:5000/api/login', {
        username,
        password
      });

      if (response.data.success) {
        setIsLoggedIn(true);
      }
    } catch (err) {
      setError('Invalid credentials');
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUsername('');
    setPassword('');
  };

  // Find the lot with most available spots
  const recommendedLot = parkingLots
    .filter(lot => !lot.isFull)
    .reduce((prev, current) => 
      (prev.spotsAvailable > current.spotsAvailable) ? prev : current
    , { name: 'No available lots', spotsAvailable: 0 });

  return (
    <div className="parking-app">
      <header className="app-header">
        <h1>VT Parking Prediction</h1>
        {!isLoggedIn ? (
          <form onSubmit={handleLogin} style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <input 
              type="text" 
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required 
              style={{
                padding: '8px',
                marginRight: '10px'
              }}
            />
            <input 
              type="password" 
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required 
              style={{
                padding: '8px',
                marginRight: '10px'
              }}
            />
            <button 
              type="submit" 
              className="login-btn"
            >
              Login
            </button>
            {error && <p style={{color: 'white', margin: 0}}>{error}</p>}
          </form>
        ) : (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '15px'
          }}>
            <span>Welcome {username}!</span>
            <button 
              onClick={handleLogout} 
              className="login-btn"
            >
              Logout
            </button>
          </div>
        )}
      </header>

      {isLoggedIn && (
        <div className="app-content">
          <div className="sidebar">
            <div className="location-selector">
              <label>Starting Location?</label>
              <select 
                value={startingLocation}
                onChange={(e) => setStartingLocation(e.target.value)}
              >
                <option value="">Select Location</option>
                <option value="campus-center">Campus Center</option>
                <option value="library">Library</option>
              </select>
            </div>

            <div className="parking-lots">
              {parkingLots.map((lot) => (
                <div 
                  key={lot.name} 
                  className={`lot-info ${lot.isFull ? 'lot-full' : 'lot-available'}`}
                >
                  <h3>{lot.name}</h3>
                  <p>
                    {lot.isFull 
                      ? 'Lot Full!' 
                      : `${lot.spotsAvailable} Spots Available`
                    }
                  </p>
                </div>
              ))}
            </div>

            {recommendedLot && (
              <div className="recommendation">
                <h2>Recommended Lot</h2>
                <p>{recommendedLot.name} - {recommendedLot.spotsAvailable} Spots</p>
              </div>
            )}
          </div>

          <MapContainer 
            center={[37.2295, -80.4139]} 
            zoom={13} 
            className="map-container"
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
          </MapContainer>
        </div>
      )}
    </div>
  );
};

export default ParkingLotApp;