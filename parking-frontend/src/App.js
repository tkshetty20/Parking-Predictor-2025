import React, { useState, useEffect } from 'react';
import './App.css';
import Login from './Login';
import SignUp from './SignUp';
import HomePage from './HomePage';

import 'leaflet/dist/leaflet.css';

function App() {
  const [user, setUser] = useState(null); // user is null by default (not logged in)
  const [showLogin, setShowLogin] = useState(true);

  // Load user from localStorage on first render
  useEffect(() => {
    const savedUser = localStorage.getItem('parking-user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const handleLogin = (userData) => {
    setUser(userData); // e.g., { username: 'admin' }
    localStorage.setItem('parking-user', JSON.stringify(userData));
  };

  const handleLogout = () => {
    setUser(null); // log out
    localStorage.removeItem('parking-user');
  };

  return (
    <div className="App">
      {!user ? (
        showLogin ? (
          <Login onLogin={handleLogin} onSwitchToSignUp={() => setShowLogin(false)} />
        ) : (
          <SignUp onSignUp={handleLogin} onSwitchToLogin={() => setShowLogin(true)} />
        )
      ) : (
        <HomePage user={user} onLogout={handleLogout} />
      )}
    </div>
  );
}

export default App;