import React, { useState } from 'react';
import './Login.css';

const Login = ({ onLogin, onSwitchToSignUp }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    if (!username || !password) {
      setError('Please enter both username and password');
      return;
    }
  
    try {
      const response = await fetch('http://localhost:3000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
  
      const data = await response.json();
  
      if (response.ok && data.success) {
        setError('');
        onLogin(data.user);  // Pass user info back to parent
      } else {
        setError(data.message || 'Login failed');
      }
    } catch (error) {
      setError('Error connecting to server');
    }
  };

  const handleLogin = (user) => {
    onLogin(user);
  };

  return (
    <div className="login-container">
      <form className="login-form" onSubmit={handleSubmit}>
        <h2>VT Parking App Login</h2>
        {error && <div className="error-message">{error}</div>}
        <div className="form-group">
          <label htmlFor="username">Username</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter your username"
          />
        </div>
        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
          />
        </div>
        <button type="submit" className="login-button">Login</button>
        <p>
          Don’t have an account?{' '}
          <span 
            onClick={onSwitchToSignUp} 
            style={{ color: '#0077cc', cursor: 'pointer' }}
          >
            Sign up
          </span>
        </p>
      </form>
    </div>
  );
};

export default Login;