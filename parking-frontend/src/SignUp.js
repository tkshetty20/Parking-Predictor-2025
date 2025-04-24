import React, { useState } from 'react';
import Login from './Login';
import './Login.css';

const SignUp = ({ onSignUp, onSwitchToLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!username || !password) {
      setError('Please fill in all fields');
      return;
    }

    try {
      const res = await fetch('http://localhost:5000/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      const data = await res.json();
      if (res.ok && data.success) {
        setError('');
        onSignUp(data.user); // log user in
      } else {
        setError(data.message || 'Signup failed');
      }
    } catch (err) {
      setError('Error signing up');
    }
  };

  return (
    <div className="login-container">
      <form className="login-form" onSubmit={handleSubmit}>
        <h2>Sign Up</h2>

        {error && <div className="error-message">{error}</div>}

        <div className="form-group">
          <label htmlFor="signup-username">Username</label>
          <input
            id="signup-username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter your username"
          />
        </div>

        <div className="form-group">
          <label htmlFor="signup-password">Password</label>
          <input
            id="signup-password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
          />
        </div>

        <button type="submit" className="login-button">Sign Up</button>

        <p className="switch-form-text">
          Already have an account?{' '}
          <span
            onClick={onSwitchToLogin}
            style={{ color: '#0077cc', cursor: 'pointer' }}
          >
            Log in
          </span>
        </p>
      </form>
    </div>
  );
};

export default SignUp;