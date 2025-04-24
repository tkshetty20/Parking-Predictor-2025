const express = require('express');
const cors = require('cors');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

// Create database connection
const dbPath = path.resolve(__dirname, 'parking_app.db');
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('Error opening database', err);
  } else {
    console.log('Connected to SQLite database');
    
    // Create users table if not exists
    db.run(`CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE,
      password TEXT
    )`, (err) => {
      if (err) {
        console.error('Error creating users table', err);
      } else {
        // Insert default admin user if not exists
        db.run(`
          INSERT OR IGNORE INTO users (username, password) 
          VALUES ('admin', 'password123')
        `);
      }
    });
  }
});

// Login Route
app.post('/api/login', (req, res) => {
  const { username, password } = req.body;

  // Query to check user credentials
  db.get('SELECT * FROM users WHERE username = ? AND password = ?', 
    [username, password], 
    (err, user) => {
      if (err) {
        return res.status(500).json({ 
          success: false, 
          message: 'Database error' 
        });
      }

      if (user) {
        res.json({ 
          success: true, 
          user: { username: user.username } 
        });
      } else {
        res.status(401).json({ 
          success: false, 
          message: 'Invalid credentials' 
        });
      }
    }
  );
});

// Sign Up Route
app.post('/api/signup', (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({
      success: false,
      message: 'Username and password are required'
    });
  }

  const stmt = db.prepare('INSERT INTO users (username, password) VALUES (?, ?)');
  stmt.run([username, password], function (err) {
    if (err) {
      if (err.code === 'SQLITE_CONSTRAINT') {
        return res.status(409).json({
          success: false,
          message: 'Username already taken'
        });
      }
      return res.status(500).json({
        success: false,
        message: 'Database error'
      });
    }

    return res.status(201).json({
      success: true,
      user: { username },
      message: 'User created successfully'
    });
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  db.close((err) => {
    if (err) {
      console.error(err.message);
    }
    console.log('Closed the database connection.');
    process.exit(0);
  });
});