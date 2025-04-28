const express = require('express');
const router = express.Router();
const db = require('../database/db');


router.use(express.json());
// // Create database connection
// const dbPath = path.resolve(__dirname, '../parking_app.db');
// const db = new sqlite3.Database(dbPath, (err) => {
//   if (err) {
//     console.error('Error opening database', err);
//   } else {
//     console.log('Connected to SQLite database (Login)');
//     db.run(`CREATE TABLE IF NOT EXISTS users (
//       id INTEGER PRIMARY KEY AUTOINCREMENT,
//       username TEXT UNIQUE,
//       password TEXT
//     )`);
//   }
// });

// Login Route
router.post('/login', (req, res) => {
  const { username, password } = req.body;
  db.get('SELECT * FROM users WHERE username = ? AND password = ?', 
    [username, password], 
    (err, user) => {
      if (err) {
        return res.status(500).json({ success: false, message: 'Database error' });
      }
      if (user) {
        res.json({ success: true, user: { username: user.username } });
      } else {
        res.status(401).json({ success: false, message: 'Invalid credentials' });
      }
    }
  );
});

// Signup Route
router.post('/signup', (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ success: false, message: 'Username and password required' });
  }
  const stmt = db.prepare('INSERT INTO users (username, password) VALUES (?, ?)');
  stmt.run([username, password], function (err) {
    if (err) {
      if (err.code === 'SQLITE_CONSTRAINT') {
        return res.status(409).json({ success: false, message: 'Username already taken' });
      }
      return res.status(500).json({ success: false, message: 'Database error' });
    }
    res.status(201).json({ success: true, user: { username }, message: 'User created successfully' });
  });
});

module.exports = router;
