const express = require('express');
const router = express.Router();
const db = require('../database/db');

// Get user profile
router.get('/:username', (req, res) => {
    const username = req.params.username;

    const sql = 'SELECT username, name, email, address FROM users WHERE username = ?';
    db.get(sql, [username], (err, row) => {
        if (err) {
            console.error('Error querying database:', err);
            return res.status(500).json({ error: 'Internal server error' });
        }
        if (!row) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.json(row);
    });
});

// Update user profile
router.put('/:username', (req, res) => {
    const username = req.params.username;
    const { name, email, address } = req.body;

    const sql = `
      UPDATE users
      SET name = ?, email = ?, address = ?
      WHERE username = ?
    `;

    db.run(sql, [name, email, address, username], function(err) {
        if (err) {
            console.error('Error updating database:', err);
            return res.status(500).json({ error: 'Internal server error' });
        }

        if (this.changes === 0) {
            return res.status(404).json({ error: 'User not found' });
        }

        res.json({ success: true, message: 'Profile updated successfully' });
    });
});

module.exports = router;
