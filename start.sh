#!/bin/bash
mysqld_safe --datadir=/var/lib/mysql &

until mysqladmin ping -h localhost --silent; do
  echo "Waiting for MySQL..."
  sleep 2
done

mysql -u root <<EOF
CREATE DATABASE IF NOT EXISTS mydb;
CREATE USER IF NOT EXISTS 'myuser'@'localhost' IDENTIFIED BY 'mypassword';
GRANT ALL PRIVILEGES ON mydb.* TO 'myuser'@'localhost';
FLUSH PRIVILEGES;
EOF

# Start Flask
cd /app/persistence && npm run flask &

# Start Express
cd /app/persistence && npm run serve &

# Start React
cd /app/parking-frontend && npm start
