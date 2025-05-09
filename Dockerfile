FROM node:18

RUN apt-get update && \
    apt-get install -y python3 python3-pip default-mysql-server && \
    apt-get clean

WORKDIR /app
COPY . .

WORKDIR /app/persistence
RUN npm install --build-from-source sqlite3
RUN pip install --break-system-packages -r ml-model/requirements.txt

WORKDIR /app/parking-frontend
RUN npm install

WORKDIR /app
RUN chmod +x start.sh

VOLUME /var/lib/mysql

EXPOSE 3000 3002 5000 3306

CMD ["./start.sh"]
