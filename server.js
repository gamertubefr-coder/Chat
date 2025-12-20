const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const geoip = require('geoip-lite');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Servir les fichiers statiques (le frontend)
app.use(express.static('public'));

io.on('connection', (socket) => {
    // 1. Générer un nom aléatoire
    const randomName = `Anonyme-${Math.floor(Math.random() * 10000)}`;

    // 2. Détecter le pays via l'IP
    // Note: En local (localhost), l'IP est souvent ::1 ou 127.0.0.1, donc le pays sera null.
    let ip = socket.handshake.headers['x-forwarded-for'] || socket.request.connection.remoteAddress;
    
    // Nettoyage de l'IP pour le format IPv6 mapé IPv4
    if (ip.substr(0, 7) == "::ffff:") {
      ip = ip.substr(7);
    }

    const geo = geoip.lookup(ip);
    const country = geo ? geo.country : 'Inconnu'; // 'Inconnu' si on est en local

    // Stocker les infos dans le socket de l'utilisateur
    socket.userData = {
        name: randomName,
        country: country
    };

    // Message de bienvenue pour l'utilisateur
    socket.emit('system_message', `Bienvenue ! Tu es connecté en tant que ${randomName} (${country})`);

    // Notifier les autres qu'une personne est arrivée
    socket.broadcast.emit('system_message', `${randomName} a rejoint le chat depuis : ${country}`);

    // 3. Gérer la réception d'un message
    socket.on('chat_message', (msg) => {
        io.emit('chat_message', {
            name: socket.userData.name,
            country: socket.userData.country,
            text: msg
        });
    });

    // Gérer la déconnexion
    socket.on('disconnect', () => {
        io.emit('system_message', `${socket.userData.name} a quitté le chat.`);
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Serveur lancé sur http://localhost:${PORT}`);
});