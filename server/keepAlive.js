const fetch = require('node-fetch');

const services = [
  'https://nasaspaceproject.onrender.com',
  'https://nasaspacebackend.onrender.com/health',
  'https://nasa-ml-models.onrender.com/health'
];

async function pingServices() {
  console.log('🔄 Pinging services to keep them alive...');
  
  for (const service of services) {
    try {
      const response = await fetch(service, { 
        timeout: 10000,
        headers: { 'User-Agent': 'Keep-Alive-Agent' }
      });
      console.log(`✅ ${service}: ${response.status}`);
    } catch (error) {
      console.log(`❌ ${service}: ${error.message}`);
    }
  }
}

// Ping every 10 minutes (600000 ms)
setInterval(pingServices, 10 * 60 * 1000);

// Initial ping after 30 seconds
setTimeout(pingServices, 30000);

module.exports = { pingServices };