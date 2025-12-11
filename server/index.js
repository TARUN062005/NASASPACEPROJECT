const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const path = require("path");

const { ConnectDb } = require("./utils/dbConnector");

// Routers
const userLoginRouter = require("./routes/userLogin");
const mlRoutes = require("./routes/mlRoutes");
const compareRoutes = require('./routes/compareRouter');
const otherLogin = require("./routes/otherLogin");

// Load environment variables
dotenv.config();

const app = express();

// Keep alive for production
if (process.env.NODE_ENV === 'production') {
  try {
    require('./keepAlive');
    console.log('✅ Keep-alive service started');
  } catch (error) {
    console.log('⚠️ Keep-alive service not found, skipping...');
  }
}

// ----------------- Middleware -----------------
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve static files from React app
app.use(express.static(path.join(__dirname, 'public')));

// CORS configuration
app.use(cors({
  origin: function (origin, callback) {
    // Allow requests from same origin (since frontend and backend are combined)
    if (!origin) return callback(null, true);
    
    const allowedOrigins = [
      'https://nasaspaceproject.onrender.com',
      'http://localhost:5173'
    ];
    
    if (allowedOrigins.includes(origin) || process.env.NODE_ENV === 'development') {
      callback(null, true);
    } else {
      console.log(`CORS blocked: ${origin}`);
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-User-ID']
}));

// ----------------- ML Service URLs -----------------
const ML_SERVICES = {
    TOI: process.env.TOI_URL || 'https://nasa-ml-models.onrender.com/toi',
    KOI: process.env.KOI_URL || 'https://nasa-ml-models.onrender.com/koi',
    K2: process.env.K2_URL || 'https://nasa-ml-models.onrender.com/k2',
    CUSTOM: process.env.CUSTOM_URL || 'https://nasa-ml-models.onrender.com/custom'
};

// ----------------- Routes -----------------
// Health endpoint
app.get("/health", (req, res) => {
    res.json({
        status: "healthy",
        timestamp: new Date().toISOString(),
        service: "NASA Combined API",
        environment: process.env.NODE_ENV || 'development',
        uptime: process.uptime(),
        ml_services: Object.keys(ML_SERVICES)
    });
});

// API Routes
app.use("/api/user", userLoginRouter);
app.use("/api/other", otherLogin);
app.use("/api/ml", mlRoutes);
app.use("/api/compare", compareRoutes);

// Info endpoint
app.get("/api/info", (req, res) => {
    res.json({
        message: "NASA Exoplanet Detection API",
        version: "2.0.0",
        deployed: true,
        endpoints: {
            health: "/api/health",
            ml: "/api/ml",
            user: "/api/user",
            other: "/api/other",
            compare: "/api/compare"
        }
    });
});

// All other GET requests return React app
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ----------------- Error Handling -----------------
app.use((err, req, res, next) => {
    console.error("🚨 Server Error:", err.message);
    res.status(500).json({
        success: false,
        message: "Internal server error",
        error: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
    });
});

// ----------------- Start Server -----------------
const startServer = async () => {
    try {
        await ConnectDb();
        const port = process.env.PORT || 10000;

        app.listen(port, () => {
            console.log(`
✅ Server running on port ${port}
🌍 Environment: ${process.env.NODE_ENV || 'development'}
🔗 URL: https://nasaspaceproject.onrender.com
📊 Database: Connected
🔬 ML Services:
    - TOI: ${ML_SERVICES.TOI}
    - KOI: ${ML_SERVICES.KOI}
    - K2: ${ML_SERVICES.K2}
    - Custom: ${ML_SERVICES.CUSTOM}
🚀 Ready to receive requests!
            `);
        });
    } catch (err) {
        console.error("❌ Failed to start server:", err.message);
        process.exit(1);
    }
};

startServer();