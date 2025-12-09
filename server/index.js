const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");

const { ConnectDb } = require("./utils/dbConnector");

// Routers
const userLoginRouter = require("./routes/userLogin");
const mlRoutes = require("./routes/mlRoutes");
const compareRoutes = require('./routes/compareRouter');
const otherLogin = require("./routes/otherLogin");

// Load environment variables
dotenv.config();

// ML Service URLs - FOR PRODUCTION
const ML_SERVICES = {
    TOI: process.env.TOI_URL || 'https://nasa-ml-models.onrender.com/toi',
    KOI: process.env.KOI_URL || 'https://nasa-ml-models.onrender.com/koi',
    K2: process.env.K2_URL || 'https://nasa-ml-models.onrender.com/k2',
    CUSTOM: process.env.CUSTOM_URL || 'https://nasa-ml-models.onrender.com/custom'
};

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

// ----------------- Enhanced Middleware -----------------
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// CORS configuration - SIMPLIFIED VERSION
const allowedOrigins = [
  'https://nasaspaceproject.onrender.com',
  'http://localhost:5173',
  'https://nasaspacebackend.onrender.com'
];

app.use(cors({
  origin: function (origin, callback) {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) return callback(null, true);
    
    if (allowedOrigins.indexOf(origin) !== -1 || process.env.NODE_ENV === 'development') {
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

// ----------------- Basic Routes -----------------
app.get("/", (req, res) => {
    res.json({
        message: "NASA Exoplanet Detection API is running...",
        version: "2.0.0",
        deployed: true,
        frontend: "https://nasaspaceproject.onrender.com",
        backend: "https://nasaspacebackend.onrender.com",
        endpoints: {
            health: "/health",
            ml: "/api/ml",
            user: "/user",
            other: "/other",
            compare: "/api/compare"
        }
    });
});

// Health endpoint for pinging
app.get("/health", (req, res) => {
    res.json({
        status: "healthy",
        timestamp: new Date().toISOString(),
        service: "NASA Backend API",
        environment: process.env.NODE_ENV || 'development',
        uptime: process.uptime(),
        ml_services: Object.keys(ML_SERVICES)
    });
});

// ----------------- API Routes -----------------
app.use("/user", userLoginRouter);
app.use("/other", otherLogin);
app.use("/api/ml", mlRoutes);
app.use("/api", compareRoutes);

// ----------------- Error Handling -----------------
app.use((err, req, res, next) => {
    console.error("🚨 Server Error:", {
        message: err.message,
        url: req.url,
        method: req.method
    });

    // Handle CORS errors
    if (err.message === 'Not allowed by CORS') {
        return res.status(403).json({
            success: false,
            message: "CORS error: Origin not allowed"
        });
    }

    res.status(500).json({
        success: false,
        message: "Internal server error",
        error: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
    });
});

// ----------------- 404 Handler -----------------
app.use((req, res) => {
    res.status(404).json({
        success: false,
        message: "Route not found",
        path: req.path,
        available_routes: {
            ml: "/api/ml/*",
            user: "/user/*",
            other: "/other/*",
            compare: "/api/compare",
            health: "/health"
        }
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
🔗 Frontend: https://nasaspaceproject.onrender.com
🔗 Backend URL: https://nasaspacebackend.onrender.com
🔬 ML Services:
    - TOI: ${ML_SERVICES.TOI}
    - KOI: ${ML_SERVICES.KOI}
    - K2: ${ML_SERVICES.K2}
    - Custom: ${ML_SERVICES.CUSTOM}
📊 Database: Connected
🚀 Ready to receive requests!
            `);
        });
    } catch (err) {
        console.error("❌ Failed to start server:", err.message);
        process.exit(1);
    }
};

startServer();