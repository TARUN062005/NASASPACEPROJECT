const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");

const { ConnectDb } = require("./utils/dbConnector");

// Routers
const userLoginRouter = require("./routes/userLogin");
const mlRoutes = require("./routes/mlRoutes");
const compareRoutes = require('./routes/compareRouter'); // Import the new router

// Load environment variables
dotenv.config();

// ML Service URLs configuration
const ML_SERVICES = {
    TOI: process.env.TOI_URL || 'http://localhost:5001',
    KOI: process.env.KOI_URL || 'http://localhost:5002',
    K2: process.env.K2_URL || 'http://localhost:5003',
    CUSTOM: process.env.CUSTOM_URL || 'http://localhost:5004'
};

// For combined deployment (optional)
const ML_BASE_URL = process.env.ML_BASE_URL || 'http://localhost:10000';

const app = express();

// ----------------- Enhanced Middleware -----------------
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
// CORS configuration for production and development
const corsOptions = {
    origin: process.env.NODE_ENV === 'production' 
        ? (process.env.CLIENT_URL || ['https://your-frontend-url.vercel.app']).split(',')
        : ['http://localhost:5173'],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-User-ID']
};
app.use(cors(corsOptions));

// ----------------- Enhanced Routes -----------------
app.get("/", (req, res) => {
    res.json({
        message: "NASA Exoplanet Detection API is running...",
        version: "2.0.0",
        models: {
            pre_trained: ["TOI", "KOI", "K2"],
            custom: "User-trained models"
        },
        features: {
            image_generation: true,
            bulk_predictions: true,
            data_export: true,
            real_time_charts: true
        }
    });
});

// Enhanced health check for all ML services
app.get("/health/all", async (req, res) => {
    try {
        const { checkMLServices } = require('./utils/mlUtils');
        const healthStatus = await checkMLServices();

        healthStatus.system = {
            node_version: process.version,
            platform: process.platform,
            memory_usage: process.memoryUsage(),
            uptime: process.uptime(),
            environment: process.env.NODE_ENV || 'development'
        };

        res.json(healthStatus);
    } catch (error) {
        res.status(500).json({
            success: false,
            message: "Health check failed",
            error: error.message
        });
    }
});

app.use("/user", userLoginRouter);
app.use("/api/ml", mlRoutes);
app.use("/api", compareRoutes); // Add the new comparison route here

// ----------------- Enhanced Error Handling Middleware -----------------
app.use((err, req, res, next) => {
    console.error("🚨 Server Error:", {
        message: err.message,
        stack: err.stack,
        url: req.url,
        method: req.method,
        ip: req.ip
    });

    res.status(500).json({
        success: false,
        message: "Internal server error",
        error: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong',
        request_id: req.id || Date.now().toString(36)
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
            compare: "/api/compare",
            health: "/health/all"
        }
    });
});

// ----------------- Graceful Shutdown -----------------
process.on('SIGTERM', async () => {
    console.log('🔄 SIGTERM received, starting graceful shutdown...');
    process.exit(0);
});

process.on('SIGINT', async () => {
    console.log('🔄 SIGINT received, starting graceful shutdown...');
    process.exit(0);
});

// ----------------- Start Server -----------------
const startServer = async () => {
    try {
        await ConnectDb();
        const port = process.env.PORT || 5000;

        app.listen(port, () => {
            console.log(`
✅ Server running on port ${port}
🔬 ML Services Status:
    - TOI Model: ${process.env.TOI_SERVICE_URL || 'http://localhost:5001'}
    - KOI Model: ${process.env.KOI_SERVICE_URL || 'http://localhost:5002'}
    - K2 Model: ${process.env.K2_SERVICE_URL || 'http://localhost:5003'}
    - Custom Model: ${process.env.CUSTOM_SERVICE_URL || 'http://localhost:5004'}
📊 Database: Connected
🌐 API Base: http://localhost:${port}
🚀 Features: Image Generation, Bulk Predictions, Data Export, Real-time Charts
      `);
        });
    } catch (err) {
        console.error("❌ Failed to start server:", err.message);
        process.exit(1);
    }
};

startServer();