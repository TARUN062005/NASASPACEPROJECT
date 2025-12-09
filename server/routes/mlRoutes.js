const express = require("express");
const router = express.Router();
const multer = require('multer');

// Import controller methods
const mlController = require("../controller/mlController");
const { authenticateToken } = require("../middleware/authmiddleware");

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  },
  fileFilter: (req, file, cb) => {
    // Check if file is CSV
    if (file.mimetype === 'text/csv' ||
        file.mimetype === 'application/vnd.ms-excel' ||
        file.originalname.toLowerCase().endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'), false);
    }
  }
});

// Apply auth middleware to all routes
router.use(authenticateToken);

// ----------------- Pre-trained Models -----------------
router.post("/predict/toi", mlController.predictTOI);
router.post("/predict/koi", mlController.predictKOI);
router.post("/predict/k2", mlController.predictK2);

// ----------------- Model Information -----------------
router.get("/model-info/:modelType", mlController.getModelInformation);

// ----------------- Enhanced Entry Management -----------------
router.get("/entries/:modelType", mlController.getEntries);
router.put("/entries/:modelType/:entryId", mlController.updateEntry);
router.delete("/entries/:modelType/:entryId", mlController.deleteEntry);
router.get("/export/:modelType", mlController.exportPredictions);

// ----------------- Custom Models -----------------
// MODIFIED: Added upload.single('file') to handle file uploads
router.post("/custom-models", upload.single('file'), mlController.createCustomModel);
router.get("/custom-models", mlController.getCustomModels);
router.put("/custom-models", mlController.updateCustomModel);
router.delete("/custom-models", mlController.removeCustomModel);
router.post("/custom-models/predict", mlController.predictCustomModel);

// ----------------- File Processing -----------------
router.post("/process-file/:modelType", upload.single('file'), (req, res, next) => {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      message: 'No file uploaded. Please select a CSV file.'
    });
  }
  if (!req.file.originalname.toLowerCase().endsWith('.csv')) {
    return res.status(400).json({
      success: false,
      message: 'Only CSV files are allowed'
    });
  }
  next();
}, mlController.processFile);

router.get("/file-status/:jobId", mlController.getFileStatus);

// ----------------- Dashboard & Analytics -----------------
router.get("/dashboard", mlController.getDashboardStats);

// ----------------- Model Performance Tracker Routes -----------------
router.get('/model-performance/:modelType', mlController.getModelPerformance);
router.post('/record-prediction/:modelType', mlController.recordPredictionForTraining);
router.post('/retrain-model/:modelType', mlController.retrainModelWithNewData);
router.get('/training-suggestions/:modelType', mlController.getTrainingSuggestions);
router.post('/predict/:modelType', (req, res) => {
  // This is a generic endpoint for model prediction
  // Your predict functions (predictTOI, predictKOI, etc.) handle this
  // The client-side code sends to /api/ml/predict/toi etc.
  // This route is a placeholder and may not be needed if the others are used directly.
  res.status(400).json({ success: false, message: 'Invalid prediction endpoint. Use /predict/toi, /predict/koi, or /predict/k2.' });
});

module.exports = router;