const express = require("express");
const router = express.Router();
const { authenticateToken } = require("../middleware/authmiddleware");

// Try to load controllers, but provide fallback if not available
let authController = {};
let dashboardController = {};

try {
  const authControllerModule = require("../controller/authController");
  authController = authControllerModule;
} catch (error) {
  console.log('⚠️  authController not found, using fallback');
}

try {
  const dashboardControllerModule = require("../controller/dashboardController");
  dashboardController = dashboardControllerModule;
} catch (error) {
  console.log('⚠️  dashboardController not found, using fallback');
}

// Fallback handler
const fallbackHandler = (req, res) => {
  res.status(501).json({
    success: false,
    message: 'This endpoint is not yet implemented',
    path: req.path
  });
};

// Auth routes for "other" role
router.post("/register", authController.registerUser ? 
  (req, res) => authController.registerUser(req, res, "other") : fallbackHandler);
router.post("/verify-otp", authController.verifyOtp || fallbackHandler);
router.post("/complete-registration", authController.completeRegistration || fallbackHandler);
router.post("/login", authController.loginUser ? 
  (req, res) => authController.loginUser(req, res, "other") : fallbackHandler);
router.post("/forgot-password", authController.forgotPassword || fallbackHandler);
router.post("/reset-password", authController.resetPassword || fallbackHandler);
router.post("/google-login", authController.googleLogin || fallbackHandler);
router.post("/facebook-login", authController.facebookLogin || fallbackHandler);
router.post("/twitter-login", authController.twitterLogin || fallbackHandler);

// Protected dashboard routes
router.get("/profile", authenticateToken, dashboardController.getProfile || fallbackHandler);
router.put("/profile", authenticateToken, dashboardController.updateProfile || fallbackHandler);

// Account management routes
router.get("/linked-accounts", authenticateToken, dashboardController.getLinkedAccounts || fallbackHandler);
router.post("/link-account", authenticateToken, authController.linkAccount || fallbackHandler);
router.post("/verify-linked-account", authenticateToken, authController.verifyLinkedAccount || fallbackHandler);
router.post("/make-account-primary", authenticateToken, authController.makeAccountPrimary || fallbackHandler);
router.delete("/linked-accounts/:linkedAccountId", authenticateToken, dashboardController.removeLinkedAccount || fallbackHandler);

// OTP protected operations
router.post("/send-otp-for-operation", authenticateToken, dashboardController.sendOtpForOperation || fallbackHandler);
router.put("/change-password", authenticateToken, dashboardController.changePassword || fallbackHandler);
router.post("/request-deletion", authenticateToken, dashboardController.requestAccountDeletion || fallbackHandler);
router.delete("/account", authenticateToken, dashboardController.deleteAccount || fallbackHandler);

module.exports = router;
