const express = require("express");
const router = express.Router();
const { authenticateToken } = require("../middleware/authmiddleware");
const { 
  registerUser, 
  loginUser, 
  verifyOtp, 
  completeRegistration,
  forgotPassword,
  resetPassword,
  googleLogin,
  facebookLogin,
  twitterLogin,
  linkAccount,
  verifyLinkedAccount,
  makeAccountPrimary
} = require("../controller/authController");

const { 
  getProfile, 
  updateProfile, 
  getLinkedAccounts,
  removeLinkedAccount,
  changePassword,
  requestAccountDeletion,
  deleteAccount,
  sendOtpForOperation
} = require("../controller/dashboardController");

// Auth routes for "user" role
router.post("/register", (req, res) => registerUser(req, res, "user"));
router.post("/verify-otp", verifyOtp);
router.post("/complete-registration", completeRegistration);
router.post("/login", (req, res) => loginUser(req, res, "user"));
router.post("/forgot-password", forgotPassword);
router.post("/reset-password", resetPassword);
router.post("/google-login", googleLogin);
router.post("/facebook-login", facebookLogin);
router.post("/twitter-login", twitterLogin);

// ADD THIS /me ENDPOINT
router.get("/me", authenticateToken, async (req, res) => {
  try {
    const user = req.user;
    
    console.log('🔍 /user/me endpoint called for user:', user.email);
    
    // Return user data (exclude sensitive fields)
    res.json({
      success: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name || user.email.split('@')[0], // Use name or first part of email
        role: user.role || 'USER',
        profileCompleted: user.profileCompleted || false,
        // Add any other safe fields you want to expose
        createdAt: user.createdAt,
        updatedAt: user.updatedAt
      }
    });
  } catch (error) {
    console.error('❌ Error in /user/me endpoint:', error);
    res.status(500).json({
      success: false,
      message: 'Server error fetching user profile'
    });
  }
});

// Protected dashboard routes
router.get("/profile", authenticateToken, getProfile);
router.put("/profile", authenticateToken, updateProfile);

// Account management routes
router.get("/linked-accounts", authenticateToken, getLinkedAccounts);
router.post("/link-account", authenticateToken, linkAccount);
router.post("/verify-linked-account", authenticateToken, verifyLinkedAccount);
router.post("/make-account-primary", authenticateToken, makeAccountPrimary);
router.delete("/linked-accounts/:linkedAccountId", authenticateToken, removeLinkedAccount);

// OTP protected operations
router.post("/send-otp-for-operation", authenticateToken, sendOtpForOperation);
router.put("/change-password", authenticateToken, changePassword);
router.post("/request-deletion", authenticateToken, requestAccountDeletion);
router.delete("/account", authenticateToken, deleteAccount);

module.exports = router;