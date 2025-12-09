// client/src/main.jsx
import React, { createContext, useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import axios from "axios";
import './index.css'

export const AuthContext = createContext();
export const API = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL || "https://nasaspacebackend.onrender.com",
});

// Enhanced security: Only log in development with sensitive data filtered
const secureLog = (...args) => {
  if (import.meta.env.DEV) {
    // Filter out sensitive data
    const filteredArgs = args.map(arg => {
      if (typeof arg === 'string') {
        // Hide passwords, tokens, and sensitive form data
        if (arg.includes('password') || arg.includes('Password') || 
            arg.includes('token') || arg.includes('Token') ||
            arg.includes('otp') || arg.includes('OTP') ||
            arg.includes('currentPassword') || arg.includes('newPassword') ||
            arg.includes('confirmPassword')) {
          return '[SENSITIVE_DATA]';
        }
        // Hide email/phone from logs
        if (arg.includes('@') || (arg.match(/\d/g) && arg.length >= 10)) {
          return arg.replace(/(?<=.{3}).(?=.*@)/g, '*').replace(/(?<=.{2})\d(?=\d{2})/g, '*');
        }
      }
      // For objects, recursively filter sensitive data
      if (typeof arg === 'object' && arg !== null) {
        const filteredObj = {};
        for (const [key, value] of Object.entries(arg)) {
          if (key.toLowerCase().includes('password') || 
              key.toLowerCase().includes('token') ||
              key.toLowerCase().includes('otp')) {
            filteredObj[key] = '[SENSITIVE_DATA]';
          } else if (typeof value === 'string' && (value.includes('@') || (value.match(/\d/g) && value.length >= 10))) {
            filteredObj[key] = value.replace(/(?<=.{3}).(?=.*@)/g, '*').replace(/(?<=.{2})\d(?=\d{2})/g, '*');
          } else {
            filteredObj[key] = value;
          }
        }
        return filteredObj;
      }
      return arg;
    });
    console.log(...filteredArgs);
  }
};

const secureError = (...args) => {
  if (import.meta.env.DEV) {
    const filteredArgs = args.map(arg => {
      if (typeof arg === 'string' && (arg.includes('token') || arg.includes('password') || arg.includes('Token'))) {
        return '[SENSITIVE_DATA]';
      }
      return arg;
    });
    console.error(...filteredArgs);
  }
};

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => {
    const storedToken = localStorage.getItem("token");
    const storedTime = localStorage.getItem("token_time");
    
    // Auto-logout if token is older than 24 hours
    if (storedToken && storedTime) {
      const tokenAge = Date.now() - parseInt(storedTime);
      const maxAge = 24 * 60 * 60 * 1000; // 24 hours
      
      if (tokenAge > maxAge) {
        secureLog("🔐 Token expired, auto-logout");
        localStorage.removeItem("token");
        localStorage.removeItem("token_time");
        localStorage.removeItem("user");
        return null;
      }
    }
    
    return storedToken;
  });
  
  const [loading, setLoading] = useState(true);
  const [needsProfileCompletion, setNeedsProfileCompletion] = useState(false);

  // Enhanced login with timestamp
  // In the main.jsx login function, update this part:

const login = (userData, authToken) => {
  secureLog('🔐 Login called with user data:', userData);
  
  const loginTime = Date.now();
  
  localStorage.setItem("token", authToken);
  localStorage.setItem("token_time", loginTime.toString());
  localStorage.setItem('user', JSON.stringify(userData));
  
  setToken(authToken);
  setUser(userData);
  
  // FIXED: Simplified profile completion check - only check profileCompleted field
  const requiresProfileCompletion = userData.profileCompleted === false;
  secureLog('🔍 Profile completion check:', { 
    profileCompleted: userData.profileCompleted,
    requiresProfileCompletion 
  });
  
  setNeedsProfileCompletion(requiresProfileCompletion);
  
  // FIXED: Force a page reload to ensure clean state
  if (requiresProfileCompletion) {
    setTimeout(() => {
      window.location.href = "/user/profile";
    }, 100);
  } else {
    setTimeout(() => {
      window.location.href = "/user/dashboard";
    }, 100);
  }
};
  // Enhanced logout
  const logout = () => {
    secureLog("🔒 Logging out");
    localStorage.removeItem("token");
    localStorage.removeItem("token_time");
    localStorage.removeItem("user");
    setToken(null);
    setUser(null);
    setNeedsProfileCompletion(false);
  };

  const completeProfile = () => {
    secureLog('✅ Profile completion called');
    setNeedsProfileCompletion(false);
    if (user) {
      const updatedUser = { 
        ...user, 
        profileCompleted: true 
      };
      setUser(updatedUser);
      localStorage.setItem('user', JSON.stringify(updatedUser));
      secureLog('✅ Profile marked as completed:', updatedUser);
    }
  };

  // Auto-logout after inactivity
  useEffect(() => {
    let inactivityTimer;

    const resetInactivityTimer = () => {
      clearTimeout(inactivityTimer);
      // Auto-logout after 30 minutes of inactivity
      inactivityTimer = setTimeout(() => {
        if (token) {
          secureLog("🕒 Auto-logout due to inactivity");
          logout();
        }
      }, 30 * 60 * 1000); // 30 minutes
    };

    const events = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];
    events.forEach(event => {
      document.addEventListener(event, resetInactivityTimer);
    });

    resetInactivityTimer();

    return () => {
      clearTimeout(inactivityTimer);
      events.forEach(event => {
        document.removeEventListener(event, resetInactivityTimer);
      });
    };
  }, [token]);

  // Enhanced request interceptor
  useEffect(() => {
    const requestInterceptor = API.interceptors.request.use(
      (config) => {
        const currentToken = localStorage.getItem("token");
        const tokenTime = localStorage.getItem("token_time");
        
        // Check token expiration
        if (currentToken && tokenTime) {
          const tokenAge = Date.now() - parseInt(tokenTime);
          const maxAge = 24 * 60 * 60 * 1000; // 24 hours
          
          if (tokenAge > maxAge) {
            secureLog("🔐 Token expired in interceptor");
            logout();
            return Promise.reject(new Error("Token expired"));
          }
          
          config.headers.Authorization = `Bearer ${currentToken}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    const responseInterceptor = API.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          secureLog("🔐 Caught 401 Unauthorized, logging out");
          logout();
        }
        return Promise.reject(error);
      }
    );
  
    return () => {
      API.interceptors.request.eject(requestInterceptor);
      API.interceptors.response.eject(responseInterceptor);
    };
  }, []);

  // Real-time user state synchronization
  useEffect(() => {
    const handleStorageChange = () => {
      const storedUser = localStorage.getItem('user');
      if (storedUser) {
        try {
          const userData = JSON.parse(storedUser);
          setUser(userData);
          
          // Update profile completion status when storage changes
          const requiresProfileCompletion = userData.profileCompleted === false;
          setNeedsProfileCompletion(requiresProfileCompletion);
        } catch (error) {
          secureError('❌ Error parsing user data from storage:', error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Enhanced user profile fetch with security - FIXED PROFILE COMPLETION LOGIC
  useEffect(() => {
    const fetchUserProfile = async () => {
      const storedToken = localStorage.getItem('token');
      const storedUser = localStorage.getItem('user');
      const storedTime = localStorage.getItem('token_time');

      secureLog('🔍 Auth check started');
      
      // Check token expiration
      if (storedToken && storedTime) {
        const tokenAge = Date.now() - parseInt(storedTime);
        const maxAge = 24 * 60 * 60 * 1000;
        
        if (tokenAge > maxAge) {
          secureLog("🔐 Token expired on page load");
          logout();
          setLoading(false);
          return;
        }
      }
      
      if (!storedToken) {
        setUser(null);
        setNeedsProfileCompletion(false);
        setLoading(false);
        return;
      }
      
      try {
        setLoading(true);
        
        if (storedUser) {
          try {
            const userData = JSON.parse(storedUser);
            secureLog('✅ Using stored user data:', userData);
            setUser(userData);
            
            // FIXED: Check profile completion status from stored user data
            const requiresProfileCompletion = userData.profileCompleted === false;
            secureLog('🔍 Stored user profile completion check:', {
              profileCompleted: userData.profileCompleted,
              requiresProfileCompletion
            });
            
            setNeedsProfileCompletion(requiresProfileCompletion);
            
            setLoading(false);
            return;
          } catch (parseError) {
            secureError('❌ Error parsing stored user data:', parseError);
          }
        }

        secureLog('🔄 Attempting to fetch user profile from /user/me');
        const userRes = await API.get("/user/me").catch((error) => {
          secureLog('⚠️ /user/me endpoint not available:', error.message);
          return null;
        });

        if (userRes?.data) {
          secureLog('✅ Auth verified via /user/me');
          const userData = userRes.data.user || userRes.data;
          setUser(userData);
          
          // FIXED: Check profile completion status from API response
          const requiresProfileCompletion = userData.profileCompleted === false;
          secureLog('🔍 API user profile completion check:', {
            profileCompleted: userData.profileCompleted,
            requiresProfileCompletion
          });
          
          setNeedsProfileCompletion(requiresProfileCompletion);
          
          localStorage.setItem('user', JSON.stringify(userData));
        } else {
          secureLog('ℹ️ No user data from /user/me, but token is valid');
          const minimalUser = { 
            role: 'USER', 
            email: 'user@example.com',
            name: 'User',
            profileCompleted: false // Default to false for new users
          };
          setUser(minimalUser);
          setNeedsProfileCompletion(true);
          localStorage.setItem('user', JSON.stringify(minimalUser));
        }
      } catch (error) {
        secureError("❌ Failed to fetch user profile:", error.message);
        if (storedUser) {
          try {
            const userData = JSON.parse(storedUser);
            setUser(userData);
            
            // Check profile completion status even on error
            const requiresProfileCompletion = userData.profileCompleted === false;
            setNeedsProfileCompletion(requiresProfileCompletion);
          } catch (parseError) {
            secureError('❌ Error parsing stored user data:', parseError);
          }
        }
      } finally {
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, [token]);

  const authContextValue = { 
    user, 
    token, 
    loading, 
    needsProfileCompletion,
    login, 
    logout, 
    completeProfile,
    API 
  };

  // NASA-themed loading screen
  if (loading) {
    return (
      <div style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #0a0a2a 0%, #1a237e 50%, #311b92 100%)",
        color: "#ffffff",
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        position: "relative",
        overflow: "hidden",
      }}>
        <div style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          background: `
            radial-gradient(1px 1px at 20px 30px, #eee, transparent),
            radial-gradient(1px 1px at 40px 70px, #fff, transparent),
            radial-gradient(0.5px 0.5px at 90px 40px, #fff, transparent)
          `,
          backgroundSize: "200px 200px",
          animation: "twinkle 3s infinite ease-in-out",
        }}></div>
        
        <div style={{
          width: "80px",
          height: "80px",
          background: "linear-gradient(45deg, #ff6d00, #ffab00, #ff6d00)",
          borderRadius: "50%",
          marginBottom: "2rem",
          position: "relative",
          boxShadow: "0 0 30px rgba(255, 109, 0, 0.5)",
          animation: "planetRotate 2s infinite linear",
        }}>
          <div style={{
            position: "absolute",
            width: "15px",
            height: "15px",
            background: "rgba(0, 0, 0, 0.3)",
            borderRadius: "50%",
            top: "20px",
            left: "20px",
          }}></div>
          <div style={{
            position: "absolute",
            width: "10px",
            height: "10px",
            background: "rgba(0, 0, 0, 0.3)",
            borderRadius: "50%",
            top: "50px",
            left: "50px",
          }}></div>
        </div>
        
        <p style={{
          fontSize: "1.2rem",
          fontWeight: "600",
          background: "linear-gradient(45deg, #ffffff, #448aff)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          backgroundClip: "text",
          marginBottom: "1rem",
        }}>
          Initializing Mission Control...
        </p>
        
        <p style={{
          fontSize: "0.9rem",
          opacity: 0.7,
          textAlign: "center",
          maxWidth: "300px",
        }}>
          Loading celestial navigation systems
        </p>

        <style>
          {`
            @keyframes twinkle {
              0%, 100% { opacity: 0.3; }
              50% { opacity: 0.8; }
            }
            @keyframes planetRotate {
              from { transform: rotate(0deg); }
              to { transform: rotate(360deg); }
            }
          `}
        </style>
      </div>
    );
  }

  return (
    <AuthContext.Provider value={authContextValue}>
      {children}
    </AuthContext.Provider>
  );
};

const Root = () => (
  <BrowserRouter>
    <AuthProvider>
      <App />
    </AuthProvider>
  </BrowserRouter>
);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);