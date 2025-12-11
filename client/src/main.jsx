import React, { createContext, useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import axios from "axios";
import './index.css'

export const AuthContext = createContext();

// Log API URL for debugging
console.log('🌐 VITE_BACKEND_URL:', import.meta.env.VITE_BACKEND_URL);

export const API = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL || "https://nasaspacebackend.onrender.com",
  timeout: 10000,
  withCredentials: true
});

// Debug: Log all requests and responses
API.interceptors.request.use(
  (config) => {
    console.log(`🌐 ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);
    return config;
  },
  (error) => {
    console.error('🌐 Request Error:', error);
    return Promise.reject(error);
  }
);

API.interceptors.response.use(
  (response) => {
    console.log(`✅ ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ Response Error:', {
      url: error.config?.url,
      status: error.response?.status,
      message: error.message
    });
    return Promise.reject(error);
  }
);

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => {
    return localStorage.getItem("token");
  });
  
  const [loading, setLoading] = useState(true);
  const [needsProfileCompletion, setNeedsProfileCompletion] = useState(false);

  const login = (userData, authToken) => {
    console.log('🔐 Login called:', { 
      email: userData.email,
      profileCompleted: userData.profileCompleted 
    });
    
    localStorage.setItem("token", authToken);
    localStorage.setItem("user", JSON.stringify(userData));
    
    setToken(authToken);
    setUser(userData);
    
    // Check profile completion
    const requiresProfileCompletion = userData.profileCompleted === false;
    setNeedsProfileCompletion(requiresProfileCompletion);
    
    // Force redirect
    if (requiresProfileCompletion) {
      window.location.href = "/user/profile";
    } else {
      window.location.href = "/user/dashboard";
    }
  };

  const logout = () => {
    console.log("🔒 Logging out");
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setToken(null);
    setUser(null);
    setNeedsProfileCompletion(false);
    window.location.href = "/";
  };

  const completeProfile = () => {
    console.log('✅ Profile completion called');
    setNeedsProfileCompletion(false);
    if (user) {
      const updatedUser = { 
        ...user, 
        profileCompleted: true 
      };
      setUser(updatedUser);
      localStorage.setItem('user', JSON.stringify(updatedUser));
    }
  };

  // Auto-logout after inactivity
  useEffect(() => {
    let inactivityTimer;

    const resetInactivityTimer = () => {
      clearTimeout(inactivityTimer);
      inactivityTimer = setTimeout(() => {
        if (token) {
          console.log("🕒 Auto-logout due to inactivity");
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

  // Add authorization header to all requests
  useEffect(() => {
    const requestInterceptor = API.interceptors.request.use(
      (config) => {
        const currentToken = localStorage.getItem("token");
        if (currentToken) {
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
          console.log("🔐 401 Unauthorized - logging out");
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

  // Fetch user profile on load
  useEffect(() => {
    const fetchUserProfile = async () => {
      console.log('🔍 Auth check started');
      
      const storedToken = localStorage.getItem('token');
      const storedUser = localStorage.getItem('user');
      
      if (!storedToken) {
        console.log('🔍 No token found');
        setUser(null);
        setNeedsProfileCompletion(false);
        setLoading(false);
        return;
      }
      
      try {
        setLoading(true);
        
        // Try to use stored user data first
        if (storedUser) {
          try {
            const userData = JSON.parse(storedUser);
            console.log('✅ Using stored user data');
            setUser(userData);
            
            const requiresProfileCompletion = userData.profileCompleted === false;
            setNeedsProfileCompletion(requiresProfileCompletion);
            
            // Validate token with backend
            console.log('🔄 Validating token with backend...');
            const userRes = await API.get("/user/me").catch((error) => {
              console.log('⚠️ Token validation failed:', error.message);
              // If token is invalid, logout
              if (error.response?.status === 401) {
                logout();
              }
              return null;
            });
            
            if (userRes?.data) {
              console.log('✅ Token is valid, updating user data');
              const freshUserData = userRes.data.user || userRes.data;
              setUser(freshUserData);
              localStorage.setItem('user', JSON.stringify(freshUserData));
              
              const requiresCompletion = freshUserData.profileCompleted === false;
              setNeedsProfileCompletion(requiresCompletion);
            }
            
          } catch (parseError) {
            console.error('❌ Error parsing stored user data:', parseError);
          }
        } else {
          // No stored user, fetch from API
          console.log('🔄 Fetching user profile from API...');
          const userRes = await API.get("/user/me").catch((error) => {
            console.log('❌ Failed to fetch user profile:', error.message);
            return null;
          });

          if (userRes?.data) {
            console.log('✅ User profile fetched successfully');
            const userData = userRes.data.user || userRes.data;
            setUser(userData);
            
            const requiresProfileCompletion = userData.profileCompleted === false;
            setNeedsProfileCompletion(requiresProfileCompletion);
            
            localStorage.setItem('user', JSON.stringify(userData));
          }
        }
      } catch (error) {
        console.error("❌ Error in fetchUserProfile:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, []);

  // Debug: Log state changes
  useEffect(() => {
    console.log('🔄 Auth State Changed:', {
      user: user?.email,
      loading,
      needsProfileCompletion
    });
  }, [user, loading, needsProfileCompletion]);

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