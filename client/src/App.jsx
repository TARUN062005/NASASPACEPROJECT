import React, { useContext } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { AuthContext } from "./main.jsx";

// Common components
import Header from "./components/common/header";
import LandingPage from "./components/common/landingPage";
import AuthPage from "./components/common/authPage.jsx";
import ProtectedRoutes from "./components/common/ProtectedRoutes.jsx";
import NotFoundPage from "./components/common/NotFoundPage.jsx";
import AIChatbot from "./components/common/AIChatbot.jsx";

// Dashboards
import UserDashboard from "./components/user/userDashboard.jsx";
import Matching from './components/common/matching.jsx';

const App = () => {
    const { user, needsProfileCompletion } = useContext(AuthContext);

    const appContainerStyle = {
        minHeight: "100vh",
        background: "linear-gradient(135deg, #0a0a2a 0%, #1a237e 50%, #311b92 100%)",
        color: "#ffffff",
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        position: "relative",
    };

    const starsStyle = {
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        background: `
            radial-gradient(1px 1px at 20px 30px, #eee, transparent),
            radial-gradient(1px 1px at 40px 70px, #fff, transparent),
            radial-gradient(0.5px 0.5px at 90px 40px, #fff, transparent),
            radial-gradient(0.5px 0.5px at 130px 80px, #fff, transparent)
        `,
        backgroundSize: "200px 200px",
        animation: "twinkle 8s infinite ease-in-out",
        pointerEvents: "none",
        zIndex: 0,
    };

    const mainContentStyle = {
        position: "relative",
        zIndex: 1,
        minHeight: "calc(100vh - 80px)",
    };

    return (
        <div style={appContainerStyle}>
            <div style={starsStyle}></div>
            <Header />
            <main style={mainContentStyle}>
                <Routes>
                    {/* Public Routes */}
                    <Route path="/" element={<LandingPage />} />

                    {/* Auth (Login/Register) */}
                    <Route
                        path="/auth/:userType/:mode"
                        element={
                            user ? (
                                <Navigate
                                    to={
                                        needsProfileCompletion
                                            ? "/user/profile"
                                            : "/user/dashboard"
                                    }
                                    replace
                                />
                            ) : (
                                <AuthPage />
                            )
                        }
                    />

                    {/* Protected: USER routes - ALL user routes handled by UserDashboard */}
                    <Route element={<ProtectedRoutes allowedRoles={["USER"]} />}>
                        <Route path="/user/*" element={<UserDashboard />} />
                        <Route path="/user/compare" element={<Matching />} />
                    </Route>

                    {/* 404 Fallback - NASA Themed */}
                    <Route path="*" element={<NotFoundPage />} />
                </Routes>
            </main>

            {/* AIChatbot - Only show when user is logged in */}
            {user && <AIChatbot />}

            {/* Global Styles */}
            <style>
                {`
                    @keyframes twinkle {
                        0%, 100% { opacity: 0.3; }
                        50% { opacity: 0.8; }
                    }
                    
                    * {
                        box-sizing: border-box;
                        margin: 0;
                        padding: 0;
                    }
                    
                    body {
                        overflow-x: hidden;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    }
                    
                    /* Smooth scrolling */
                    html {
                        scroll-behavior: smooth;
                    }

                    /* Custom scrollbar */
                    ::-webkit-scrollbar {
                        width: 8px;
                    }
                    
                    ::-webkit-scrollbar-track {
                        background: #1a237e;
                    }
                    
                    ::-webkit-scrollbar-thumb {
                        background: #448aff;
                        border-radius: 4px;
                    }
                    
                    ::-webkit-scrollbar-thumb:hover {
                        background: #2979ff;
                    }
                `}
            </style>
        </div>
    );
};

export default App;