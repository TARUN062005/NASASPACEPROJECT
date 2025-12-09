// src/components/common/header.jsx
import React, { useContext, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AuthContext } from "../../main";

const Header = () => {
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate();
  const [showDropdown, setShowDropdown] = useState(false);

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  // NASA-inspired styles
  const headerStyle = {
    background: "linear-gradient(135deg, #0b3d91 0%, #1a237e 100%)",
    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3)",
    padding: "1rem 2rem",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    position: "sticky",
    top: 0,
    zIndex: 1000,
    borderBottom: "2px solid #ff6d00",
  };

  const navStyle = {
    display: "flex",
    alignItems: "center",
    gap: "2rem",
  };

  const dropdownStyle = {
    position: "absolute",
    top: "100%",
    right: 0,
    background: "rgba(13, 19, 33, 0.95)",
    backdropFilter: "blur(10px)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    borderRadius: "8px",
    padding: "0.5rem",
    minWidth: "200px",
    boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
    zIndex: 1001,
  };

  const dropdownItemStyle = {
    display: "block",
    width: "100%",
    textAlign: "left",
    padding: "0.75rem 1rem",
    color: "#e3f2fd",
    textDecoration: "none",
    borderRadius: "4px",
    transition: "all 0.3s ease",
    border: "none",
    background: "transparent",
    cursor: "pointer",
  };

  return (
    <header style={headerStyle}>
      {/* NASA-themed Logo */}
      <Link
        to="/"
        style={{
          textDecoration: "none",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
        }}
      >
        <div style={{
          width: "40px",
          height: "40px",
          background: "linear-gradient(45deg, #ff6d00, #ffab00)",
          borderRadius: "50%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "white",
          fontWeight: "bold",
          fontSize: "1.2rem",
          boxShadow: "0 0 15px rgba(255, 109, 0, 0.5)"
        }}>
          🌌
        </div>
        <span style={{
          fontSize: "1.4rem",
          fontWeight: "bold",
          background: "linear-gradient(45deg, #ffffff, #e3f2fd)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          backgroundClip: "text",
        }}>
          ExoDiscover AI
        </span>
      </Link>

      {/* Navigation */}
      <nav style={navStyle}>
        {user ? (
          <div style={{ position: "relative" }}>
            <button
              onClick={() => setShowDropdown(!showDropdown)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                background: "rgba(255, 255, 255, 0.1)",
                border: "1px solid rgba(255, 255, 255, 0.2)",
                borderRadius: "25px",
                padding: "0.5rem 1rem",
                color: "#e3f2fd",
                cursor: "pointer",
                transition: "all 0.3s ease",
              }}
              onMouseEnter={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.2)";
              }}
              onMouseLeave={(e) => {
                if (!showDropdown) {
                  e.target.style.background = "rgba(255, 255, 255, 0.1)";
                }
              }}
            >
              <img
                src={user?.profileImage || `https://ui-avatars.com/api/?name=${user?.name || 'User'}&background=0f5c6e&color=fff`}
                alt="Profile"
                style={{
                  width: "32px",
                  height: "32px",
                  borderRadius: "50%",
                }}
              />
              <span style={{ fontWeight: 500 }}>
                {user?.name || "User"}
              </span>
              <span style={{ transform: showDropdown ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.3s ease" }}>
                ▼
              </span>
            </button>

            {showDropdown && (
              <div 
                style={dropdownStyle}
                onMouseLeave={() => setShowDropdown(false)}
              >
                <Link
                  to="/user/profile"
                  style={dropdownItemStyle}
                  onMouseEnter={(e) => e.target.style.background = "rgba(255, 255, 255, 0.1)"}
                  onMouseLeave={(e) => e.target.style.background = "transparent"}
                >
                  👤 Profile
                </Link>
                <Link
                  to="/user/security"
                  style={dropdownItemStyle}
                  onMouseEnter={(e) => e.target.style.background = "rgba(255, 255, 255, 0.1)"}
                  onMouseLeave={(e) => e.target.style.background = "transparent"}
                >
                  🔒 Security
                </Link>
                <div style={{ 
                  height: "1px", 
                  background: "rgba(255, 255, 255, 0.1)", 
                  margin: "0.5rem 0" 
                }} />
                <button
                  onClick={handleLogout}
                  style={{
                    ...dropdownItemStyle,
                    color: "#ff6b6b",
                  }}
                  onMouseEnter={(e) => e.target.style.background = "rgba(255, 107, 107, 0.1)"}
                  onMouseLeave={(e) => e.target.style.background = "transparent"}
                >
                  🚪 Logout
                </button>
              </div>
            )}
          </div>
        ) : (
          <Link 
            to="/auth/user/login" 
            style={{
              padding: "0.75rem 1.5rem",
              borderRadius: "6px",
              background: "linear-gradient(45deg, #ff6d00, #ff9100)",
              color: "white",
              fontWeight: 600,
              textDecoration: "none",
              transition: "all 0.3s ease",
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = "translateY(-2px)";
              e.target.style.boxShadow = "0 4px 15px rgba(255, 109, 0, 0.4)";
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = "translateY(0)";
              e.target.style.boxShadow = "none";
            }}
          >
            🚀 Launch Mission
          </Link>
        )}
      </nav>

      {/* Close dropdown when clicking outside */}
      {showDropdown && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 1000,
          }}
          onClick={() => setShowDropdown(false)}
        />
      )}
    </header>
  );
};

export default Header;