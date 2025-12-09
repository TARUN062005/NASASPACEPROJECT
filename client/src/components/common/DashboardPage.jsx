// client/src/components/common/DashboardPage.jsx

import React, { useState, useContext, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext } from "../../main.jsx";

const DashboardPage = ({ user }) => {
    const navigate = useNavigate();
    const { API } = useContext(AuthContext);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [recentPredictions, setRecentPredictions] = useState([]);

    const missionCards = [
        {
            id: "toi",
            title: "TESS Objects of Interest",
            description: "Analyze TESS Mission data for exoplanet detection",
            icon: "🪐",
            color: "from-blue-500 to-cyan-500",
            route: "/user/dashboard/toi",
            features: ["Single Prediction", "Bulk Analysis", "Real-time Charts"]
        },
        {
            id: "koi",
            title: "Kepler Objects of Interest",
            description: "Analyze Kepler Mission data for exoplanet detection",
            icon: "🌟",
            color: "from-purple-500 to-pink-500",
            route: "/user/dashboard/koi",
            features: ["Advanced Classification", "Probability Charts", "Export Data"]
        },
        {
            id: "k2",
            title: "K2 Mission Data",
            description: "Analyze K2 Mission data for exoplanet detection",
            icon: "🚀",
            color: "from-orange-500 to-red-500",
            route: "/user/dashboard/k2",
            features: ["Multi-field Analysis", "Visual Reports", "Batch Processing"]
        }
    ];

    const comparisonCard = {
        id: "compare",
        title: "Compare Datasets",
        description: "Verify model predictions against real-world data.",
        icon: "⚖️",
        color: "from-teal-500 to-green-500",
        route: "/user/compare",
        features: ["CSV File Upload", "Match Percentage", "Detailed Report"]
    };

    useEffect(() => {
        fetchDashboardData();
    }, []);

    const fetchDashboardData = async () => {
        try {
            setLoading(true);
            const [statsResponse, recentResponse] = await Promise.all([
                API.get("/api/ml/dashboard"),
                API.get("/api/ml/entries/toi?limit=3")
            ]);
            
            setStats(statsResponse.data.data);
            setRecentPredictions(recentResponse.data.data.entries || []);
        } catch (error) {
            console.error("Failed to fetch dashboard data:", error);
            setStats({
                counts: { toi: 0, koi: 0, k2: 0, customModels: 0, total: 0 },
                summary: { totalPredictions: 0, averageConfidence: 0 }
            });
        } finally {
            setLoading(false);
        }
    };

    const getServiceStatus = (service) => {
        return stats?.services?.[service]?.status === 'healthy' ? '🟢' : '🔴';
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white p-6">
                <div className="max-w-7xl mx-auto">
                    <div className="flex items-center justify-center h-96">
                        <div className="text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                            <p className="text-gray-400">Loading mission control...</p>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                        Mission Control Center
                    </h1>
                    <p className="text-gray-400 mt-2">
                        Welcome back, {user?.name || "Astronomer"}! Ready to explore the cosmos?
                    </p>
                </div>

                {/* Stats Overview */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-gray-400 text-sm">Total Predictions</p>
                                <p className="text-2xl font-bold text-white">
                                    {stats?.counts?.total || 0}
                                </p>
                            </div>
                            <div className="text-3xl">📊</div>
                        </div>
                    </div>

                    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-gray-400 text-sm">Avg Confidence</p>
                                <p className="text-2xl font-bold text-green-400">
                                    {((stats?.summary?.averageConfidence || 0) * 100).toFixed(1)}%
                                </p>
                            </div>
                            <div className="text-3xl">🎯</div>
                        </div>
                    </div>

                    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-gray-400 text-sm">Active Models</p>
                                <p className="text-2xl font-bold text-yellow-400">
                                    {3 + (stats?.counts?.customModels || 0)}
                                </p>
                            </div>
                            <div className="text-3xl">🤖</div>
                        </div>
                    </div>

                    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-gray-400 text-sm">Services Status</p>
                                <p className="text-lg font-bold text-white flex items-center gap-2">
                                    <span>{getServiceStatus('TOI')}</span>
                                    <span>{getServiceStatus('KOI')}</span>
                                    <span>{getServiceStatus('K2')}</span>
                                </p>
                            </div>
                            <div className="text-3xl">⚡</div>
                        </div>
                    </div>
                </div>

                {/* Mission Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    {[...missionCards, comparisonCard].map((mission) => (
                        <div
                            key={mission.id}
                            className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-gray-500 transition-all duration-300 hover:transform hover:scale-105 cursor-pointer group"
                            onClick={() => navigate(mission.route)}
                        >
                            <div className="flex items-start justify-between mb-4">
                                <div className={`text-4xl group-hover:scale-110 transition-transform`}>
                                    {mission.icon}
                                </div>
                                <div className="text-2xl opacity-0 group-hover:opacity-100 transition-opacity">
                                    →
                                </div>
                            </div>
                            
                            <h3 className="text-xl font-bold text-white mb-2">
                                {mission.title}
                            </h3>
                            
                            <p className="text-gray-400 text-sm mb-4">
                                {mission.description}
                            </p>
                            
                            <div className="space-y-1">
                                {mission.features.map((feature, index) => (
                                    <div key={index} className="flex items-center text-xs text-gray-400">
                                        <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                                        {feature}
                                    </div>
                                ))}
                            </div>

                            {mission.id !== "compare" && (
                                <div className={`mt-4 text-xs px-2 py-1 rounded-full bg-gradient-to-r ${mission.color} text-white inline-block`}>
                                    {stats?.counts?.[mission.id] || 0} predictions
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                {/* Recent Activity & Quick Actions */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Recent Predictions */}
                    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
                        <h3 className="text-xl font-bold text-white mb-4">Recent Activity</h3>
                        {recentPredictions.length > 0 ? (
                            <div className="space-y-4">
                                {recentPredictions.map((prediction, index) => (
                                    <div key={prediction.id} className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                                        <div>
                                            <p className="text-white font-medium">TOI Prediction</p>
                                            <p className="text-gray-400 text-sm">
                                                {new Date(prediction.createdAt).toLocaleDateString()}
                                            </p>
                                        </div>
                                        <div className="text-right">
                                            <p className={`font-semibold ${
                                                prediction.data?.output?.predicted_class === 'CONFIRMED' ? 'text-green-400' :
                                                    prediction.data?.output?.predicted_class === 'CANDIDATE' ? 'text-yellow-400' :
                                                        'text-red-400'
                                            }`}>
                                                {prediction.data?.output?.predicted_class || 'Unknown'}
                                            </p>
                                            <p className="text-gray-400 text-sm">
                                                {((prediction.data?.output?.confidence || 0) * 100).toFixed(1)}% confidence
                                            </p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-8 text-gray-400">
                                <div className="text-4xl mb-2">🔭</div>
                                <p>No predictions yet</p>
                                <p className="text-sm">Start exploring by selecting a mission</p>
                            </div>
                        )}
                    </div>

                    {/* Quick Actions */}
                    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
                        <h3 className="text-xl font-bold text-white mb-4">Quick Actions</h3>
                        <div className="space-y-3">
                            <button
                                onClick={() => navigate("/user/dashboard/toi")}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
                            >
                                <span>🪐</span>
                                Quick TESS Analysis
                            </button>
                            
                            <button
                                onClick={() => navigate("/user/profile")}
                                className="w-full bg-gray-700 hover:bg-gray-600 text-white py-3 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
                            >
                                <span>👤</span>
                                Update Profile
                            </button>
                            
                            <button
                                onClick={fetchDashboardData}
                                className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
                            >
                                <span>🔄</span>
                                Refresh Data
                            </button>
                        </div>

                        {/* System Status */}
                        <div className="mt-6 pt-4 border-t border-gray-700">
                            <h4 className="font-semibold text-white mb-3">System Status</h4>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-gray-400">ML Services:</span>
                                    <span className="text-green-400">Operational</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Database:</span>
                                    <span className="text-green-400">Connected</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Last Update:</span>
                                    <span className="text-gray-400">{new Date().toLocaleTimeString()}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DashboardPage;