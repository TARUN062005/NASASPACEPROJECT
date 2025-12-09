import React, { useState, useContext, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext } from "../../main.jsx";

// Single Prediction Tab
const PredictionTab = ({ config, API, modelInfo }) => {
    const [predictionData, setPredictionData] = useState({});
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [predictionTime, setPredictionTime] = useState(null);

    const handlePredict = async () => {
        if (Object.keys(predictionData).length === 0) {
            setError("Please enter prediction data");
            return;
        }

        setLoading(true);
        setError(null);
        setPredictionTime(null);
        const startTime = Date.now();

        try {
            const response = await API.post("/api/ml/predict/toi", {
                data: predictionData
            });

            const endTime = Date.now();
            setPredictionTime(((endTime - startTime) / 1000).toFixed(2));

            if (response.data.success) {
                setResult(response.data);
            } else {
                throw new Error(response.data.message || "Prediction failed");
            }
        } catch (error) {
            console.error("TOI Prediction failed:", error);
            setError(error.response?.data?.message || error.message);
        } finally {
            setLoading(false);
        }
    };

    const handleInputChange = (featureName, value) => {
        setPredictionData(prev => ({
            ...prev,
            [featureName]: value ? parseFloat(value) : 0
        }));
    };

    const loadSampleData = () => {
        setPredictionData(config.sampleData);
        setResult(null);
        setError(null);
    };

    const clearForm = () => {
        setPredictionData({});
        setResult(null);
        setError(null);
        setPredictionTime(null);
    };
    
    // New refresh function
    const generateRandomData = () => {
        const newData = {};
        config.features.forEach(feature => {
            let randomValue;
            // Generate random values within a plausible range for each feature
            switch (feature.name) {
                case "pl_orbper":
                    randomValue = (Math.random() * (100 - 0.5) + 0.5).toFixed(3);
                    break;
                case "pl_trandurh":
                    randomValue = (Math.random() * (10 - 0.1) + 0.1).toFixed(2);
                    break;
                case "pl_trandep":
                    randomValue = (Math.random() * (5000 - 10) + 10).toFixed(2);
                    break;
                case "pl_rade":
                    randomValue = (Math.random() * (10 - 0.5) + 0.5).toFixed(2);
                    break;
                case "pl_insol":
                    randomValue = (Math.random() * (50000 - 100) + 100).toFixed(2);
                    break;
                case "pl_eqt":
                    randomValue = (Math.random() * (5000 - 200) + 200).toFixed(2);
                    break;
                case "st_tmag":
                    randomValue = (Math.random() * (15 - 5) + 5).toFixed(2);
                    break;
                case "st_dist":
                    randomValue = (Math.random() * (1000 - 10) + 10).toFixed(2);
                    break;
                case "st_teff":
                    randomValue = (Math.random() * (15000 - 2000) + 2000).toFixed(0);
                    break;
                case "st_logg":
                    randomValue = (Math.random() * (5 - 3) + 3).toFixed(2);
                    break;
                case "st_rad":
                    randomValue = (Math.random() * (5 - 0.5) + 0.5).toFixed(2);
                    break;
                default:
                    randomValue = (Math.random() * 100).toFixed(2);
            }
            newData[feature.name] = parseFloat(randomValue);
        });
        setPredictionData(newData);
        setResult(null);
        setError(null);
    };

    const getClassColor = (className) => {
        const colors = {
            'FP': 'text-red-400',
            'PC': 'text-yellow-400',
            'KP': 'text-green-400',
            'CP': 'text-green-400',
            'APC': 'text-orange-400',
            'FA': 'text-red-400'
        };
        return colors[className] || 'text-white';
    };

    const getClassDescription = (className) => {
        const descriptions = {
            'FP': 'False Positive',
            'PC': 'Planetary Candidate',
            'KP': 'Known Planet',
            'CP': 'Confirmed Planet',
            'APC': 'Ambiguous Planetary Candidate',
            'FA': 'False Alarm'
        };
        return descriptions[className] || className;
    };

    return (
        <div className="space-y-6">
            {error && (
                <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 animate-pulse">
                    <p className="text-red-300">{error}</p>
                </div>
            )}
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Input Form */}
                <div className="space-y-4">
                    <div className="flex justify-between items-center">
                        <h3 className="text-xl font-bold text-white">Enter TESS Observation Data</h3>
                        <div className="flex space-x-2">
                            <button
                                onClick={loadSampleData}
                                className="px-3 py-1 bg-green-600 text-white rounded-lg text-sm hover:bg-green-700 transition-colors"
                            >
                                Load Sample
                            </button>
                            {/* New refresh button */}
                            <button
                                onClick={generateRandomData}
                                className="px-3 py-1 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition-colors"
                            >
                                Refresh Values
                            </button>
                            <button
                                onClick={clearForm}
                                className="px-3 py-1 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700 transition-colors"
                            >
                                Clear
                            </button>
                        </div>
                    </div>
                    
                    <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
                        {config.features.map((feature) => (
                            <div key={feature.name} className="bg-gray-900/50 p-3 rounded-lg">
                                <label className="block text-sm font-medium text-gray-300 mb-1">
                                    {feature.label}
                                </label>
                                <input
                                    type={feature.type}
                                    step="any"
                                    placeholder={feature.placeholder}
                                    value={predictionData[feature.name] || ''}
                                    className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                                    onChange={(e) => handleInputChange(feature.name, e.target.value)}
                                />
                                <p className="text-xs text-gray-400 mt-1">
                                    Example: {feature.placeholder}
                                </p>
                            </div>
                        ))}
                    </div>

                    <button
                        onClick={handlePredict}
                        disabled={loading}
                        className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50 flex items-center justify-center gap-2 relative overflow-hidden"
                    >
                        {loading && (
                            <div className="absolute inset-0 bg-blue-600 animate-pulse"></div>
                        )}
                        {loading ? (
                            <>
                                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white z-10"></div>
                                <span className="z-10">Analyzing TESS Data...</span>
                            </>
                        ) : (
                            <>
                                <span className="z-10">🔍</span>
                                <span className="z-10">Detect Exoplanets</span>
                            </>
                        )}
                    </button>

                    {loading && (
                        <div className="bg-gray-800 rounded-lg p-4">
                            <div className="flex justify-between text-sm text-gray-300 mb-2">
                                <span>Processing TESS data...</span>
                                <span className="text-blue-400">AI Analyzing</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2">
                                <div className="bg-blue-500 h-2 rounded-full animate-pulse"></div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Results */}
                <div>
                    <h3 className="text-xl font-bold text-white mb-4">Prediction Results</h3>
                    {loading ? (
                        <div className="space-y-4 animate-pulse">
                            <div className="bg-gray-900 rounded-lg p-6 border border-blue-500">
                                <div className="h-4 bg-gray-700 rounded w-3/4 mb-2"></div>
                                <div className="h-4 bg-gray-700 rounded w-1/2"></div>
                            </div>
                            <div className="bg-gray-900 rounded-lg p-4">
                                <div className="h-24 bg-gray-700 rounded"></div>
                            </div>
                        </div>
                    ) : result ? (
                        <div className="space-y-4 animate-fade-in">
                            <div className="bg-gray-900 rounded-lg p-6 border border-blue-500">
                                <h4 className="text-lg font-bold text-white mb-2">Detection Result</h4>
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-gray-300">Predicted Class:</span>
                                        <span className={`font-bold ${getClassColor(result.data?.prediction?.predicted_class)}`}>
                                            {getClassDescription(result.data?.prediction?.predicted_class)}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-300">Confidence:</span>
                                        <span className="font-bold text-blue-400">
                                            {((result.data?.prediction?.confidence || 0) * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                    {predictionTime && (
                                        <div className="flex justify-between">
                                            <span className="text-gray-300">Processing Time:</span>
                                            <span className="font-bold text-green-400">
                                                {predictionTime}s
                                            </span>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Charts Display */}
                            {result.data?.prediction?.charts && (
                                <div className="bg-gray-900 rounded-lg p-4">
                                    <h5 className="font-semibold text-white mb-3">Visual Analysis</h5>
                                    <div className="grid grid-cols-1 gap-4">
                                        {Object.entries(result.data.prediction.charts).map(([chartName, chartData]) => (
                                            <div key={chartName} className="text-center">
                                                <img
                                                    src={`data:image/png;base64,${chartData}`}
                                                    alt={chartName}
                                                    className="max-w-full h-auto rounded-lg border border-gray-600"
                                                />
                                                <p className="text-gray-400 text-sm mt-2 capitalize">
                                                    {chartName.replace('_', ' ')}
                                                </p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {result.data?.prediction?.explanation && (
                                <div className="bg-gray-900 rounded-lg p-4">
                                    <h5 className="font-semibold text-white mb-2">Scientific Explanation</h5>
                                    <p className="text-gray-300 text-sm">
                                        {result.data.prediction.explanation}
                                    </p>
                                </div>
                            )}

                            {/* Probabilities */}
                            {result.data?.prediction?.probabilities && (
                                <div className="bg-gray-900 rounded-lg p-4">
                                    <h5 className="font-semibold text-white mb-3">Class Probabilities</h5>
                                    <div className="space-y-2">
                                        {Object.entries(result.data.prediction.probabilities)
                                            .sort(([, a], [, b]) => b - a)
                                            .map(([className, probability]) => (
                                                <div key={className} className="flex justify-between items-center">
                                                    <span className={`text-sm ${getClassColor(className)}`}>
                                                        {getClassDescription(className)}
                                                    </span>
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-20 bg-gray-700 rounded-full h-2">
                                                            <div
                                                                className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                                                                style={{ width: `${probability * 100}%` }}
                                                            ></div>
                                                        </div>
                                                        <span className="text-white font-medium w-12 text-right">
                                                            {(probability * 100).toFixed(1)}%
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-center py-12 text-gray-400">
                            <div className="text-6xl mb-4">🪐</div>
                            <p>Enter TESS data and click detect to see results</p>
                            <p className="text-sm mt-2">Real-time TESS Objects of Interest analysis</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

// Enhanced Bulk Analysis Tab
const BulkTab = ({ config, API, onResults, onExport }) => {
    const [file, setFile] = useState(null);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [progress, setProgress] = useState(0);
    const [currentRow, setCurrentRow] = useState(0);
    const [totalRows, setTotalRows] = useState(0);
    const [exporting, setExporting] = useState({ csv: false, excel: false });

    const handleFileUpload = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            if (!selectedFile.name.toLowerCase().endsWith('.csv')) {
                setError("Please upload a CSV file");
                return;
            }

            if (selectedFile.size > 100 * 1024 * 1024) {
                setError("File size too large. Maximum 100MB allowed.");
                return;
            }

            setFile(selectedFile);
            setError(null);
            setResults(null);
            setProgress(0);
            setCurrentRow(0);
            setTotalRows(0);
        }
    };

    const processBulkFile = async () => {
        if (!file) {
            setError("Please select a CSV file first");
            return;
        }

        setLoading(true);
        setError(null);
        setProgress(0);
        setCurrentRow(0);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const text = await file.text();
            const estimatedRows = text.split('\n').filter(row => row.trim() && !row.startsWith('#')).length;
            setTotalRows(estimatedRows);
            
            const progressInterval = setInterval(() => {
                setProgress(prev => Math.min(prev + 1, 95));
            }, 100);

            const response = await API.post("/api/ml/process-file/toi", formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                },
                timeout: 300000,
                onUploadProgress: (progressEvent) => {
                    if (progressEvent.total) {
                        const uploadProgress = (progressEvent.loaded / progressEvent.total) * 100;
                        setProgress(Math.min(uploadProgress, 95));
                    }
                }
            });

            clearInterval(progressInterval);
            setProgress(100);

            if (response.data.success) {
                setResults(response.data.data);
                onResults(response.data.data);

                if (response.data.data.processed > 0) {
                    setTimeout(() => {
                        onExport('csv');
                    }, 1500);
                }
            } else {
                throw new Error(response.data.message || "Processing failed");
            }
        } catch (error) {
            console.error("Bulk processing failed:", error);
            const errorMessage = error.response?.data?.message || error.message || "Processing service unavailable";
            setError(`Processing failed: ${errorMessage}`);
        } finally {
            setLoading(false);
            setProgress(0);
        }
    };

    const downloadTemplate = () => {
        const headers = config.features.map(f => f.name).join(',');
        const sampleRow = config.features.map(f => config.sampleData[f.name]).join(',');
        const csvContent = `${headers}\n${sampleRow}`;

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'toi_bulk_template.csv');
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
    };

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* File Upload */}
                <div className="space-y-4">
                    <h3 className="text-xl font-bold text-white">Bulk TESS Data Analysis</h3>
                    <div className={`bg-gray-900/50 rounded-lg p-6 border-2 border-dashed transition-all ${
                        file ? 'border-green-500 bg-green-900/20' : 'border-gray-600 hover:border-blue-500'
                    }`}>
                        <div className="text-center">
                            <div className="text-4xl mb-4">📁</div>
                            <p className="text-gray-300 mb-2">Upload CSV File</p>
                            <p className="text-gray-400 text-sm mb-4">
                                Supports large files (up to 100MB, 100,000+ rows)
                            </p>
                            <input
                                type="file"
                                accept=".csv"
                                onChange={handleFileUpload}
                                className="hidden"
                                id="bulk-file-input"
                                disabled={loading}
                            />
                            <label
                                htmlFor="bulk-file-input"
                                className={`inline-block px-6 py-3 rounded-lg cursor-pointer transition-all ${
                                    loading
                                        ? 'bg-gray-600 cursor-not-allowed opacity-50'
                                        : 'bg-blue-600 hover:bg-blue-700'
                                } text-white font-semibold`}
                            >
                                {loading ? 'Processing...' : 'Choose File'}
                            </label>
                            {file && (
                                <div className="mt-3 p-2 bg-gray-800 rounded">
                                    <p className="text-green-400 text-sm">
                                        ✅ Selected: {file.name}
                                    </p>
                                    <p className="text-gray-400 text-xs">
                                        Size: {(file.size / (1024 * 1024)).toFixed(2)} MB
                                        {totalRows > 0 && ` • Estimated: ${totalRows} rows`}
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                    {loading && (
                        <div className="bg-gray-800 rounded-lg p-4 space-y-3">
                            <div className="flex justify-between text-sm text-gray-300">
                                <span>Processing...</span>
                                <span>{progress}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-3">
                                <div
                                    className="bg-blue-500 h-3 rounded-full transition-all duration-300"
                                    style={{ width: `${progress}%` }}
                                ></div>
                            </div>
                            {currentRow > 0 && totalRows > 0 && (
                                <div className="text-xs text-gray-400">
                                    Processing row {currentRow} of {totalRows}
                                    ({Math.round((currentRow / totalRows) * 100)}%)
                                </div>
                            )}
                            <div className="text-xs text-blue-400 animate-pulse">
                                ⏳ Processing large file - this may take several minutes...
                            </div>
                        </div>
                    )}
                    <div className="flex space-x-3">
                        <button
                            onClick={processBulkFile}
                            disabled={!file || loading}
                            className="flex-1 bg-gradient-to-r from-green-600 to-green-700 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 relative overflow-hidden"
                        >
                            {loading && (
                                <div className="absolute inset-0 bg-green-600 animate-pulse"></div>
                            )}
                            {loading ? (
                                <>
                                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white z-10"></div>
                                    <span className="z-10">Processing...</span>
                                </>
                            ) : (
                                <>
                                    <span className="z-10">🚀</span>
                                    <span className="z-10">Process File</span>
                                </>
                            )}
                        </button>
                        <button
                            onClick={downloadTemplate}
                            disabled={loading}
                            className="px-4 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            📋 Template
                        </button>
                    </div>
                    {error && (
                        <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 animate-pulse">
                            <div className="flex items-center">
                                <span className="text-red-400 text-lg mr-2">⚠️</span>
                                <p className="text-red-300">{error}</p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Results */}
                <div>
                    <h3 className="text-xl font-bold text-white mb-4">Bulk Results</h3>
                    {loading ? (
                        <div className="space-y-4 animate-pulse">
                            <div className="bg-gray-900 rounded-lg p-6 border border-gray-700">
                                <div className="h-4 bg-gray-700 rounded w-3/4 mb-2"></div>
                                <div className="h-4 bg-gray-700 rounded w-1/2"></div>
                            </div>
                            <div className="bg-gray-900 rounded-lg p-4">
                                <div className="h-24 bg-gray-700 rounded"></div>
                            </div>
                        </div>
                    ) : results ? (
                        <div className="space-y-4 animate-fade-in">
                            <div className={`bg-gray-900 rounded-lg p-6 border ${
                                results.errors > 0 ? 'border-yellow-500' : 'border-green-500'
                            }`}>
                                <h4 className="text-lg font-bold text-white mb-2">
                                    {results.errors > 0 ? '⚠️ Processing Complete with Errors' : '✅ Processing Complete'}
                                </h4>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-300">File:</span>
                                        <span className="text-white font-mono">{file.name}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-300">Total Records:</span>
                                        <span className="text-white">{results.total}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-300">Successfully Processed:</span>
                                        <span className="text-green-400 font-semibold">{results.processed}</span>
                                    </div>
                                    {results.errors > 0 && (
                                        <div className="flex justify-between">
                                            <span className="text-gray-300">Errors:</span>
                                            <span className="text-yellow-400 font-semibold">{results.errors}</span>
                                        </div>
                                    )}
                                    <div className="flex justify-between">
                                        <span className="text-gray-300">Stored in Database:</span>
                                        <span className="text-blue-400 font-semibold">{results.stored} predictions</span>
                                    </div>
                                </div>
                            </div>
                            {results.errors > 0 && (
                                <div className="bg-yellow-900/20 rounded-lg p-4 border border-yellow-700">
                                    <h5 className="font-semibold text-yellow-400 mb-2">Processing Errors</h5>
                                    <div className="max-h-32 overflow-y-auto">
                                        {results.errorsList?.slice(0, 5).map((error, index) => (
                                            <div key={index} className="text-yellow-300 text-sm mb-1">
                                                Row {error.row}: {error.error}
                                            </div>
                                        ))}
                                        {results.errors > 5 && (
                                            <div className="text-yellow-400 text-xs">
                                                ... and {results.errors - 5} more errors
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                            <div className="bg-gray-900 rounded-lg p-4">
                                <h5 className="font-semibold text-white mb-3">Download Results</h5>
                                <p className="text-gray-300 text-sm mb-3">
                                    {results.processed > 0
                                        ? `Your ${results.processed} predictions have been processed and stored. Download the results for analysis.`
                                        : 'No successful predictions to download.'
                                    }
                                </p>
                                {results.processed > 0 && (
                                    <div className="flex space-x-3">
                                        <button
                                            onClick={() => onExport('csv')}
                                            disabled={exporting.csv}
                                            className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2 relative overflow-hidden"
                                        >
                                            {exporting.csv && (
                                                <div className="absolute inset-0 bg-blue-700 animate-pulse"></div>
                                            )}
                                            {exporting.csv ? (
                                                <>
                                                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white z-10"></div>
                                                    <span className="z-10">Downloading...</span>
                                                </>
                                            ) : (
                                                <>
                                                    <span className="z-10">📥</span>
                                                    <span className="z-10">Download CSV</span>
                                                </>
                                            )}
                                        </button>
                                        <button
                                            onClick={() => onExport('excel')}
                                            disabled={exporting.excel}
                                            className="flex-1 bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2 relative overflow-hidden"
                                        >
                                            {exporting.excel && (
                                                <div className="absolute inset-0 bg-green-700 animate-pulse"></div>
                                            )}
                                            {exporting.excel ? (
                                                <>
                                                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white z-10"></div>
                                                    <span className="z-10">Downloading...</span>
                                                </>
                                            ) : (
                                                <>
                                                    <span className="z-10">📊</span>
                                                    <span className="z-10">Download Excel</span>
                                                </>
                                            )}
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="text-center py-12 text-gray-400">
                            <div className="text-6xl mb-4">📈</div>
                            <p>Upload a CSV file to process multiple TESS observations</p>
                            <p className="text-sm mt-2">Supports large datasets with 100,000+ rows</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

// Enhanced History Tab with Download Animations
const HistoryTab = ({ predictions, loading, onRefresh, onExport }) => {
    const [exporting, setExporting] = useState({ csv: false, excel: false });

    const handleExportWithAnimation = async (format) => {
        setExporting(prev => ({ ...prev, [format]: true }));
        try {
            await onExport(format);
        } finally {
            setTimeout(() => {
                setExporting(prev => ({ ...prev, [format]: false }));
            }, 1000);
        }
    };

    return (
        <div>
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold text-white">TOI Prediction History</h3>
                <div className="flex space-x-3">
                    <button
                        onClick={() => handleExportWithAnimation('csv')}
                        disabled={exporting.csv}
                        className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center gap-2 relative overflow-hidden"
                    >
                        {exporting.csv && (
                            <div className="absolute inset-0 bg-blue-700 animate-pulse"></div>
                        )}
                        {exporting.csv ? (
                            <>
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white z-10"></div>
                                <span className="z-10">Downloading...</span>
                            </>
                        ) : (
                            <>
                                <span className="z-10">📥</span>
                                <span className="z-10">CSV</span>
                            </>
                        )}
                    </button>
                    <button
                        onClick={() => handleExportWithAnimation('excel')}
                        disabled={exporting.excel}
                        className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 flex items-center gap-2 relative overflow-hidden"
                    >
                        {exporting.excel && (
                            <div className="absolute inset-0 bg-green-700 animate-pulse"></div>
                        )}
                        {exporting.excel ? (
                            <>
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white z-10"></div>
                                <span className="z-10">Downloading...</span>
                            </>
                        ) : (
                            <>
                                <span className="z-10">📊</span>
                                <span className="z-10">Excel</span>
                            </>
                        )}
                    </button>
                    <button
                        onClick={onRefresh}
                        className="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors flex items-center gap-2"
                    >
                        <span>🔄</span>
                        <span>Refresh</span>
                    </button>
                </div>
            </div>

            {loading ? (
                <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto"></div>
                    <p className="text-gray-400 mt-2">Loading prediction history...</p>
                </div>
            ) : predictions.length > 0 ? (
                <div className="space-y-4">
                    {predictions.map((prediction, index) => (
                        <div key={prediction.id} className="bg-gray-900 rounded-lg p-4 border border-gray-700 hover:border-blue-500 transition-colors">
                            <div className="flex justify-between items-start">
                                <div className="flex-1">
                                    <p className="font-semibold text-white">TOI Prediction #{predictions.length - index}</p>
                                    <p className="text-gray-400 text-sm">
                                        {new Date(prediction.createdAt).toLocaleString()}
                                    </p>
                                    
                                    {prediction.data?.input && (
                                        <div className="mt-2">
                                            <p className="text-gray-400 text-xs">Features:</p>
                                            <div className="flex flex-wrap gap-1 mt-1">
                                                {Object.entries(prediction.data.input).slice(0, 3).map(([key, value]) => (
                                                    <span key={key} className="text-xs bg-gray-800 px-2 py-1 rounded">
                                                        {key}: {typeof value === 'number' ? value.toFixed(2) : value}
                                                    </span>
                                                ))}
                                                {Object.keys(prediction.data.input).length > 3 && (
                                                    <span className="text-xs bg-gray-800 px-2 py-1 rounded">
                                                        +{Object.keys(prediction.data.input).length - 3} more
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                                
                                <div className="text-right ml-4">
                                    <p className={`font-semibold ${
                                        prediction.data?.output?.predicted_class === 'CP' || prediction.data?.output?.predicted_class === 'KP' ? 'text-green-400' :
                                        prediction.data?.output?.predicted_class === 'PC' ? 'text-yellow-400' :
                                        'text-red-400'
                                    }`}>
                                        {prediction.data?.output?.predicted_class || "Unknown"}
                                    </p>
                                    <p className="text-blue-400 text-sm">
                                        {((prediction.data?.output?.confidence || 0) * 100).toFixed(1)}% confidence
                                    </p>
                                    {prediction.data?.metadata?.has_charts && (
                                        <p className="text-green-400 text-xs mt-1">📊 Charts Available</p>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="text-center py-12 text-gray-400">
                    <div className="text-6xl mb-4">📊</div>
                    <p>No TOI prediction history yet</p>
                    <p className="text-sm mt-2">Make your first prediction to see it here</p>
                </div>
            )}
        </div>
    );
};

// Info Tab Component
const InfoTab = ({ config, modelInfo }) => (
    <div className="space-y-6">
        <h3 className="text-xl font-bold text-white">TOI Model Information</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-900 rounded-lg p-4">
                <h4 className="font-semibold text-white mb-2">Model Details</h4>
                <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                        <span className="text-gray-400">Type:</span>
                        <span className="text-white">{modelInfo?.model_type || "Ensemble Classifier"}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-400">Trained:</span>
                        <span className="text-white">{modelInfo?.is_trained ? "Yes" : "No"}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-400">Target:</span>
                        <span className="text-white">{modelInfo?.target_column || "tfopwg_disp"}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-400">Classes:</span>
                        <span className="text-white">{modelInfo?.class_names?.join(", ") || "FP, PC, KP, CP, APC, FA"}</span>
                    </div>
                </div>
            </div>

            <div className="bg-gray-900 rounded-lg p-4">
                <h4 className="font-semibold text-white mb-2">TESS Features Used</h4>
                <div className="text-sm text-gray-300 max-h-32 overflow-y-auto">
                    {modelInfo?.selected_features?.map((feature, index) => (
                        <div key={index} className="mb-1 flex items-center">
                            <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                            {feature}
                        </div>
                    )) || config.features.slice(0, 6).map((feature, index) => (
                        <div key={index} className="mb-1 flex items-center">
                            <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                            {feature.name}
                        </div>
                    ))}
                </div>
            </div>
        </div>

        <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700">
            <h4 className="font-semibold text-blue-400 mb-2">About TESS Mission</h4>
            <p className="text-blue-300 text-sm">
                The TESS (Transiting Exoplanet Survey Satellite) mission searches for exoplanets using the transit method, 
                monitoring the brightness of stars for periodic dips caused by planets passing in front of them.
                TESS covers an area 400 times larger than the Kepler mission and has discovered thousands of exoplanet candidates.
            </p>
        </div>

        <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="font-semibold text-white mb-3">Class Descriptions</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                <div className="flex items-center">
                    <span className="w-3 h-3 bg-red-500 rounded-full mr-2"></span>
                    <span className="text-gray-300">FP/FA:</span>
                    <span className="text-gray-400 ml-2">False Positive/Alarm</span>
                </div>
                <div className="flex items-center">
                    <span className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></span>
                    <span className="text-gray-300">PC/APC:</span>
                    <span className="text-gray-400 ml-2">Planetary Candidate</span>
                </div>
                <div className="flex items-center">
                    <span className="w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                    <span className="text-gray-300">KP/CP:</span>
                    <span className="text-gray-400 ml-2">Confirmed Planet</span>
                </div>
            </div>
        </div>
    </div>
);

const TOIDashboard = () => {
    const navigate = useNavigate();
    const { API } = useContext(AuthContext);
    const [activeTab, setActiveTab] = useState("predict");
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(false);
    const [modelInfo, setModelInfo] = useState(null);
    const [bulkResults, setBulkResults] = useState(null);

    const config = {
        name: "TESS Objects of Interest",
        description: "TESS Mission data analysis with advanced visualization",
        color: "blue",
        icon: "🪐",
        features: [
            { name: "pl_orbper", label: "Orbital Period (days)", type: "number", placeholder: "2.1713484" },
            { name: "pl_trandurh", label: "Transit Duration (hours)", type: "number", placeholder: "2.0172196" },
            { name: "pl_trandep", label: "Transit Depth (ppm)", type: "number", placeholder: "656.8860989" },
            { name: "pl_rade", label: "Planet Radius (Earth radii)", type: "number", placeholder: "5.8181633" },
            { name: "pl_insol", label: "Insolation Flux (Earth flux)", type: "number", placeholder: "22601.94858" },
            { name: "pl_eqt", label: "Equilibrium Temperature (K)", type: "number", placeholder: "3127.204052" },
            { name: "st_tmag", label: "TESS Magnitude", type: "number", placeholder: "9.604" },
            { name: "st_dist", label: "Star Distance (pc)", type: "number", placeholder: "485.735" },
            { name: "st_teff", label: "Star Temperature (K)", type: "number", placeholder: "10249" },
            { name: "st_logg", label: "Star Surface Gravity (log g)", type: "number", placeholder: "4.19" },
            { name: "st_rad", label: "Star Radius (Solar radii)", type: "number", placeholder: "2.16986" }
        ],
        sampleData: {
            pl_orbper: 2.1713484,
            pl_trandurh: 2.0172196,
            pl_trandep: 656.8860989,
            pl_rade: 5.8181633,
            pl_insol: 22601.94858,
            pl_eqt: 3127.204052,
            st_tmag: 9.604,
            st_dist: 485.735,
            st_teff: 10249,
            st_logg: 4.19,
            st_rad: 2.16986
        }
    };

    const fetchModelInfo = async () => {
        try {
            const response = await API.get("/api/ml/model-info/toi");
            setModelInfo(response.data.data);
        } catch (error) {
            console.error("Failed to fetch TOI model info:", error);
            setModelInfo({
                model_type: "Ensemble Classifier",
                is_trained: true,
                class_names: ["FP", "PC", "KP", "CP", "APC", "FA"],
                selected_features: config.features.map(f => f.name),
                target_column: "tfopwg_disp"
            });
        }
    };

    const fetchPredictionHistory = async () => {
        try {
            setLoading(true);
            const response = await API.get("/api/ml/entries/toi?limit=20");
            setPredictions(response.data.data.entries || []);
        } catch (error) {
            console.error("Failed to fetch TOI prediction history:", error);
            setPredictions([]);
        } finally {
            setLoading(false);
        }
    };

    const handleExport = async (format = 'csv') => {
        try {
            const response = await API.get(`/api/ml/export/toi?format=${format}`, {
                responseType: 'blob',
                timeout: 300000
            });

            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            
            const extension = format === 'excel' ? 'xlsx' : 'csv';
            link.setAttribute('download', `toi_predictions_${Date.now()}.${extension}`);
            
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
            
        } catch (error) {
            console.error("Export failed:", error);
            const errorMessage = error.message.includes('timeout')
                ? "The download timed out. The file may be too large or the network is slow. Please try again."
                : (error.response?.data?.message || error.message || "An unexpected error occurred during export.");
            alert("Export failed: " + errorMessage);
        }
    };

    useEffect(() => {
        fetchModelInfo();
        if (activeTab === "history") {
            fetchPredictionHistory();
        }
    }, [activeTab]);

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <button
                            onClick={() => navigate("/user/dashboard")}
                            className="flex items-center space-x-2 text-gray-400 hover:text-white transition-colors mb-4"
                        >
                            <span>←</span>
                            <span>Back to Dashboard</span>
                        </button>
                        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                            {config.name}
                        </h1>
                        <p className="text-gray-400 mt-2">{config.description}</p>
                    </div>
                    <div className="text-6xl">
                        {config.icon}
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex space-x-1 bg-gray-800 rounded-lg p-1 mb-8">
                    {["predict", "bulk", "history", "info"].map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`flex-1 py-3 px-6 rounded-md font-semibold transition-all ${
                                activeTab === tab
                                    ? "bg-blue-600 text-white shadow-lg"
                                    : "text-gray-400 hover:text-white"
                            }`}
                        >
                            {tab === "predict" && "🔮 Single Prediction"}
                            {tab === "bulk" && "📁 Bulk Analysis"}
                            {tab === "history" && "📊 History"}
                            {tab === "info" && "ℹ️ Model Info"}
                        </button>
                    ))}
                </div>

                {/* Tab Content */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
                    {activeTab === "predict" && (
                        <PredictionTab config={config} API={API} modelInfo={modelInfo} />
                    )}
                    {activeTab === "bulk" && (
                        <BulkTab config={config} API={API} onResults={setBulkResults} onExport={handleExport} />
                    )}
                    {activeTab === "history" && (
                        <HistoryTab
                            predictions={predictions}
                            loading={loading}
                            onRefresh={fetchPredictionHistory}
                            onExport={handleExport}
                        />
                    )}
                    {activeTab === "info" && (
                        <InfoTab config={config} modelInfo={modelInfo} />
                    )}
                </div>
            </div>
        </div>
    );
};

export default TOIDashboard;