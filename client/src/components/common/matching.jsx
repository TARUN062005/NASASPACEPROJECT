import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

/**
 * A component for uploading, comparing, and downloading matched data from two CSV files.
 */
function Matching() {
    const navigate = useNavigate();
    const [file1, setFile1] = useState(null);
    const [file2, setFile2] = useState(null);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const resetState = () => {
        setResult(null);
        setError('');
    };

    const handleFile1Change = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && !selectedFile.name.toLowerCase().endsWith('.csv')) {
            setError('Please upload a valid CSV file.');
            setFile1(null);
            return;
        }
        setFile1(selectedFile);
        resetState();
    };

    const handleFile2Change = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && !selectedFile.name.toLowerCase().endsWith('.csv')) {
            setError('Please upload a valid CSV file.');
            setFile2(null);
            return;
        }
        setFile2(selectedFile);
        resetState();
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file1 || !file2) {
            setError('Please select both CSV files.');
            return;
        }

        setLoading(true);
        resetState();

        const formData = new FormData();
        formData.append('file1', file1);
        formData.append('file2', file2);

        try {
            const response = await axios.post('https://nasaspacebackend.onrender.com/api/compare', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setResult(response.data.summary);
        } catch (err) {
            const errorMessage = err.response?.data?.message || 'An unexpected error occurred.';
            console.error("Comparison Error:", err);
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    /**
     * Converts an array of objects to a CSV string and triggers a download.
     */
    const handleDownload = () => {
        if (!result || result.matchingRows.length === 0) {
            setError('No matching data to download.');
            return;
        }

        const dataToDownload = result.matchingRows.map(row => ({
            'ID': row.id,
            ...row.file1,
            'File2_Matched_Data': JSON.stringify(row.file2),
        }));

        const headers = Object.keys(dataToDownload[0]);
        const csvContent = [
            headers.join(','),
            ...dataToDownload.map(row =>
                headers.map(header => {
                    let cell = row[header] === null || row[header] === undefined ? '' : row[header];
                    cell = String(cell);
                    if (cell.includes('"') || cell.includes(',') || cell.includes('\n')) {
                        cell = `"${cell.replace(/"/g, '""')}"`;
                    }
                    return cell;
                }).join(',')
            )
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', `matched_data_${Date.now()}.csv`);
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white p-6">
            <div className="max-w-4xl mx-auto">
                <button
                    onClick={() => navigate("/user/dashboard")}
                    className="flex items-center space-x-2 text-gray-400 hover:text-white transition-colors mb-4"
                >
                    <span>←</span>
                    <span>Back to Dashboard</span>
                </button>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-teal-400 to-green-500 bg-clip-text text-transparent mb-2">
                    File Comparison
                </h1>
                <p className="text-gray-400 mb-8">
                    Upload two CSV files to compare their contents and find matching records based on common identifiers.
                </p>

                <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* File 1 Input */}
                            <div className="space-y-2">
                                <label htmlFor="file1" className="block text-sm font-medium text-gray-300">
                                    File 1 (Your Data)
                                </label>
                                <input 
                                    id="file1" 
                                    type="file" 
                                    accept=".csv" 
                                    onChange={handleFile1Change} 
                                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100"
                                />
                                {file1 && <p className="text-green-400 text-xs mt-1">✅ Selected: {file1.name}</p>}
                            </div>
                            
                            {/* File 2 Input */}
                            <div className="space-y-2">
                                <label htmlFor="file2" className="block text-sm font-medium text-gray-300">
                                    File 2 (Predicted Data)
                                </label>
                                <input 
                                    id="file2" 
                                    type="file" 
                                    accept=".csv" 
                                    onChange={handleFile2Change} 
                                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100"
                                />
                                {file2 && <p className="text-green-400 text-xs mt-1">✅ Selected: {file2.name}</p>}
                            </div>
                        </div>

                        {error && <div className="bg-red-900/50 border border-red-700 rounded-lg p-3 text-red-300">{error}</div>}

                        <button 
                            type="submit" 
                            disabled={loading || !file1 || !file2} 
                            className="w-full bg-gradient-to-r from-teal-500 to-green-600 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                                    <span>Comparing...</span>
                                </>
                            ) : (
                                <>
                                    <span>🔍</span>
                                    <span>Compare Files</span>
                                </>
                            )}
                        </button>
                    </form>

                    {result && (
                        <div className="result-card mt-8 p-6 bg-gray-900 rounded-lg border border-green-500">
                            <h2 className="text-2xl font-bold text-white mb-4">Comparison Result</h2>
                            <div className="flex items-center justify-center mb-4">
                                <div className="relative w-32 h-32 flex items-center justify-center">
                                    <svg className="w-full h-full" viewBox="0 0 100 100">
                                        <circle className="text-gray-700" strokeWidth="10" cx="50" cy="50" r="45" fill="transparent"/>
                                        <circle
                                            className="text-green-500"
                                            strokeWidth="10"
                                            strokeDasharray={2 * Math.PI * 45}
                                            strokeDashoffset={2 * Math.PI * 45 - (2 * Math.PI * 45 * (result.matchingPercentage / 100))}
                                            cx="50"
                                            cy="50"
                                            r="45"
                                            fill="transparent"
                                            strokeLinecap="round"
                                            transform="rotate(-90 50 50)"
                                        />
                                        <text x="50%" y="50%" dominantBaseline="middle" textAnchor="middle" className="text-3xl font-bold fill-current text-green-400">
                                            {result.matchingPercentage}%
                                        </text>
                                    </svg>
                                </div>
                            </div>
                            <p className="text-center font-semibold text-gray-300">Data Match Percentage</p>
                            
                            <div className="result-details mt-4 space-y-2 text-sm">
                                <p className="text-gray-400"><strong>Common Columns:</strong> {result.commonHeaders.join(', ')}</p>
                                <p className="text-gray-400"><strong>Total Rows in File 1:</strong> {result.totalItemsFile1}</p>
                                <p className="text-gray-400"><strong>Matching Rows Found:</strong> <span className="text-green-400">{result.totalMatches}</span></p>
                                <p className="text-gray-400"><strong>Mismatched Rows:</strong> <span className="text-yellow-400">{result.unmatchedRows.length}</span></p>
                            </div>
                            
                            {result.matchingRows.length > 0 && (
                                <div className="download-container mt-6">
                                    <button onClick={handleDownload} className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg font-semibold transition-colors">
                                        Download Matched Data (.csv)
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default Matching;