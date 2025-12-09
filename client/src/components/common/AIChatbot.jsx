import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

// Gemini API Configuration
// Using environment variable for API key
const API_KEY = import.meta.env.VITE_GEMINI_API_KEY || "";
// UPDATED: Use the exact model name from your available models
const MODEL_NAME = "gemini-2.5-flash"; // This is confirmed available in your project
const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_NAME}:generateContent?key=${API_KEY}`;
const SYSTEM_INSTRUCTION = {
    parts: [{
        text: `You are ExoBot, an advanced AI assistant for the ExoDiscover exoplanet discovery platform. Your primary purpose is to help users navigate the site and understand the machine learning models (Kepler/KOI, K2, TESS/TOI, Custom Models).

EXPANDED KNOWLEDGE DOMAIN:
You can now discuss topics related to:
- NASA missions, spacecraft, and programs (past, present, and future)
- International space agencies including ISRO, ESA, Roscosmos, etc.
- Astronomy, astrophysics, and space science
- Planetary science and exploration
- Space technology and engineering
- Exoplanet research and discovery
- Rocket launches and space missions
- Space telescopes and observatories
- Astronauts and human spaceflight
- Cosmology and universe studies

RULES:
1. Prioritize answering questions related to astronomy, exoplanets, space missions, NASA programs, and the functionality of this website (ExoDiscover AI). 
2. For off-topic queries (completely unrelated to space/NASA/astronomy), politely decline and redirect focus to space-related topics.
3. Maintain a friendly, concise, and professional tone, focusing on scientific facts or helpful site navigation.
4. When discussing ISRO or other space agencies, maintain focus on their space missions and relate them to broader space exploration context.
5. If the user mentions a site section or task (like 'TESS analysis', 'Kepler model', 'profile update', 'security', 'file comparison', 'matching', 'compare files'), you MUST respond using the structured JSON output below, including the relevant internal route.

AVAILABLE ROUTES/LABELS:
- TESS Dashboard: /user/dashboard/toi
- Kepler Dashboard: /user/dashboard/koi
- K2 Dashboard: /user/dashboard/k2
- Custom Models: /user/dashboard/custom
- Main Dashboard: /user/dashboard
- Profile Settings: /user/profile
- Security Settings: /user/security
- File Comparison: /user/compare
- Upload Data: /user/upload
- Data Analysis: /user/analysis
- Results History: /user/history
- Help & Documentation: /user/help
- About Platform: /user/about
- About delete: /user/security
- About change password: /user/security
- About link Accounts: /user/security
OUTPUT FORMAT:
ALWAYS respond with a single JSON object conforming to this schema. DO NOT include any Markdown, headers, or surrounding text outside the JSON object.

{
  "text_response": "The main, concise answer to the user's question (Max 200 words). Use markdown formatting like bold for emphasis. For space mission questions, provide accurate, up-to-date information about NASA and other space agencies.",
  "suggested_route": {
    "label": "Optional label for the button (if routing is relevant).",
    "path": "Optional internal path for routing (if routing is relevant)."
  }
}`
    }]
};

// Helper function to handle basic markdown (bold) and ensure string rendering
const renderTextWithMarkdown = (text) => {
    if (!text || typeof text !== 'string') return text || ''; 
    
    // Split the text by the bold markdown pattern (**)
    const parts = text.split(/(\*\*.+?\*\*)/g).map((part, index) => {
        // If the part starts and ends with **, render it as <strong>
        if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={index} style={{ fontWeight: '600' }}>{part.slice(2, -2)}</strong>;
        }
        // Otherwise, render it as plain text
        return part;
    });

    return <>{parts}</>;
};

// Chat Message Component
const ChatMessage = ({ message, onRouteClick }) => {
    const isBot = message.sender === 'bot';
    
    // Parse the JSON content if it's the bot, otherwise use plain text
    let textResponse = message.text;
    let suggestedRoute = null;

    if (isBot && message.isStructured) {
        try {
            const parsed = JSON.parse(message.text);
            // Ensure textResponse is a string before rendering
            textResponse = String(parsed.text_response) || "I have generated a response, but it seems there was an issue displaying the text.";
            suggestedRoute = parsed.suggested_route || null;
        } catch (e) {
            textResponse = "I'm sorry, I encountered an error processing the structured response. My apologies.";
            console.error("Error parsing AI response:", e, message.text);
        }
    }

    return (
        <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
            <div className={`max-w-xs md:max-w-md p-3 rounded-lg shadow-md ${
                isBot ? 'bg-gray-700 rounded-bl-none text-white' : 'bg-blue-600 rounded-br-none text-white'
            }`}>
                {/* Applied: Use the markdown rendering helper */}
                <p className="text-sm whitespace-pre-wrap">{renderTextWithMarkdown(textResponse)}</p>
                {isBot && suggestedRoute?.path && suggestedRoute?.label && (
                    <button
                        onClick={() => onRouteClick(suggestedRoute.path)}
                        className="mt-3 w-full bg-cyan-600 hover:bg-cyan-700 text-white text-xs font-semibold py-2 px-3 rounded-md transition-colors"
                    >
                        {suggestedRoute.label} ↗
                    </button>
                )}
            </div>
        </div>
    );
};

const AIChatbot = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [apiStatus, setApiStatus] = useState('connected'); // We know API is connected now
    const navigate = useNavigate();
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);
    
    // Welcome message on open
    useEffect(() => {
        if (isOpen && messages.length === 0) {
            setMessages([{
                sender: 'bot',
                text: JSON.stringify({
                    text_response: "Hello! I'm ExoBot, your AI space mission assistant. I can help you understand NASA missions (like **James Webb**, **Artemis**, **Perseverance**), discuss ISRO achievements, explain exoplanet discoveries, or guide you through our platform features. What would you like to know about space exploration?",
                    suggested_route: { label: "Explore Missions", path: "/user/dashboard" }
                }),
                isStructured: true
            }]);
        }
    }, [isOpen]);

    const callGeminiApi = async (userPrompt) => {
        // Check if API key is available
        if (!API_KEY) {
            throw new Error("API key is not configured. Please add VITE_GEMINI_API_KEY to your environment variables.");
        }

        // We only use text responses for history since structured output is expected
        const history = messages.slice(1).map(msg => {
            let textPart = msg.text;
            if (msg.isStructured) {
                try {
                    // Extract only the text_response for conversation continuity
                    const parsed = JSON.parse(msg.text);
                    textPart = parsed.text_response;
                } catch (e) {
                    textPart = 'Internal error message.'; // Fallback for model context
                }
            }
            return {
                role: msg.sender === 'bot' ? 'model' : 'user',
                parts: [{ text: textPart }]
            };
        });
        
        // Add current user prompt
        history.push({ role: 'user', parts: [{ text: userPrompt }] });

        const payload = {
            contents: history,
            generationConfig: {
                temperature: 0.7,
                topP: 0.95,
                topK: 64, // Matching the model's default
                maxOutputTokens: 1024,
                responseMimeType: "application/json",
                responseSchema: {
                    type: "OBJECT",
                    properties: {
                        "text_response": { 
                            "type": "STRING", 
                            "description": "A concise, conversational response (Max 200 words). Uses markdown." 
                        },
                        "suggested_route": {
                            "type": "OBJECT",
                            "properties": {
                                "label": { 
                                    "type": "STRING", 
                                    "description": "Short label for the button." 
                                },
                                "path": { 
                                    "type": "STRING", 
                                    "description": "Internal route path (e.g., /user/profile)." 
                                }
                            }
                        }
                    },
                    "required": ["text_response"],
                    "propertyOrdering": ["text_response", "suggested_route"]
                }
            },
            safetySettings: [
                {
                    category: "HARM_CATEGORY_HARASSMENT",
                    threshold: "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    category: "HARM_CATEGORY_HATE_SPEECH",
                    threshold: "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold: "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    category: "HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold: "BLOCK_MEDIUM_AND_ABOVE"
                }
            ],
            systemInstruction: SYSTEM_INSTRUCTION
        };
        
        // --- Exponential Backoff Fetch Logic ---
        const maxRetries = 3;
        let lastError = null;
        
        for (let i = 0; i < maxRetries; i++) {
            try {
                const fetchResponse = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                if (!fetchResponse.ok) {
                    // Try to get error details
                    let errorMsg = `HTTP error! status: ${fetchResponse.status}`;
                    try {
                        const errorBody = await fetchResponse.json();
                        if (errorBody?.error?.message) {
                            errorMsg += ` - ${errorBody.error.message}`;
                        }
                    } catch (e) {
                        // Ignore if we can't parse error body
                    }
                    
                    throw new Error(errorMsg);
                }
                
                const response = await fetchResponse.json();
                
                // Check for response candidates
                if (response?.candidates?.[0]?.content?.parts?.[0]?.text) {
                    return response.candidates[0].content.parts[0].text;
                } else if (response?.promptFeedback?.blockReason) {
                    throw new Error(`Response blocked due to: ${response.promptFeedback.blockReason}`);
                } else {
                    throw new Error("Invalid response structure from AI");
                }
                
            } catch (error) {
                lastError = error;
                console.warn(`Attempt ${i + 1} failed:`, error.message);
                
                if (i === maxRetries - 1) {
                    // On last retry, throw a more informative error
                    throw new Error(`API failed after ${maxRetries} retries: ${lastError.message}`);
                }
                
                // Exponential backoff with jitter
                const delay = Math.pow(2, i) * 1000 + Math.random() * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
        
        throw lastError || new Error("Unknown API error");
    };

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput('');
        
        setMessages(prev => [...prev, { sender: 'user', text: userMessage, isStructured: false }]);
        setIsLoading(true);

        try {
            const botResponseText = await callGeminiApi(userMessage);

            setMessages(prev => [...prev, { sender: 'bot', text: botResponseText, isStructured: true }]);
        } catch (error) {
            console.error("AI Chatbot Error:", error);
            setMessages(prev => [...prev, { 
                sender: 'bot', 
                text: JSON.stringify({
                    text_response: `**Connection Error**: ${error.message}. Using Gemini 2.5 Flash model. Please ensure your API key has proper permissions.`,
                    suggested_route: null
                }),
                isStructured: true
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleRouteClick = (path) => {
        setIsOpen(false);
        navigate(path);
    };

    // Clear chat history when closing
    const handleCloseChat = () => {
        setMessages([]);
        setIsOpen(false);
    };

    return (
        <div className="fixed bottom-4 right-4 z-50">
            {/* Chat Window - Only show when isOpen is true */}
            {isOpen && (
                <div 
                    className="w-80 md:w-96 h-96 bg-gray-900 border border-blue-500/50 rounded-xl shadow-2xl flex flex-col overflow-hidden transform transition-all duration-300 mb-4"
                    style={{ 
                        animation: 'fadeInUp 0.3s ease-out'
                    }}
                >
                    {/* Chat Header */}
                    <div className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-700 to-purple-700 text-white shadow-md">
                        <div className="flex items-center space-x-2">
                            <span className="text-xl">🚀</span>
                            <div>
                                <h3 className="font-bold text-lg">ExoBot AI</h3>
                                <p className="text-xs opacity-80">Gemini 2.5 Flash</p>
                            </div>
                        </div>
                        <button
                            onClick={handleCloseChat}
                            className="text-gray-300 hover:text-white text-xl transition-colors"
                            aria-label="Close chat"
                        >
                            ✕
                        </button>
                    </div>

                    {/* Chat Messages Area */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-800" style={{ scrollbarWidth: 'thin' }}>
                        {messages.map((msg, index) => (
                            <ChatMessage key={index} message={msg} onRouteClick={handleRouteClick} />
                        ))}
                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="max-w-xs md:max-w-md p-3 rounded-lg bg-gray-700 rounded-bl-none text-white">
                                    <div className="flex items-center space-x-2">
                                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse delay-150"></div>
                                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse delay-300"></div>
                                    </div>
                                    <span className="text-sm text-gray-400 mt-2 block">ExoBot is thinking...</span>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Chat Input */}
                    <form onSubmit={handleSend} className="p-3 border-t border-gray-700 bg-gray-800">
                        <div className="flex space-x-2">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask about NASA missions, space, or exoplanets..."
                                className="flex-1 p-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:outline-none placeholder-gray-400"
                                disabled={isLoading}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        handleSend(e);
                                    }
                                }}
                            />
                            <button
                                type="submit"
                                disabled={!input.trim() || isLoading}
                                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isLoading ? '...' : 'Send'}
                            </button>
                        </div>
                        <p className="text-xs text-gray-400 mt-2">
                            Powered by Gemini 2.5 Flash • Press Enter to send
                        </p>
                    </form>
                </div>
            )}

            {/* Floating Chat Button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-16 h-16 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 text-white text-3xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 active:scale-95 flex items-center justify-center"
                style={{
                    boxShadow: '0 4px 15px rgba(59, 130, 246, 0.5)',
                    animation: 'floatBounce 4s ease-in-out infinite'
                }}
                aria-label="Open AI chat"
            >
                🚀
            </button>

            {/* Custom CSS for Animations */}
            <style>
                {`
                @keyframes floatBounce {
                    0% { transform: translateY(0) rotate(0deg); box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5); }
                    25% { transform: translateY(-5px) rotate(5deg); box-shadow: 0 8px 20px rgba(59, 130, 246, 0.8); }
                    50% { transform: translateY(-2px) rotate(-5deg); box-shadow: 0 6px 18px rgba(59, 130, 246, 0.7); }
                    75% { transform: translateY(-5px) rotate(5deg); box-shadow: 0 8px 20px rgba(59, 130, 246, 0.8); }
                    100% { transform: translateY(0) rotate(0deg); box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5); }
                }
                @keyframes fadeInUp {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                /* Custom scrollbar for chat area */
                .flex-1::-webkit-scrollbar {
                    width: 6px;
                }
                .flex-1::-webkit-scrollbar-thumb {
                    background-color: #4f46e5;
                    border-radius: 3px;
                }
                .flex-1::-webkit-scrollbar-track {
                    background: #1f2937;
                }
                `}
            </style>
        </div>
    );
};

export default AIChatbot;