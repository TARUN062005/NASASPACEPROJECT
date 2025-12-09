// server/test_koi_connection.js - UPDATED
const axios = require('axios');

async function testKOIConnection() {
    console.log('🔍 Testing KOI Server Connection...');
    
    // Test data in correct format
    const testData = {
        data: {
            koi_period: 9.48803557,
            koi_impact: 0.146,
            koi_duration: 2.9575,
            koi_depth: 616,
            koi_prad: 2.26,
            koi_teq: 793,
            koi_insol: 93.59,
            koi_model_snr: 35.8,
            koi_steff: 5455,
            koi_slogg: 4.467,
            koi_srad: 0.927,
            koi_kepmag: 15.347
        }
    };
    
    try {
        // Test 1: Health endpoint
        console.log('\n1️⃣ Testing /health endpoint...');
        const healthResponse = await axios.get('http://localhost:5002/health');
        console.log('✅ Health check passed!');
        console.log('   Status:', healthResponse.data.status);
        console.log('   Model loaded:', healthResponse.data.model_loaded);
        console.log('   Accuracy:', healthResponse.data.accuracy);
        
        // Test 2: Model info endpoint
        console.log('\n2️⃣ Testing /model_info endpoint...');
        const modelInfoResponse = await axios.get('http://localhost:5002/model_info');
        console.log('✅ Model info retrieved!');
        console.log('   Model trained:', modelInfoResponse.data.is_trained);
        console.log('   Classes:', modelInfoResponse.data.class_names);
        console.log('   Features:', modelInfoResponse.data.selected_features.length);
        
        // Test 3: Performance endpoint
        console.log('\n3️⃣ Testing /model/performance endpoint...');
        const performanceResponse = await axios.get('http://localhost:5002/model/performance');
        console.log('✅ Performance data retrieved!');
        console.log('   Accuracy:', performanceResponse.data.performance_summary.current_accuracy);
        console.log('   Training samples:', performanceResponse.data.performance_summary.training_samples);
        
        // Test 4: Prediction endpoint
        console.log('\n4️⃣ Testing /predict endpoint...');
        console.log('📊 Sending test data:', JSON.stringify(testData, null, 2));
        
        const predictResponse = await axios.post('http://localhost:5002/predict', testData, {
            headers: { 'Content-Type': 'application/json' }
        });
        
        console.log('✅ Prediction successful!');
        console.log('   Response success:', predictResponse.data.success);
        console.log('   Model type:', predictResponse.data.model_type);
        
        if (predictResponse.data.data && predictResponse.data.data.prediction) {
            const pred = predictResponse.data.data.prediction;
            console.log('   Predicted class:', pred.predicted_class);
            console.log('   Confidence:', pred.confidence);
            console.log('   Is mock:', pred.is_mock || false);
        } else if (predictResponse.data.prediction) {
            const pred = predictResponse.data.prediction;
            console.log('   Predicted class:', pred.predicted_class);
            console.log('   Confidence:', pred.confidence);
            console.log('   Is mock:', pred.is_mock || false);
        }
        
        console.log('\n🎉 All tests passed! The KOI server is working correctly.');
        return true;
        
    } catch (error) {
        console.error('\n❌ Test failed:', error.message);
        if (error.response) {
            console.error('Status:', error.response.status);
            console.error('Response data:', JSON.stringify(error.response.data, null, 2));
        } else if (error.request) {
            console.error('No response received. Is the server running?');
            console.error('Make sure KOI server is running on port 5002:');
            console.error('   python app.py  (in ML_Model/KOI_Model directory)');
        } else {
            console.error('Error setting up request:', error.message);
        }
        return false;
    }
}

// Run the test
testKOIConnection().then(success => {
    if (success) {
        console.log('\n✅ All tests completed successfully!');
        console.log('🚀 You can now restart your Node.js server and test the frontend.');
    } else {
        console.log('\n❌ Tests failed. Please fix the issues above.');
        console.log('💡 Try restarting the KOI server and running the tests again.');
    }
    process.exit(success ? 0 : 1);
});