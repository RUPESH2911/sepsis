import React, { useState, useEffect } from 'react';
import { Upload, FileText, BarChart3, AlertCircle, CheckCircle, Loader } from 'lucide-react';
import { ExcelProcessor } from '../utils/excelProcessor';
import { EnhancedSepsisMLModel } from '../utils/enhancedMLModel';
import { DatasetRow } from '../types/dataset';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const ModelPerformance: React.FC = () => {
  const [trainingData, setTrainingData] = useState<DatasetRow[]>([]);
  const [testData, setTestData] = useState<DatasetRow[]>([]);
  const [model, setModel] = useState<EnhancedSepsisMLModel | null>(null);
  const [chartData, setChartData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<{
    training: 'idle' | 'uploading' | 'success' | 'error';
    test: 'idle' | 'uploading' | 'success' | 'error';
  }>({ training: 'idle', test: 'idle' });
  const [errorMessages, setErrorMessages] = useState<{ training?: string; test?: string }>({});

  const handleFileUpload = async (file: File, type: 'training' | 'test') => {
    setUploadStatus(prev => ({ ...prev, [type]: 'uploading' }));
    setErrorMessages(prev => ({ ...prev, [type]: undefined }));
    
    try {
      const { data } = await ExcelProcessor.processExcelFile(file);
      
      if (type === 'training') {
        const hasSepsisLabel = data.some(row => 
          row.SepsisLabel !== undefined || 
          row.sepsislabel !== undefined || 
          Object.keys(row).some(key => key.toLowerCase().includes('sepsis'))
        );
        
        if (!hasSepsisLabel) {
          throw new Error('Training data must contain sepsis labels (SepsisLabel column)');
        }
        
        setTrainingData(data);
      } else {
        setTestData(data);
      }
      
      setUploadStatus(prev => ({ ...prev, [type]: 'success' }));
    } catch (error) {
      console.error(`Error uploading ${type} data:`, error);
      setErrorMessages(prev => ({ ...prev, [type]: (error as Error).message }));
      setUploadStatus(prev => ({ ...prev, [type]: 'error' }));
    }
  };

  // Enhanced model training
  useEffect(() => {
    if (trainingData.length > 0 && !model) {
      const trainModel = async () => {
        setLoading(true);
        try {
          const newModel = new EnhancedSepsisMLModel();
          await newModel.trainModelInBackground(trainingData);
          setModel(newModel);
        } catch (error) {
          console.error('Model training failed:', error);
        } finally {
          setLoading(false);
        }
      };
      trainModel();
    }
  }, [trainingData]);

  // Enhanced prediction generation
  useEffect(() => {
    let cancelled = false;
    
    const generatePredictions = async () => {
      if (model && testData.length > 0 && trainingData.length > 0) {
        setLoading(true);
        setProgress(0);
        setChartData([]);
        
        try {
          // Process data in batches for better performance
          const batchSize = 10;
          const aggregatedData: any[] = [];
          
          for (let i = 0; i < testData.length; i += batchSize) {
            if (cancelled) break;
            
            const batch = testData.slice(i, i + batchSize);
            
            for (const testRow of batch) {
              const patientId = String(testRow.Patient_ID || testRow.PatientID || testRow.patient_id || '');
              
              // Enhanced AI prediction using clinical sepsis criteria
              const prediction = model.predictWithUncertainty(testRow);
              
              // Find corresponding actual data
              const actualRow = trainingData.find(row => 
                String(row.Patient_ID || row.PatientID || row.patient_id || '') === patientId
              );
              
              const actualSepsis = actualRow ? Number(actualRow.SepsisLabel || actualRow.sepsislabel || 0) : 0;
              
              aggregatedData.push({
                index: aggregatedData.length + 1,
                patientId,
                predicted: prediction.probability * 100,
                actual: actualSepsis * 100,
                confidence: prediction.confidence * 100,
                riskLevel: prediction.riskLevel
              });
            }
            
            setChartData([...aggregatedData]);
            setProgress(Math.round(((i + batchSize) / testData.length) * 100));
            
            // Small delay for smooth progress
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        } catch (error) {
          console.error('Prediction generation failed:', error);
        } finally {
          setLoading(false);
        }
      }
    };

    generatePredictions();
    
    return () => {
      cancelled = true;
    };
  }, [model, testData, trainingData]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return <CheckCircle className="w-5 h-5 text-success-600" />;
      case 'error': return <AlertCircle className="w-5 h-5 text-danger-600" />;
      case 'uploading': return <Loader className="w-5 h-5 text-primary-600 animate-spin" />;
      default: return <Upload className="w-5 h-5 text-gray-400" />;
    }
  };

  // Calculate simplified accuracy metrics
  const calculateAccuracy = () => {
    if (chartData.length === 0) return null;
    
    const threshold = 50;
    let correct = 0;
    let truePositives = 0;
    let falseNegatives = 0;
    let falsePositives = 0;
    let trueNegatives = 0;
    
    chartData.forEach(point => {
      const predictedPositive = point.predicted >= threshold;
      const actualPositive = point.actual >= threshold;
      
      if (predictedPositive === actualPositive) correct++;
      
      if (predictedPositive && actualPositive) truePositives++;
      else if (!predictedPositive && actualPositive) falseNegatives++;
      else if (predictedPositive && !actualPositive) falsePositives++;
      else trueNegatives++;
    });
    
    const accuracy = (correct / chartData.length) * 100;
    
    return { 
      accuracy, 
      truePositives, 
      falseNegatives, 
      falsePositives, 
      trueNegatives,
      totalPatients: chartData.length
    };
  };

  const accuracyData = calculateAccuracy();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-white/20 rounded-lg">
            <BarChart3 className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-2xl font-bold">AI Model Performance Analysis</h2>
            <p className="text-blue-100">Upload datasets to compare predicted vs actual sepsis outcomes</p>
          </div>
        </div>
      </div>

      {/* File Upload Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training Data Upload */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-4">
            <FileText className="w-5 h-5 text-primary-600" />
            <h3 className="text-lg font-semibold text-gray-900">Training Data</h3>
            {getStatusIcon(uploadStatus.training)}
          </div>
          
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-400 transition-colors">
            <input
              type="file"
              accept=".xlsx,.xls,.csv"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file, 'training');
              }}
              className="hidden"
              id="training-upload"
            />
            <label htmlFor="training-upload" className="cursor-pointer">
              <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <div className="text-sm font-medium text-primary-600">Upload Training Dataset</div>
              <div className="text-xs text-gray-500 mt-1">Must include sepsis labels</div>
            </label>
          </div>
          
          {errorMessages.training && (
            <div className="mt-3 p-3 bg-danger-50 border border-danger-200 rounded-lg">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-danger-600" />
                <span className="text-sm text-danger-800">{errorMessages.training}</span>
              </div>
            </div>
          )}
          
          {trainingData.length > 0 && (
            <div className="mt-3 p-3 bg-success-50 border border-success-200 rounded-lg">
              <div className="text-sm text-success-800">
                ✓ {trainingData.length.toLocaleString()} records loaded
              </div>
            </div>
          )}
        </div>

        {/* Test Data Upload */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-4">
            <FileText className="w-5 h-5 text-orange-600" />
            <h3 className="text-lg font-semibold text-gray-900">Test Data</h3>
            {getStatusIcon(uploadStatus.test)}
          </div>
          
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-orange-400 transition-colors">
            <input
              type="file"
              accept=".xlsx,.xls,.csv"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file, 'test');
              }}
              className="hidden"
              id="test-upload"
            />
            <label htmlFor="test-upload" className="cursor-pointer">
              <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <div className="text-sm font-medium text-orange-600">Upload Test Dataset</div>
              <div className="text-xs text-gray-500 mt-1">Patient data for prediction</div>
            </label>
          </div>
          
          {errorMessages.test && (
            <div className="mt-3 p-3 bg-danger-50 border border-danger-200 rounded-lg">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-danger-600" />
                <span className="text-sm text-danger-800">{errorMessages.test}</span>
              </div>
            </div>
          )}
          
          {testData.length > 0 && (
            <div className="mt-3 p-3 bg-success-50 border border-success-200 rounded-lg">
              <div className="text-sm text-success-800">
                ✓ {testData.length.toLocaleString()} records loaded
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Progress Indicator */}
      {loading && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-4">
            <Loader className="w-5 h-5 text-primary-600 animate-spin" />
            <span className="font-medium text-gray-900">
              {trainingData.length > 0 && !model ? 'Training AI model...' : 'Generating AI predictions...'}
            </span>
          </div>
          {progress > 0 && (
            <div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-primary-500 to-purple-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="text-sm text-gray-600 mt-2">{progress}% complete</div>
            </div>
          )}
        </div>
      )}

      {/* Simplified Accuracy Display */}
      {accuracyData && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Accuracy</h3>
          <div className="flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl font-bold text-primary-600 mb-2">
                {accuracyData.accuracy.toFixed(1)}%
              </div>
              <div className="text-gray-600">Overall Accuracy</div>
              <div className="text-sm text-gray-500 mt-1">
                {accuracyData.totalPatients} patients analyzed
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Comparison Chart */}
      {chartData.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">AI Prediction vs Actual Results</h3>
              <p className="text-sm text-gray-600">Red line: Actual sepsis | Blue line: AI predictions</p>
            </div>
          </div>
          
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="index" 
                  stroke="#6b7280" 
                  fontSize={12}
                  label={{ value: 'Patient Index', position: 'insideBottom', offset: -10 }}
                />
                <YAxis 
                  stroke="#6b7280" 
                  fontSize={12}
                  label={{ value: 'Sepsis Probability (%)', angle: -90, position: 'insideLeft' }}
                  domain={[0, 100]}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                  formatter={(value: any, name: string, props: any) => {
                    const data = props.payload;
                    if (name === 'predicted') {
                      return [
                        `${value.toFixed(1)}% (AI Prediction)`,
                        `Confidence: ${data.confidence.toFixed(1)}%`
                      ];
                    }
                    return [`${value.toFixed(1)}% (Actual)`, 'Actual Sepsis'];
                  }}
                  labelFormatter={(label, payload) => {
                    if (payload && payload.length > 0) {
                      return `Patient: ${payload[0].payload.patientId}`;
                    }
                    return `Patient ${label}`;
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#dc2626" 
                  strokeWidth={3}
                  name="Actual Sepsis"
                  dot={{ fill: '#dc2626', strokeWidth: 2, r: 4 }}
                  connectNulls={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  stroke="#3b82f6" 
                  strokeWidth={3}
                  strokeDasharray="6 6"
                  name="AI Prediction"
                  dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-1 bg-danger-500 rounded"></div>
              <span className="text-gray-600">Actual Sepsis</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-6 h-1 bg-primary-500 rounded" style={{ backgroundImage: 'repeating-linear-gradient(90deg, #3b82f6 0, #3b82f6 6px, transparent 6px, transparent 12px)' }}></div>
              <span className="text-gray-600">AI Prediction</span>
            </div>
          </div>
        </div>
      )}

      {/* Optimized Confusion Matrix */}
      {accuracyData && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Results Breakdown</h3>
          <div className="grid grid-cols-2 gap-4 max-w-lg mx-auto">
            <div className="p-4 bg-success-50 border-2 border-success-200 rounded-lg text-center">
              <div className="text-sm font-medium text-success-800">Correct Non-Sepsis</div>
              <div className="text-3xl font-bold text-success-900">{accuracyData.trueNegatives}</div>
            </div>
            <div className="p-4 bg-warning-50 border-2 border-warning-200 rounded-lg text-center">
              <div className="text-sm font-medium text-warning-800">False Alarms</div>
              <div className="text-3xl font-bold text-warning-900">{accuracyData.falsePositives}</div>
            </div>
            <div className="p-4 bg-danger-50 border-2 border-danger-200 rounded-lg text-center">
              <div className="text-sm font-medium text-danger-800">Missed Cases</div>
              <div className="text-3xl font-bold text-danger-900">{accuracyData.falseNegatives}</div>
            </div>
            <div className="p-4 bg-primary-50 border-2 border-primary-200 rounded-lg text-center">
              <div className="text-sm font-medium text-primary-800">Correct Sepsis</div>
              <div className="text-3xl font-bold text-primary-900">{accuracyData.truePositives}</div>
            </div>
          </div>
          
          {accuracyData.falseNegatives > 0 && (
            <div className="mt-4 p-4 bg-orange-50 border border-orange-200 rounded-lg">
              <div className="text-sm text-orange-800">
                <strong>Note:</strong> {accuracyData.falseNegatives} missed sepsis cases detected. 
                The AI model is continuously learning to reduce false negatives and improve patient safety.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};