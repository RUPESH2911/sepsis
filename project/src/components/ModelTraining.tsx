import React, { useState, useEffect } from 'react';
import { Play, Square, BarChart3, Target, TrendingUp, Loader, CheckCircle } from 'lucide-react';
import { SepsisMLModel } from '../utils/mlModel';
import { ModelMetrics, DatasetRow } from '../types/dataset';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

interface ModelTrainingProps {
  dataset: DatasetRow[];
  onModelTrained: (metrics: ModelMetrics) => void;
}

export const ModelTraining: React.FC<ModelTrainingProps> = ({ dataset, onModelTrained }) => {
  const [model] = useState(() => new SepsisMLModel());
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [trainingLog, setTrainingLog] = useState<string[]>([]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isTraining) {
      interval = setInterval(() => {
        setTrainingProgress(model.getTrainingProgress());
      }, 100);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isTraining, model]);

  const startTraining = async () => {
    if (!dataset || dataset.length === 0) {
      alert('Please upload a dataset first');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingLog(['Starting model training...']);

    try {
      // Add training log messages
      setTimeout(() => setTrainingLog(prev => [...prev, 'Preprocessing data...']), 500);
      setTimeout(() => setTrainingLog(prev => [...prev, 'Feature engineering...']), 1000);
      setTimeout(() => setTrainingLog(prev => [...prev, 'Training Random Forest classifier...']), 1500);
      setTimeout(() => setTrainingLog(prev => [...prev, 'Optimizing hyperparameters...']), 2500);
      setTimeout(() => setTrainingLog(prev => [...prev, 'Validating model performance...']), 3500);

      const trainedMetrics = await model.trainModel(dataset);
      setMetrics(trainedMetrics);
      onModelTrained(trainedMetrics);
      
      setTrainingLog(prev => [...prev, 'Model training completed successfully!']);
    } catch (error) {
      console.error('Training failed:', error);
      setTrainingLog(prev => [...prev, 'Training failed: ' + (error as Error).message]);
    } finally {
      setIsTraining(false);
    }
  };

  const stopTraining = () => {
    setIsTraining(false);
    setTrainingLog(prev => [...prev, 'Training stopped by user']);
  };

  const formatConfusionMatrix = (matrix: number[][]) => {
    return [
      { label: 'True Negative', value: matrix[0][0], color: '#10b981' },
      { label: 'False Positive', value: matrix[0][1], color: '#f59e0b' },
      { label: 'False Negative', value: matrix[1][0], color: '#ef4444' },
      { label: 'True Positive', value: matrix[1][1], color: '#3b82f6' }
    ];
  };

  return (
    <div className="space-y-6">
      {/* Training Controls */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Model Training</h3>
          <div className="flex items-center space-x-3">
            {!isTraining ? (
              <button
                onClick={startTraining}
                disabled={!dataset || dataset.length === 0}
                className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                <Play className="w-4 h-4" />
                <span>Start Training</span>
              </button>
            ) : (
              <button
                onClick={stopTraining}
                className="flex items-center space-x-2 px-4 py-2 bg-danger-600 text-white rounded-lg hover:bg-danger-700 transition-colors"
              >
                <Square className="w-4 h-4" />
                <span>Stop Training</span>
              </button>
            )}
          </div>
        </div>

        {/* Dataset Info */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-600">Dataset Size</div>
            <div className="text-2xl font-bold text-gray-900">{dataset?.length.toLocaleString() || 0}</div>
            <div className="text-xs text-gray-500">patient records</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-600">Features</div>
            <div className="text-2xl font-bold text-gray-900">{dataset?.[0] ? Object.keys(dataset[0]).length - 2 : 0}</div>
            <div className="text-xs text-gray-500">excluding ID and target</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-600">Sepsis Cases</div>
            <div className="text-2xl font-bold text-gray-900">
              {dataset ? dataset.filter(row => row.SepsisLabel === 1).length : 0}
            </div>
            <div className="text-xs text-gray-500">
              {dataset && dataset.length > 0 
                ? `${((dataset.filter(row => row.SepsisLabel === 1).length / dataset.length) * 100).toFixed(1)}%`
                : '0%'
              }
            </div>
          </div>
        </div>

        {/* Training Progress */}
        {isTraining && (
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <Loader className="w-5 h-5 text-primary-600 animate-spin" />
              <span className="font-medium text-gray-900">Training in progress...</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-primary-600 h-3 rounded-full transition-all duration-300"
                style={{ width: `${trainingProgress}%` }}
              />
            </div>
            <div className="text-sm text-gray-600">{trainingProgress.toFixed(1)}% complete</div>
          </div>
        )}

        {/* Training Log */}
        {trainingLog.length > 0 && (
          <div className="mt-4">
            <h4 className="font-medium text-gray-900 mb-2">Training Log</h4>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-32 overflow-y-auto">
              {trainingLog.map((log, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span>
                  <span>{log}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Model Performance Metrics */}
      {metrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Overview */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center space-x-2 mb-4">
              <CheckCircle className="w-5 h-5 text-success-600" />
              <h4 className="text-lg font-semibold text-gray-900">Model Performance</h4>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-primary-50 rounded-lg">
                <div className="flex items-center space-x-2">
                  <Target className="w-5 h-5 text-primary-600" />
                  <span className="text-sm font-medium text-primary-800">Accuracy</span>
                </div>
                <div className="text-2xl font-bold text-primary-900">{(metrics.accuracy * 100).toFixed(1)}%</div>
              </div>
              
              <div className="p-4 bg-success-50 rounded-lg">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-5 h-5 text-success-600" />
                  <span className="text-sm font-medium text-success-800">AUC Score</span>
                </div>
                <div className="text-2xl font-bold text-success-900">{metrics.auc.toFixed(3)}</div>
              </div>
              
              <div className="p-4 bg-warning-50 rounded-lg">
                <div className="text-sm font-medium text-warning-800">Precision</div>
                <div className="text-xl font-bold text-warning-900">{(metrics.precision * 100).toFixed(1)}%</div>
              </div>
              
              <div className="p-4 bg-orange-50 rounded-lg">
                <div className="text-sm font-medium text-orange-800">Recall</div>
                <div className="text-xl font-bold text-orange-900">{(metrics.recall * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>

          {/* Feature Importance */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance</h4>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={metrics.featureImportance.slice(0, 8)} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis type="number" stroke="#6b7280" fontSize={12} />
                  <YAxis 
                    type="category" 
                    dataKey="feature" 
                    stroke="#6b7280" 
                    fontSize={10}
                    width={80}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px'
                    }}
                    formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Importance']}
                  />
                  <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Confusion Matrix */}
      {metrics && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Confusion Matrix</h4>
          <div className="grid grid-cols-2 gap-4 max-w-md">
            {formatConfusionMatrix(metrics.confusionMatrix).map((item, index) => (
              <div key={index} className="p-4 rounded-lg border-2" style={{ borderColor: item.color }}>
                <div className="text-sm font-medium" style={{ color: item.color }}>
                  {item.label}
                </div>
                <div className="text-2xl font-bold text-gray-900">{item.value}</div>
              </div>
            ))}
          </div>
          
          <div className="mt-4 text-sm text-gray-600">
            <p><strong>True Positive:</strong> Correctly identified sepsis cases</p>
            <p><strong>True Negative:</strong> Correctly identified non-sepsis cases</p>
            <p><strong>False Positive:</strong> Incorrectly flagged as sepsis</p>
            <p><strong>False Negative:</strong> Missed sepsis cases</p>
          </div>
        </div>
      )}
    </div>
  );
};