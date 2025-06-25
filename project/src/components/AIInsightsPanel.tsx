import React, { useState, useEffect } from 'react';
import { Brain, Lightbulb, AlertTriangle, TrendingUp, Loader, Sparkles } from 'lucide-react';
import { AIService } from '../utils/aiService';
import { AIInsight } from '../types/dataset';

interface AIInsightsPanelProps {
  patientData?: any;
  prediction?: any;
  dataset?: any[];
}

export const AIInsightsPanel: React.FC<AIInsightsPanelProps> = ({
  patientData,
  prediction,
  dataset
}) => {
  const [insights, setInsights] = useState<AIInsight[]>([]);
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [analysis, setAnalysis] = useState<any>(null);

  const aiService = AIService.getInstance();

  useEffect(() => {
    if (patientData && prediction) {
      generatePatientInsights();
    }
  }, [patientData, prediction]);

  useEffect(() => {
    if (dataset && dataset.length > 0) {
      generateDatasetInsights();
    }
  }, [dataset]);

  const generatePatientInsights = async () => {
    if (!patientData || !prediction) return;
    
    setLoading(true);
    try {
      const [aiAnalysis, clinicalRecs] = await Promise.all([
        aiService.analyzePatientData(patientData),
        aiService.generateClinicalRecommendations(patientData, prediction)
      ]);
      
      setAnalysis(aiAnalysis);
      setRecommendations(clinicalRecs);
      
      // Generate insights
      const newInsights: AIInsight[] = [
        {
          id: '1',
          type: 'risk_assessment',
          title: 'AI Risk Assessment',
          description: `Overall sepsis risk: ${aiAnalysis.riskAssessment.overallRisk.toUpperCase()}`,
          confidence: aiAnalysis.riskAssessment.confidence,
          priority: aiAnalysis.riskAssessment.overallRisk === 'critical' ? 'critical' : 
                   aiAnalysis.riskAssessment.overallRisk === 'high' ? 'high' : 'medium',
          timestamp: new Date()
        }
      ];

      if (aiAnalysis.riskAssessment.riskFactors.length > 0) {
        newInsights.push({
          id: '2',
          type: 'pattern_analysis',
          title: 'Risk Factors Identified',
          description: `${aiAnalysis.riskAssessment.riskFactors.length} risk factors detected: ${aiAnalysis.riskAssessment.riskFactors.slice(0, 3).join(', ')}`,
          confidence: 0.9,
          priority: 'high',
          timestamp: new Date()
        });
      }

      if (prediction.probability > 0.7) {
        newInsights.push({
          id: '3',
          type: 'clinical_alert',
          title: 'High Risk Alert',
          description: 'Patient requires immediate clinical attention and sepsis protocol activation',
          confidence: 0.95,
          priority: 'critical',
          timestamp: new Date()
        });
      }

      setInsights(newInsights);
    } catch (error) {
      console.error('Error generating insights:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateDatasetInsights = async () => {
    if (!dataset) return;
    
    setLoading(true);
    try {
      const datasetAnalysis = await aiService.generateDatasetInsights(dataset);
      
      const datasetInsights: AIInsight[] = [
        {
          id: 'dataset-1',
          type: 'pattern_analysis',
          title: 'Dataset Analysis Complete',
          description: `Analyzed ${dataset.length} patient records with ${datasetAnalysis.patterns.sepsisPrevalence}% sepsis prevalence`,
          confidence: 0.95,
          priority: 'medium',
          timestamp: new Date()
        },
        {
          id: 'dataset-2',
          type: 'risk_assessment',
          title: 'Data Quality Assessment',
          description: `Dataset completeness: ${datasetAnalysis.qualityAssessment.completeness}% with ${datasetAnalysis.qualityAssessment.missingValueRate}% missing values`,
          confidence: 0.9,
          priority: 'medium',
          timestamp: new Date()
        }
      ];

      setInsights(prev => [...prev, ...datasetInsights]);
      setAnalysis(datasetAnalysis);
    } catch (error) {
      console.error('Error analyzing dataset:', error);
    } finally {
      setLoading(false);
    }
  };

  const getInsightIcon = (type: AIInsight['type']) => {
    switch (type) {
      case 'risk_assessment': return AlertTriangle;
      case 'intervention_recommendation': return Lightbulb;
      case 'pattern_analysis': return TrendingUp;
      case 'clinical_alert': return AlertTriangle;
      default: return Brain;
    }
  };

  const getPriorityColor = (priority: AIInsight['priority']) => {
    switch (priority) {
      case 'critical': return 'bg-danger-100 text-danger-800 border-danger-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-warning-100 text-warning-800 border-warning-200';
      case 'low': return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className="space-y-6">
      {/* AI Insights Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg p-6 text-white">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-white/20 rounded-lg">
            <Brain className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-xl font-semibold">AI Clinical Intelligence</h3>
            <p className="text-purple-100">Advanced ML-powered insights and recommendations</p>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-center space-x-3">
            <Loader className="w-6 h-6 text-primary-600 animate-spin" />
            <span className="text-gray-600">AI is analyzing patient data...</span>
          </div>
        </div>
      )}

      {/* Clinical Recommendations */}
      {recommendations.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Sparkles className="w-5 h-5 text-purple-600" />
            <h4 className="text-lg font-semibold text-gray-900">AI Clinical Recommendations</h4>
          </div>
          <div className="space-y-3">
            {recommendations.map((rec, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-purple-50 rounded-lg">
                <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 flex-shrink-0" />
                <span className="text-gray-800">{rec}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AI Insights */}
      {insights.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">AI Insights</h4>
          <div className="space-y-4">
            {insights.map((insight) => {
              const Icon = getInsightIcon(insight.type);
              return (
                <div key={insight.id} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3">
                      <div className="p-2 bg-gray-100 rounded-lg">
                        <Icon className="w-5 h-5 text-gray-600" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">{insight.title}</div>
                        <div className="text-sm text-gray-600 mt-1">{insight.description}</div>
                        <div className="text-xs text-gray-500 mt-2">
                          Confidence: {(insight.confidence * 100).toFixed(0)}% • {insight.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getPriorityColor(insight.priority)}`}>
                      {insight.priority.toUpperCase()}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Analysis Summary */}
      {analysis && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Analysis Summary</h4>
          
          {analysis.patterns && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="text-sm font-medium text-blue-800">Sepsis Prevalence</div>
                <div className="text-2xl font-bold text-blue-900">{analysis.patterns.sepsisPrevalence}%</div>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <div className="text-sm font-medium text-green-800">Avg ICU Stay</div>
                <div className="text-2xl font-bold text-green-900">{analysis.patterns.avgICUStay}h</div>
              </div>
            </div>
          )}

          {analysis.modelRecommendations && (
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Model Optimization Recommendations</h5>
              <div className="space-y-2">
                {analysis.modelRecommendations.map((rec: string, index: number) => (
                  <div key={index} className="text-sm text-gray-600 flex items-start space-x-2">
                    <span className="text-primary-500">•</span>
                    <span>{rec}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};