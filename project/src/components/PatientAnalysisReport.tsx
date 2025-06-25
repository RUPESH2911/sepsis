import React from 'react';
import { FileText, AlertTriangle, CheckCircle, Clock, User, TrendingUp } from 'lucide-react';
import { PatientAnalysisReport as ReportType } from '../types/dataset';

interface PatientAnalysisReportProps {
  report: ReportType;
}

export const PatientAnalysisReport: React.FC<PatientAnalysisReportProps> = ({ report }) => {
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'LOW': return 'bg-success-100 text-success-800 border-success-200';
      case 'MODERATE': return 'bg-warning-100 text-warning-800 border-warning-200';
      case 'HIGH': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'CRITICAL': return 'bg-danger-100 text-danger-800 border-danger-200';
      case 'UNCERTAIN': return 'bg-gray-100 text-gray-800 border-gray-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'CRITICAL': return <AlertTriangle className="w-5 h-5" />;
      case 'HIGH': return <AlertTriangle className="w-5 h-5" />;
      case 'UNCERTAIN': return <AlertTriangle className="w-5 h-5" />;
      default: return <CheckCircle className="w-5 h-5" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-primary-100 rounded-lg">
            <FileText className="w-6 h-6 text-primary-600" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Patient Analysis Report</h3>
            <p className="text-sm text-gray-600">Generated: {report.timestamp.toLocaleString()}</p>
          </div>
        </div>
        <div className={`px-4 py-2 rounded-full border flex items-center space-x-2 ${getRiskColor(report.overallRisk)}`}>
          {getRiskIcon(report.overallRisk)}
          <span className="font-medium">{report.overallRisk} RISK</span>
        </div>
      </div>

      {/* Patient Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <User className="w-4 h-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-600">Patient ID</span>
          </div>
          <div className="text-lg font-bold text-gray-900">{report.patientId}</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="w-4 h-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-600">Risk Probability</span>
          </div>
          <div className="text-lg font-bold text-gray-900">{(report.riskProbability * 100).toFixed(1)}%</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <CheckCircle className="w-4 h-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-600">Confidence</span>
          </div>
          <div className="text-lg font-bold text-gray-900">{(report.confidence * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* Uncertainty Warning */}
      {report.overallRisk === 'UNCERTAIN' && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-yellow-600" />
            <span className="font-medium text-yellow-800">Diagnostic Uncertainty Detected</span>
          </div>
          <p className="text-sm text-yellow-700">
            The analysis indicates insufficient data or conflicting indicators for a confident diagnosis. 
            Exercise caution in treatment decisions and consider obtaining additional clinical data.
          </p>
        </div>
      )}

      {/* Clinical Findings */}
      {report.clinicalFindings.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-3">Clinical Findings</h4>
          <div className="space-y-2">
            {report.clinicalFindings.map((finding, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                <span className="text-gray-800">{finding}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Threshold Violations */}
      {report.thresholdViolations.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-3">Threshold Violations</h4>
          <div className="space-y-2">
            {report.thresholdViolations.map((violation, index) => (
              <div key={index} className={`p-3 rounded-lg border ${
                violation.severity === 'critical' 
                  ? 'bg-danger-50 border-danger-200' 
                  : 'bg-warning-50 border-warning-200'
              }`}>
                <div className="flex items-center justify-between">
                  <span className={`font-medium ${
                    violation.severity === 'critical' ? 'text-danger-800' : 'text-warning-800'
                  }`}>
                    {violation.parameter}
                  </span>
                  <span className={`text-sm ${
                    violation.severity === 'critical' ? 'text-danger-600' : 'text-warning-600'
                  }`}>
                    {violation.severity.toUpperCase()}
                  </span>
                </div>
                <div className={`text-sm ${
                  violation.severity === 'critical' ? 'text-danger-700' : 'text-warning-700'
                }`}>
                  Current: {violation.value} | Threshold: {violation.threshold}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {report.recommendations.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-3">Clinical Recommendations</h4>
          <div className="space-y-2">
            {report.recommendations.map((rec, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
                <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0" />
                <span className="text-gray-800">{rec}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Treatment Plan */}
      {report.treatmentPlan.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-3">Treatment Plan</h4>
          <div className="space-y-2">
            {report.treatmentPlan.map((plan, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-purple-50 rounded-lg">
                <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-medium text-purple-600">{index + 1}</span>
                </div>
                <span className="text-gray-800">{plan}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Uncertainty Factors */}
      {report.uncertaintyFactors.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-3">Uncertainty Factors</h4>
          <div className="space-y-2">
            {report.uncertaintyFactors.map((factor, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg">
                <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
                <span className="text-gray-800">{factor}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Follow-up Actions */}
      {report.followUpActions.length > 0 && (
        <div>
          <h4 className="text-lg font-semibold text-gray-900 mb-3">Follow-up Actions</h4>
          <div className="space-y-2">
            {report.followUpActions.map((action, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                <Clock className="w-4 h-4 text-gray-600 mt-0.5 flex-shrink-0" />
                <span className="text-gray-800">{action}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};