import React from 'react';
import { AlertTriangle, X } from 'lucide-react';

interface ThresholdAlertProps {
  violations: Array<{
    parameter: string;
    value: number;
    threshold: number;
    severity: 'warning' | 'critical';
  }>;
  patientId: string;
  onClose: () => void;
}

export const ThresholdAlert: React.FC<ThresholdAlertProps> = ({
  violations,
  patientId,
  onClose
}) => {
  const criticalViolations = violations.filter(v => v.severity === 'critical');
  const warningViolations = violations.filter(v => v.severity === 'warning');

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 flex flex-col max-h-[90vh]">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-danger-100 rounded-full">
              <AlertTriangle className="w-6 h-6 text-danger-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Threshold Alert</h3>
              <p className="text-sm text-gray-600">Patient: {patientId}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {criticalViolations.length > 0 && (
            <div className="mb-4">
              <h4 className="font-medium text-danger-800 mb-2">Critical Violations</h4>
              <div className="space-y-2">
                {criticalViolations.map((violation, index) => (
                  <div key={index} className="p-3 bg-danger-50 border border-danger-200 rounded-lg">
                    <div className="font-medium text-danger-800">{violation.parameter}</div>
                    <div className="text-sm text-danger-700">
                      Current: {violation.value} | Threshold: {violation.threshold}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {warningViolations.length > 0 && (
            <div>
              <h4 className="font-medium text-warning-800 mb-2">Warning Violations</h4>
              <div className="space-y-2">
                {warningViolations.map((violation, index) => (
                  <div key={index} className="p-3 bg-warning-50 border border-warning-200 rounded-lg">
                    <div className="font-medium text-warning-800">{violation.parameter}</div>
                    <div className="text-sm text-warning-700">
                      Current: {violation.value} | Threshold: {violation.threshold}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-700">
              <strong>Action Required:</strong> Review patient status and consider appropriate clinical interventions based on threshold violations.
            </p>
          </div>
        </div>

        <div className="flex justify-end p-6 border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            Acknowledge
          </button>
        </div>
      </div>
    </div>
  );
};