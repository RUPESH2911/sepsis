import React, { useState } from 'react';
import { Settings, Save, RotateCcw, AlertTriangle } from 'lucide-react';
import { ThresholdSettings } from '../types/dataset';

interface DynamicSettingsProps {
  thresholds: ThresholdSettings;
  onThresholdsUpdate: (thresholds: ThresholdSettings) => void;
}

export const DynamicSettings: React.FC<DynamicSettingsProps> = ({
  thresholds,
  onThresholdsUpdate
}) => {
  const [localThresholds, setLocalThresholds] = useState<ThresholdSettings>(thresholds);
  const [hasChanges, setHasChanges] = useState(false);

  const handleThresholdChange = (parameter: string, field: string, value: number | boolean) => {
    const updated = {
      ...localThresholds,
      [parameter]: {
        ...localThresholds[parameter],
        [field]: value
      }
    };
    setLocalThresholds(updated);
    setHasChanges(true);
  };

  const handleSave = () => {
    onThresholdsUpdate(localThresholds);
    setHasChanges(false);
  };

  const handleReset = () => {
    setLocalThresholds(thresholds);
    setHasChanges(false);
  };

  const parameterLabels: { [key: string]: string } = {
    'HR': 'Heart Rate (bpm)',
    'Temp': 'Temperature (°C)',
    'SBP': 'Systolic BP (mmHg)',
    'Resp': 'Respiratory Rate (/min)',
    'O2Sat': 'Oxygen Saturation (%)',
    'WBC': 'White Blood Cells (K/μL)',
    'Lactate': 'Lactate (mmol/L)',
    'Creatinine': 'Creatinine (mg/dL)',
    'Platelets': 'Platelets (K/μL)',
    'MAP': 'Mean Arterial Pressure (mmHg)'
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary-100 rounded-lg">
              <Settings className="w-6 h-6 text-primary-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Dynamic Threshold Settings</h3>
              <p className="text-sm text-gray-600">Configure alert thresholds for clinical parameters</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {hasChanges && (
              <div className="flex items-center space-x-2 text-warning-600">
                <AlertTriangle className="w-4 h-4" />
                <span className="text-sm">Unsaved changes</span>
              </div>
            )}
            <button
              onClick={handleReset}
              disabled={!hasChanges}
              className="flex items-center space-x-2 px-3 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>
            <button
              onClick={handleSave}
              disabled={!hasChanges}
              className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Save className="w-4 h-4" />
              <span>Save Changes</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {Object.entries(localThresholds).map(([parameter, config]) => (
            <div key={parameter} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-medium text-gray-900">{parameterLabels[parameter] || parameter}</h4>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.enabled}
                    onChange={(e) => handleThresholdChange(parameter, 'enabled', e.target.checked)}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="text-sm text-gray-600">Enabled</span>
                </label>
              </div>

              {config.enabled && (
                <div className="space-y-4">
                  {config.min !== undefined && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Minimum Normal Value
                      </label>
                      <input
                        type="number"
                        value={config.min}
                        onChange={(e) => handleThresholdChange(parameter, 'min', parseFloat(e.target.value))}
                        step="0.1"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                      />
                    </div>
                  )}

                  {config.max !== undefined && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Maximum Normal Value
                      </label>
                      <input
                        type="number"
                        value={config.max}
                        onChange={(e) => handleThresholdChange(parameter, 'max', parseFloat(e.target.value))}
                        step="0.1"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                      />
                    </div>
                  )}

                  {config.critical !== undefined && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Critical Alert Threshold
                      </label>
                      <input
                        type="number"
                        value={config.critical}
                        onChange={(e) => handleThresholdChange(parameter, 'critical', parseFloat(e.target.value))}
                        step="0.1"
                        className="w-full px-3 py-2 border border-danger-300 rounded-md focus:ring-2 focus:ring-danger-500 focus:border-danger-500"
                      />
                      <p className="text-xs text-danger-600 mt-1">
                        Values beyond this threshold will trigger critical alerts
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* AI Configuration */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Analysis Configuration</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Confidence Threshold for Uncertain Diagnosis
            </label>
            <input
              type="range"
              min="0.3"
              max="0.8"
              step="0.05"
              defaultValue="0.6"
              className="w-full"
            />
            <div className="text-sm text-gray-500 mt-1">60% - Below this confidence, diagnosis marked as uncertain</div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              High Risk Probability Threshold
            </label>
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.05"
              defaultValue="0.7"
              className="w-full"
            />
            <div className="text-sm text-gray-500 mt-1">70% - Above this probability, patient marked as high risk</div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Auto-refresh Interval
            </label>
            <select className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-primary-500 focus:border-primary-500">
              <option value="5">5 seconds</option>
              <option value="10">10 seconds</option>
              <option value="30">30 seconds</option>
              <option value="60">1 minute</option>
              <option value="300">5 minutes</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Alert Notifications
            </label>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input type="checkbox" defaultChecked className="rounded border-gray-300 text-primary-600 focus:ring-primary-500" />
                <span className="text-sm text-gray-700">Critical threshold violations</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="checkbox" defaultChecked className="rounded border-gray-300 text-primary-600 focus:ring-primary-500" />
                <span className="text-sm text-gray-700">Uncertain diagnoses</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="checkbox" className="rounded border-gray-300 text-primary-600 focus:ring-primary-500" />
                <span className="text-sm text-gray-700">Model training completion</span>
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};