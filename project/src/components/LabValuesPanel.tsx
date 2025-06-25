import React from 'react';
import { TestTube, Droplets, Zap } from 'lucide-react';
import { LabValues } from '../types/patient';

interface LabValuesPanelProps {
  labs: LabValues;
}

export const LabValuesPanel: React.FC<LabValuesPanelProps> = ({ labs }) => {
  const criticalLabs = [
    {
      label: 'WBC',
      value: labs.WBC,
      unit: 'K/μL',
      normal: [4.0, 12.0],
      icon: TestTube,
      color: 'text-blue-600'
    },
    {
      label: 'Lactate',
      value: labs.Lactate,
      unit: 'mmol/L',
      normal: [0.5, 2.2],
      icon: Droplets,
      color: 'text-red-600'
    },
    {
      label: 'Creatinine',
      value: labs.Creatinine,
      unit: 'mg/dL',
      normal: [0.7, 1.3],
      icon: TestTube,
      color: 'text-green-600'
    },
    {
      label: 'pH',
      value: labs.pH,
      unit: '',
      normal: [7.35, 7.45],
      icon: Zap,
      color: 'text-purple-600'
    },
    {
      label: 'Platelets',
      value: labs.Platelets,
      unit: 'K/μL',
      normal: [150, 450],
      icon: Droplets,
      color: 'text-orange-600'
    },
    {
      label: 'Bilirubin Total',
      value: labs.Bilirubin_total,
      unit: 'mg/dL',
      normal: [0.3, 1.2],
      icon: TestTube,
      color: 'text-yellow-600'
    }
  ];

  const getLabStatus = (value: number, normal: [number, number]) => {
    if (value < normal[0] || value > normal[1]) {
      return 'abnormal';
    }
    return 'normal';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Critical Lab Values</h3>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {criticalLabs.map((lab) => {
          const Icon = lab.icon;
          const status = getLabStatus(lab.value, lab.normal);
          
          return (
            <div key={lab.label} className="flex items-center space-x-3 p-3 rounded-lg bg-gray-50">
              <div className={`p-2 rounded-full bg-white ${lab.color}`}>
                <Icon className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <div className="text-sm font-medium text-gray-600">{lab.label}</div>
                <div className={`text-lg font-bold ${
                  status === 'abnormal' ? 'text-danger-600' : 'text-gray-900'
                }`}>
                  {typeof lab.value === 'number' ? lab.value.toFixed(1) : lab.value}
                  {lab.unit && <span className="text-sm font-normal text-gray-500 ml-1">{lab.unit}</span>}
                </div>
                <div className="text-xs text-gray-500">
                  Normal: {lab.normal[0]}-{lab.normal[1]} {lab.unit}
                </div>
              </div>
              {status === 'abnormal' && (
                <div className="w-2 h-2 bg-danger-500 rounded-full animate-pulse-soft" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};