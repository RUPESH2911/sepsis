import React from 'react';
import { Heart, Thermometer, Activity, Wind } from 'lucide-react';
import { VitalSigns } from '../types/patient';

interface VitalSignsPanelProps {
  vitals: VitalSigns;
}

export const VitalSignsPanel: React.FC<VitalSignsPanelProps> = ({ vitals }) => {
  const getVitalStatus = (value: number, normal: [number, number]) => {
    if (value < normal[0] || value > normal[1]) {
      return 'abnormal';
    }
    return 'normal';
  };

  const vitalItems = [
    {
      label: 'Heart Rate',
      value: vitals.HR,
      unit: 'bpm',
      icon: Heart,
      normal: [60, 100],
      color: 'text-red-600'
    },
    {
      label: 'Temperature',
      value: vitals.Temp,
      unit: '°C',
      icon: Thermometer,
      normal: [36.1, 37.2],
      color: 'text-orange-600'
    },
    {
      label: 'Blood Pressure',
      value: `${vitals.SBP}/${vitals.DBP}`,
      unit: 'mmHg',
      icon: Activity,
      normal: [90, 140],
      normalValue: vitals.SBP,
      color: 'text-blue-600'
    },
    {
      label: 'Respiratory Rate',
      value: vitals.Resp,
      unit: '/min',
      icon: Wind,
      normal: [12, 20],
      color: 'text-green-600'
    },
    {
      label: 'O2 Saturation',
      value: vitals.O2Sat,
      unit: '%',
      icon: Activity,
      normal: [95, 100],
      color: 'text-blue-600'
    },
    {
      label: 'MAP',
      value: vitals.MAP,
      unit: 'mmHg',
      icon: Activity,
      normal: [70, 100],
      color: 'text-purple-600'
    }
  ];

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Vital Signs</h3>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {vitalItems.map((item) => {
          const Icon = item.icon;
          const status = getVitalStatus(
            typeof item.normalValue !== 'undefined' ? item.normalValue : item.value as number,
            item.normal
          );
          
          return (
            <div key={item.label} className="flex items-center space-x-3 p-3 rounded-lg bg-gray-50">
              <div className={`p-2 rounded-full bg-white ${item.color}`}>
                <Icon className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <div className="text-sm font-medium text-gray-600">{item.label}</div>
                <div className={`text-lg font-bold ${
                  status === 'abnormal' ? 'text-danger-600' : 'text-gray-900'
                }`}>
                  {item.value} <span className="text-sm font-normal text-gray-500">{item.unit}</span>
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