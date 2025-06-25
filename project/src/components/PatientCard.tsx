import React from 'react';
import { User, Clock, MapPin, AlertTriangle } from 'lucide-react';
import { PatientData, PredictionResult } from '../types/patient';

interface PatientCardProps {
  patient: PatientData;
  prediction: PredictionResult;
  isSelected: boolean;
  onClick: () => void;
}

export const PatientCard: React.FC<PatientCardProps> = ({
  patient,
  prediction,
  isSelected,
  onClick
}) => {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'LOW': return 'bg-success-100 text-success-800 border-success-200';
      case 'MODERATE': return 'bg-warning-100 text-warning-800 border-warning-200';
      case 'HIGH': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'CRITICAL': return 'bg-danger-100 text-danger-800 border-danger-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getRiskIcon = (level: string) => {
    if (level === 'HIGH' || level === 'CRITICAL') {
      return <AlertTriangle className="w-4 h-4" />;
    }
    return null;
  };

  return (
    <div
      onClick={onClick}
      className={`
        p-4 rounded-lg border-2 cursor-pointer transition-all duration-200
        ${isSelected 
          ? 'border-primary-500 bg-primary-50 shadow-lg' 
          : 'border-gray-200 bg-white hover:shadow-md hover:border-gray-300'
        }
      `}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <User className="w-5 h-5 text-gray-600" />
          <span className="font-semibold text-gray-900">{patient.Patient_ID}</span>
        </div>
        <div className={`px-2 py-1 rounded-full text-xs font-medium border flex items-center space-x-1 ${getRiskColor(prediction.riskLevel)}`}>
          {getRiskIcon(prediction.riskLevel)}
          <span>{prediction.riskLevel}</span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="flex items-center space-x-2 text-gray-600">
          <span>{patient.Gender}, {patient.Age}y</span>
        </div>
        <div className="flex items-center space-x-2 text-gray-600">
          <Clock className="w-4 h-4" />
          <span>ICU {patient.ICULOS}h</span>
        </div>
        <div className="flex items-center space-x-2 text-gray-600">
          <MapPin className="w-4 h-4" />
          <span>Unit {patient.Unit1 ? '1' : patient.Unit2 ? '2' : 'General'}</span>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-gray-200">
        <div className="flex justify-between items-center text-sm">
          <span className="text-gray-600">Risk Probability</span>
          <span className="font-semibold">{(prediction.probability * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
          <div
            className={`h-2 rounded-full transition-all duration-300 ${
              prediction.riskLevel === 'LOW' ? 'bg-success-500' :
              prediction.riskLevel === 'MODERATE' ? 'bg-warning-500' :
              prediction.riskLevel === 'HIGH' ? 'bg-orange-500' :
              'bg-danger-500'
            }`}
            style={{ width: `${prediction.probability * 100}%` }}
          />
        </div>
      </div>

      {prediction.timeToSepsis && (
        <div className="mt-2 text-xs text-orange-600 font-medium">
          Est. onset: {prediction.timeToSepsis}h
        </div>
      )}
    </div>
  );
};