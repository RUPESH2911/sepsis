import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { PatientData } from '../types/patient';
import { simulateMLPrediction } from '../utils/mlSimulation';

interface PredictionChartProps {
  patientTimeline: PatientData[];
  predictWithModel?: (patient: PatientData) => { probability: number; riskLevel: string; confidence: number };
}

export const PredictionChart: React.FC<PredictionChartProps> = ({ patientTimeline, predictWithModel }) => {
  const statusMap = { LOW: 0, MODERATE: 1, HIGH: 2, CRITICAL: 3 };
  const statusLabels = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'];
  const chartData = patientTimeline.map(patient => {
    const prediction = predictWithModel ? predictWithModel(patient) : simulateMLPrediction(patient);
    return {
      hour: patient.Hour,
      status: statusMap[prediction.riskLevel as keyof typeof statusMap],
      riskLevel: prediction.riskLevel,
      probability: prediction.probability * 100,
      lactate: patient.labs.Lactate,
      wbc: patient.labs.WBC,
      heartRate: patient.vitals.HR,
      temp: patient.vitals.Temp,
      sbp: patient.vitals.SBP,
      dbp: patient.vitals.DBP,
      o2sat: patient.vitals.O2Sat,
      platelets: patient.labs.Platelets,
      creatinine: patient.labs.Creatinine,
      confidence: prediction.confidence * 100,
    };
  });

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Patient Status Timeline</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="hour" 
              stroke="#6b7280"
              fontSize={12}
              label={{ value: 'Hours', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              stroke="#6b7280"
              fontSize={12}
              label={{ value: 'Status', angle: -90, position: 'insideLeft' }}
              domain={[0, 3]}
              ticks={[0, 1, 2, 3]}
              tickFormatter={tick => statusLabels[tick]}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
              formatter={(value: any, name: string, props: any) => {
                if (name === 'status') return [statusLabels[value], 'Status'];
                if (typeof value === 'number') return [value.toFixed(1), name];
                return [value, name];
              }}
              labelFormatter={(label, payload) => {
                if (!payload || !payload.length) return `Hour: ${label}`;
                const d = payload[0].payload;
                return (
                  <div>
                    <div><b>Hour:</b> {d.hour}</div>
                    <div><b>Status:</b> {statusLabels[d.status]}</div>
                    <div><b>Probability:</b> {d.probability.toFixed(1)}%</div>
                    <div><b>Confidence:</b> {d.confidence.toFixed(1)}%</div>
                  </div>
                );
              }}
            />
            <Line 
              type="stepAfter" 
              dataKey="status" 
              stroke="#3b82f6" 
              strokeWidth={3}
              name="Patient Status"
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-1 bg-primary-500 rounded"></div>
          <span className="text-gray-600">Patient Status (LOW, MODERATE, HIGH, CRITICAL)</span>
        </div>
      </div>
    </div>
  );
};