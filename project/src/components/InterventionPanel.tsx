import React, { useState } from 'react';
import { Plus, Clock, CheckCircle, AlertCircle, Activity } from 'lucide-react';
import { Intervention } from '../types/patient';

interface InterventionPanelProps {
  patientId: string;
}

export const InterventionPanel: React.FC<InterventionPanelProps> = ({ patientId }) => {
  const [interventions, setInterventions] = useState<Intervention[]>([
    {
      id: '1',
      type: 'LAB_ORDER',
      name: 'Blood Culture x2',
      priority: 'URGENT',
      estimatedTime: 30,
      status: 'IN_PROGRESS',
      timestamp: new Date()
    },
    {
      id: '2',
      type: 'MEDICATION',
      name: 'Broad Spectrum Antibiotics',
      priority: 'URGENT',
      estimatedTime: 15,
      status: 'PENDING',
      timestamp: new Date()
    },
    {
      id: '3',
      type: 'LAB_ORDER',
      name: 'Lactate Level',
      priority: 'HIGH',
      estimatedTime: 20,
      status: 'COMPLETED',
      timestamp: new Date(Date.now() - 3600000)
    }
  ]);

  const [showAddForm, setShowAddForm] = useState(false);

  const quickInterventions = [
    { type: 'LAB_ORDER' as const, name: 'CBC with Differential', time: 25 },
    { type: 'LAB_ORDER' as const, name: 'Comprehensive Metabolic Panel', time: 25 },
    { type: 'LAB_ORDER' as const, name: 'Procalcitonin', time: 35 },
    { type: 'MEDICATION' as const, name: 'IV Fluid Resuscitation', time: 10 },
    { type: 'MONITORING' as const, name: 'Continuous Cardiac Monitor', time: 5 },
    { type: 'CONSULTATION' as const, name: 'Infectious Disease Consult', time: 60 }
  ];

  const addIntervention = (intervention: { type: Intervention['type']; name: string; time: number }) => {
    const newIntervention: Intervention = {
      id: Date.now().toString(),
      type: intervention.type,
      name: intervention.name,
      priority: 'HIGH',
      estimatedTime: intervention.time,
      status: 'PENDING',
      timestamp: new Date()
    };
    setInterventions([...interventions, newIntervention]);
  };

  const getStatusIcon = (status: Intervention['status']) => {
    switch (status) {
      case 'COMPLETED': return <CheckCircle className="w-4 h-4 text-success-600" />;
      case 'IN_PROGRESS': return <Activity className="w-4 h-4 text-primary-600 animate-pulse" />;
      case 'PENDING': return <Clock className="w-4 h-4 text-warning-600" />;
      default: return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
  };

  const getPriorityColor = (priority: Intervention['priority']) => {
    switch (priority) {
      case 'URGENT': return 'bg-danger-100 text-danger-800 border-danger-200';
      case 'HIGH': return 'bg-warning-100 text-warning-800 border-warning-200';
      case 'MEDIUM': return 'bg-primary-100 text-primary-800 border-primary-200';
      case 'LOW': return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getTypeIcon = (type: Intervention['type']) => {
    switch (type) {
      case 'LAB_ORDER': return '🧪';
      case 'MEDICATION': return '💊';
      case 'MONITORING': return '📊';
      case 'CONSULTATION': return '👨‍⚕️';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Clinical Interventions</h3>
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className="flex items-center space-x-2 px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          <Plus className="w-4 h-4" />
          <span>Add Intervention</span>
        </button>
      </div>

      {showAddForm && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-900 mb-3">Quick Interventions</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {quickInterventions.map((intervention, index) => (
              <button
                key={index}
                onClick={() => {
                  addIntervention(intervention);
                  setShowAddForm(false);
                }}
                className="flex items-center justify-between p-3 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left"
              >
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{getTypeIcon(intervention.type)}</span>
                  <span className="text-sm font-medium text-gray-900">{intervention.name}</span>
                </div>
                <span className="text-xs text-gray-500">{intervention.time}min</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-3">
        {interventions.map((intervention) => (
          <div
            key={intervention.id}
            className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-center space-x-3">
              <div className="text-xl">{getTypeIcon(intervention.type)}</div>
              <div>
                <div className="font-medium text-gray-900">{intervention.name}</div>
                <div className="text-sm text-gray-500">
                  {intervention.timestamp.toLocaleTimeString()} • Est. {intervention.estimatedTime}min
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getPriorityColor(intervention.priority)}`}>
                {intervention.priority}
              </span>
              <div className="flex items-center space-x-1">
                {getStatusIcon(intervention.status)}
                <span className="text-sm font-medium text-gray-700 capitalize">
                  {intervention.status.replace('_', ' ').toLowerCase()}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-primary-50 rounded-lg">
        <div className="flex items-center space-x-2 text-primary-800">
          <AlertCircle className="w-5 h-5" />
          <span className="font-medium">Sepsis Protocol Active</span>
        </div>
        <div className="mt-2 text-sm text-primary-700">
          Following institutional sepsis bundle guidelines. Target: Antibiotics within 1 hour, fluid resuscitation initiated.
        </div>
      </div>
    </div>
  );
};