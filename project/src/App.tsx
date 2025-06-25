import React, { useState, useEffect } from 'react';
import { Activity, BarChart3, Users, Settings, AlertTriangle, Upload, Brain, Search } from 'lucide-react';
import { mockPatients } from './data/mockPatients';
import { simulateMLPrediction } from './utils/mlSimulation';
import { PatientCard } from './components/PatientCard';
import { VitalSignsPanel } from './components/VitalSignsPanel';
import { LabValuesPanel } from './components/LabValuesPanel';
import { PredictionChart } from './components/PredictionChart';
import { InterventionPanel } from './components/InterventionPanel';
import { ModelPerformance } from './components/ModelPerformance';
import { EnhancedDatasetManager } from './components/EnhancedDatasetManager';
import { AIInsightsPanel } from './components/AIInsightsPanel';
import { PatientSearch } from './components/PatientSearch';
import { PatientAnalysisReport } from './components/PatientAnalysisReport';
import { ThresholdAlert } from './components/ThresholdAlert';
import { DynamicSettings } from './components/DynamicSettings';
import { DatasetInfo, DatasetRow, ModelMetrics, PatientAnalysisReport as ReportType, ThresholdSettings } from './types/dataset';
import { EnhancedSepsisMLModel } from './utils/enhancedMLModel';
import { PatientData } from './types/patient';

type Tab = 'dashboard' | 'performance' | 'patients' | 'datasets' | 'ai-insights' | 'settings';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('dashboard');
  const [selectedPatient, setSelectedPatient] = useState(mockPatients[0]);
  const [currentHour, setCurrentHour] = useState(12);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>();
  const [currentDataset, setCurrentDataset] = useState<DatasetRow[]>([]);
  const [trainedModel, setTrainedModel] = useState<EnhancedSepsisMLModel | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [patientReport, setPatientReport] = useState<ReportType | null>(null);
  const [showThresholdAlert, setShowThresholdAlert] = useState(false);
  const [thresholdViolations, setThresholdViolations] = useState<any[]>([]);
  const [isTrainingInBackground, setIsTrainingInBackground] = useState(false);
  const [patientTimelines, setPatientTimelines] = useState<{ [id: string]: PatientData[] }>({});
  const [allPatients, setAllPatients] = useState<PatientData[]>([]);
  const [patientsSearchTerm, setPatientsSearchTerm] = useState('');
  
  // Get current patient data for the selected hour
  const currentPatientTimeline = selectedPatientId && patientTimelines[selectedPatientId] ? patientTimelines[selectedPatientId] : [];
  const currentPatientData = currentPatientTimeline.length > 0 ? (currentPatientTimeline[currentHour - 1] || currentPatientTimeline[0]) : null;
  const prediction = trainedModel && trainedModel.isModelTrained() && currentPatientData
    ? trainedModel.predictWithUncertainty(currentPatientData)
    : { riskLevel: 'LOW', probability: 0, confidence: 0 };
  
  // Get all current patients with their latest predictions
  const allPatientsWithPredictions = Object.keys(patientTimelines).map(id => {
    const timeline = patientTimelines[id];
    const latestData = timeline && timeline.length > 0 ? timeline[timeline.length - 1] : null;
    const pred = trainedModel && trainedModel.isModelTrained() && latestData
      ? trainedModel.predictWithUncertainty(latestData)
      : { riskLevel: 'LOW', probability: 0, confidence: 0 };
    return { patient: latestData, prediction: pred };
  }).filter(p => p.patient);

  // Count high-risk patients  
  const highRiskCount = allPatientsWithPredictions.filter(
    p => p.prediction.riskLevel === 'HIGH' || p.prediction.riskLevel === 'CRITICAL'
  ).length;

  // Get all patient IDs from datasets
  const allPatientIds = datasets.reduce((ids: string[], dataset) => {
    return [...ids, ...dataset.patientIds];
  }, []);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      if (currentHour < 24) {
        setCurrentHour(prev => prev + 1);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [currentHour]);

  // Build patient timelines from uploaded dataset
  useEffect(() => {
    if (currentDataset.length > 0) {
      const timelines: { [id: string]: PatientData[] } = {};
      const patients: PatientData[] = [];
      currentDataset.forEach(row => {
        const id = String(row.Patient_ID || row.PatientID || row.patient_id || '');
        if (!id) return;
        const hour = Number(row.Hour || row.Time || 1);
        const patient: PatientData = {
          Patient_ID: id,
          Hour: hour,
          Age: Number(row.Age || 0),
          Gender: (row.Gender === 'M' || row.Gender === 'F') ? row.Gender : 'M',
          Unit1: Boolean(row.Unit1),
          Unit2: Boolean(row.Unit2),
          HospAdmTime: Number(row.HospAdmTime || 0),
          ICULOS: Number(row.ICULOS || 0),
          SepsisLabel: Number(row.SepsisLabel || row.sepsislabel || 0),
          vitals: {
            HR: Number(row.HR || 0),
            O2Sat: Number(row.O2Sat || 0),
            Temp: Number(row.Temp || 0),
            SBP: Number(row.SBP || 0),
            MAP: Number(row.MAP || 0),
            DBP: Number(row.DBP || 0),
            Resp: Number(row.Resp || 0),
            EtCO2: Number(row.EtCO2 || 0),
          },
          labs: {
            BaseExcess: Number(row.BaseExcess || 0),
            HCO3: Number(row.HCO3 || 0),
            FiO2: Number(row.FiO2 || 0),
            pH: Number(row.pH || 0),
            PaCO2: Number(row.PaCO2 || 0),
            SaO2: Number(row.SaO2 || 0),
            AST: Number(row.AST || 0),
            BUN: Number(row.BUN || 0),
            Alkalinephos: Number(row.Alkalinephos || 0),
            Calcium: Number(row.Calcium || 0),
            Chloride: Number(row.Chloride || 0),
            Creatinine: Number(row.Creatinine || 0),
            Bilirubin_direct: Number(row.Bilirubin_direct || 0),
            Glucose: Number(row.Glucose || 0),
            Lactate: Number(row.Lactate || 0),
            Magnesium: Number(row.Magnesium || 0),
            Phosphate: Number(row.Phosphate || 0),
            Potassium: Number(row.Potassium || 0),
            Bilirubin_total: Number(row.Bilirubin_total || 0),
            TroponinI: Number(row.TroponinI || 0),
            Hct: Number(row.Hct || 0),
            Hgb: Number(row.Hgb || 0),
            PTT: Number(row.PTT || 0),
            WBC: Number(row.WBC || 0),
            Fibrinogen: Number(row.Fibrinogen || 0),
            Platelets: Number(row.Platelets || 0),
          },
        };
        if (!timelines[id]) timelines[id] = [];
        timelines[id].push(patient);
        patients.push(patient);
      });
      // Sort each timeline by hour
      Object.keys(timelines).forEach(id => {
        timelines[id].sort((a, b) => a.Hour - b.Hour);
      });
      setPatientTimelines(timelines);
      setAllPatients(patients);
    }
  }, [currentDataset]);

  const handleDatasetUpload = async (data: DatasetRow[], info: DatasetInfo) => {
    setDatasets(prev => [...prev, info]);
    setSelectedDatasetId(info.id);
    setCurrentDataset(data);

    // Start background model training
    if (!trainedModel) {
      setTrainedModel(new EnhancedSepsisMLModel());
    }
    
    setIsTrainingInBackground(true);
    try {
      const model = trainedModel || new EnhancedSepsisMLModel();
      const metrics = await model.trainModelInBackground(data);
      setModelMetrics(metrics);
      setTrainedModel(model);
    } catch (error) {
      console.error('Background training failed:', error);
    } finally {
      setIsTrainingInBackground(false);
    }
  };

  const handleDatasetDelete = (datasetId: string) => {
    setDatasets(prev => prev.filter(d => d.id !== datasetId));
    if (selectedDatasetId === datasetId) {
      setSelectedDatasetId(undefined);
      setCurrentDataset([]);
    }
  };

  const handleDatasetSelect = (datasetId: string) => {
    setSelectedDatasetId(datasetId);
    const dataset = datasets.find(d => d.id === datasetId);
    if (dataset) {
      // In a real app, you'd load the actual dataset data here
      // For now, we'll keep the current dataset
    }
  };

  const handlePatientSelect = (patientId: string) => {
    setSelectedPatientId(patientId);
    setPatientReport(null);
  };

  const handleStartDiagnosis = (patientId: string) => {
    if (!trainedModel || !trainedModel.isModelTrained()) {
      alert('Please upload and train a dataset first');
      return;
    }
    const timeline = patientTimelines[patientId];
    if (!timeline || timeline.length === 0) {
      alert('Patient not found in dataset');
      return;
    }
    const latestPatientData = timeline[currentHour - 1] || timeline[0];
    try {
      const report = trainedModel.analyzePatient(latestPatientData, patientId);
      setPatientReport(report);
      if (report.thresholdViolations.length > 0) {
        setThresholdViolations(report.thresholdViolations);
        setShowThresholdAlert(true);
      }
      setActiveTab('dashboard');
    } catch (error) {
      console.error('Diagnosis failed:', error);
      alert('Failed to generate diagnosis. Please try again.');
    }
  };

  const handleThresholdsUpdate = (newThresholds: ThresholdSettings) => {
    if (trainedModel) {
      trainedModel.updateThresholds(newThresholds);
    }
  };

  const navigation = [
    { id: 'dashboard' as Tab, name: 'Patient Dashboard', icon: Activity },
    { id: 'datasets' as Tab, name: 'Dataset Manager', icon: Upload },
    { id: 'performance' as Tab, name: 'Model Performance', icon: BarChart3 },
    { id: 'patients' as Tab, name: 'All Patients', icon: Users },
    { id: 'settings' as Tab, name: 'Settings', icon: Settings }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-primary-600 to-purple-600 rounded-lg">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">AI-Powered Sepsis Response Agent</h1>
              <p className="text-sm text-gray-600">Enhanced ML-Powered Early Detection & Clinical Intelligence System</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-600">
              ICU Time: Hour {currentHour}
            </div>
            {isTrainingInBackground && (
              <div className="px-3 py-2 bg-primary-100 text-primary-800 rounded-lg">
                <span className="font-medium">Training Model...</span>
              </div>
            )}
            {trainedModel && trainedModel.isModelTrained() && (
              <div className="px-3 py-2 bg-success-100 text-success-800 rounded-lg">
                <span className="font-medium">Enhanced Model Active</span>
              </div>
            )}
            {highRiskCount > 0 && (
              <div className="flex items-center space-x-2 px-3 py-2 bg-danger-100 text-danger-800 rounded-lg">
                <AlertTriangle className="w-4 h-4" />
                <span className="font-medium">{highRiskCount} High Risk Patient{highRiskCount !== 1 ? 's' : ''}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar Navigation */}
        <nav className="w-64 bg-white border-r border-gray-200 min-h-screen p-4">
          <div className="space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors ${
                    activeTab === item.id
                      ? 'bg-primary-100 text-primary-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.name}</span>
                </button>
              );
            })}
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              <PatientSearch
                patientIds={Object.keys(patientTimelines)}
                onPatientSelect={handlePatientSelect}
                onStartDiagnosis={handleStartDiagnosis}
                selectedPatientId={selectedPatientId}
              />
              {patientReport && (
                <>
                  <PatientAnalysisReport report={patientReport} />
                  <AIInsightsPanel 
                    patientData={currentPatientData}
                    prediction={prediction}
                    dataset={currentDataset}
                  />
                </>
              )}
              {currentPatientData && (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-gray-900">
                      Patient: {currentPatientData.Patient_ID}
                    </h2>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                      prediction.riskLevel === 'LOW' ? 'bg-success-100 text-success-800' :
                      prediction.riskLevel === 'MODERATE' ? 'bg-warning-100 text-warning-800' :
                      prediction.riskLevel === 'HIGH' ? 'bg-orange-100 text-orange-800' :
                      prediction.riskLevel === 'UNCERTAIN' ? 'bg-gray-100 text-gray-800' :
                      'bg-danger-100 text-danger-800'
                    }`}>
                      {prediction.riskLevel} RISK - {(prediction.probability * 100).toFixed(1)}%
                      {prediction.confidence && ` (${(prediction.confidence * 100).toFixed(0)}% confidence)`}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2">
                      <VitalSignsPanel vitals={currentPatientData.vitals} />
                    </div>
                    <div>
                      <LabValuesPanel labs={currentPatientData.labs} />
                    </div>
                  </div>
                </div>
              )}
              {currentPatientTimeline.length > 0 && (
                <PredictionChart 
                  patientTimeline={currentPatientTimeline}
                  predictWithModel={trainedModel && trainedModel.isModelTrained() ? (p => trainedModel.predictWithUncertainty(p)) : undefined}
                />
              )}
              {currentPatientData && (
                <InterventionPanel patientId={currentPatientData.Patient_ID} />
              )}
            </div>
          )}

          {activeTab === 'datasets' && (
            <EnhancedDatasetManager
              datasets={datasets}
              onDatasetUpload={handleDatasetUpload}
              onDatasetDelete={handleDatasetDelete}
              selectedDatasetId={selectedDatasetId}
              onDatasetSelect={handleDatasetSelect}
            />
          )}

          {activeTab === 'performance' && (
            <ModelPerformance metrics={modelMetrics} />
          )}

          {activeTab === 'patients' && (
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-gray-900">All Patients Monitor</h2>
              <input
                type="text"
                placeholder="Search patient by ID..."
                value={patientsSearchTerm}
                onChange={e => setPatientsSearchTerm(e.target.value)}
                className="mb-4 w-full p-2 border rounded"
              />
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {allPatientsWithPredictions
                  .filter(({ patient }) =>
                    !patientsSearchTerm || (patient && patient.Patient_ID.toLowerCase().includes(patientsSearchTerm.toLowerCase()))
                  )
                  .map(({ patient, prediction }) => (
                    <PatientCard
                      key={patient.Patient_ID}
                      patient={patient}
                      prediction={prediction}
                      isSelected={selectedPatientId === patient.Patient_ID}
                      onClick={() => {
                        setSelectedPatientId(patient.Patient_ID);
                        setActiveTab('dashboard');
                      }}
                    />
                  ))}
              </div>
            </div>
          )}

          {activeTab === 'settings' && (
            <DynamicSettings
              thresholds={trainedModel?.getThresholds() || {}}
              onThresholdsUpdate={handleThresholdsUpdate}
            />
          )}
        </main>
      </div>

      {/* Threshold Alert Modal */}
      {showThresholdAlert && (
        <ThresholdAlert
          violations={thresholdViolations}
          patientId={selectedPatientId}
          onClose={() => setShowThresholdAlert(false)}
        />
      )}
    </div>
  );
}

export default App;