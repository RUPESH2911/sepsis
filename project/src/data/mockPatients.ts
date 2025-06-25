import { PatientData } from '../types/patient';

export const generateMockPatient = (patientId: string, hour: number): PatientData => {
  const sepsisRisk = Math.random();
  const isSeptic = sepsisRisk > 0.7;
  
  // Base normal values with some variation
  const baseHR = 70 + Math.random() * 30;
  const baseTemp = 36.5 + Math.random() * 1.5;
  const baseSBP = 120 + Math.random() * 20;
  const baseWBC = 7 + Math.random() * 6;
  const baseLactate = 1 + Math.random() * 2;
  
  // Adjust values based on sepsis risk
  const sepsisMultiplier = isSeptic ? 1.3 + Math.random() * 0.7 : 1;
  
  return {
    Patient_ID: patientId,
    Hour: hour,
    Age: 45 + Math.floor(Math.random() * 40),
    Gender: Math.random() > 0.5 ? 'M' : 'F',
    Unit1: Math.random() > 0.7,
    Unit2: Math.random() > 0.8,
    HospAdmTime: Math.floor(Math.random() * 168), // Up to 7 days
    ICULOS: hour,
    SepsisLabel: isSeptic && hour > 10 ? 1 : 0,
    vitals: {
      HR: Math.round(baseHR * sepsisMultiplier),
      O2Sat: Math.round(98 - (sepsisMultiplier - 1) * 15),
      Temp: parseFloat((baseTemp + (sepsisMultiplier - 1) * 2).toFixed(1)),
      SBP: Math.round(baseSBP - (sepsisMultiplier - 1) * 30),
      MAP: Math.round((baseSBP - (sepsisMultiplier - 1) * 30) * 0.7),
      DBP: Math.round((baseSBP - (sepsisMultiplier - 1) * 30) * 0.6),
      Resp: Math.round(16 + (sepsisMultiplier - 1) * 8),
      EtCO2: Math.round(35 - (sepsisMultiplier - 1) * 5),
    },
    labs: {
      BaseExcess: parseFloat((-2 - (sepsisMultiplier - 1) * 3).toFixed(1)),
      HCO3: parseFloat((24 - (sepsisMultiplier - 1) * 4).toFixed(1)),
      FiO2: parseFloat((0.21 + (sepsisMultiplier - 1) * 0.3).toFixed(2)),
      pH: parseFloat((7.4 - (sepsisMultiplier - 1) * 0.1).toFixed(2)),
      PaCO2: parseFloat((40 + (sepsisMultiplier - 1) * 5).toFixed(1)),
      SaO2: parseFloat((98 - (sepsisMultiplier - 1) * 5).toFixed(1)),
      AST: Math.round(25 + (sepsisMultiplier - 1) * 75),
      BUN: Math.round(15 + (sepsisMultiplier - 1) * 25),
      Alkalinephos: Math.round(75 + (sepsisMultiplier - 1) * 50),
      Calcium: parseFloat((9.5 - (sepsisMultiplier - 1) * 1).toFixed(1)),
      Chloride: Math.round(102 + (sepsisMultiplier - 1) * 8),
      Creatinine: parseFloat((1.0 + (sepsisMultiplier - 1) * 1.5).toFixed(1)),
      Bilirubin_direct: parseFloat((0.2 + (sepsisMultiplier - 1) * 0.8).toFixed(1)),
      Glucose: Math.round(100 + (sepsisMultiplier - 1) * 80),
      Lactate: parseFloat((baseLactate * sepsisMultiplier).toFixed(1)),
      Magnesium: parseFloat((2.0 - (sepsisMultiplier - 1) * 0.5).toFixed(1)),
      Phosphate: parseFloat((3.5 - (sepsisMultiplier - 1) * 1).toFixed(1)),
      Potassium: parseFloat((4.0 + (sepsisMultiplier - 1) * 1).toFixed(1)),
      Bilirubin_total: parseFloat((1.0 + (sepsisMultiplier - 1) * 2).toFixed(1)),
      TroponinI: parseFloat((0.1 + (sepsisMultiplier - 1) * 2).toFixed(2)),
      Hct: parseFloat((42 - (sepsisMultiplier - 1) * 10).toFixed(1)),
      Hgb: parseFloat((14 - (sepsisMultiplier - 1) * 3).toFixed(1)),
      PTT: Math.round(30 + (sepsisMultiplier - 1) * 20),
      WBC: parseFloat((baseWBC * sepsisMultiplier).toFixed(1)),
      Fibrinogen: Math.round(300 + (sepsisMultiplier - 1) * 200),
      Platelets: Math.round(250 - (sepsisMultiplier - 1) * 100),
    }
  };
};

export const generatePatientTimeline = (patientId: string, hours: number = 24): PatientData[] => {
  return Array.from({ length: hours }, (_, i) => generateMockPatient(patientId, i + 1));
};

export const mockPatients = [
  'PT-001',
  'PT-002', 
  'PT-003',
  'PT-004',
  'PT-005'
].map(id => ({
  id,
  timeline: generatePatientTimeline(id, 24)
}));