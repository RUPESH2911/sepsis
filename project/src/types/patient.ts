export interface VitalSigns {
  HR: number;
  O2Sat: number;
  Temp: number;
  SBP: number;
  MAP: number;
  DBP: number;
  Resp: number;
  EtCO2: number;
}

export interface LabValues {
  BaseExcess: number;
  HCO3: number;
  FiO2: number;
  pH: number;
  PaCO2: number;
  SaO2: number;
  AST: number;
  BUN: number;
  Alkalinephos: number;
  Calcium: number;
  Chloride: number;
  Creatinine: number;
  Bilirubin_direct: number;
  Glucose: number;
  Lactate: number;
  Magnesium: number;
  Phosphate: number;
  Potassium: number;
  Bilirubin_total: number;
  TroponinI: number;
  Hct: number;
  Hgb: number;
  PTT: number;
  WBC: number;
  Fibrinogen: number;
  Platelets: number;
}

export interface PatientData {
  Patient_ID: string;
  Hour: number;
  Age: number;
  Gender: 'M' | 'F';
  Unit1: boolean;
  Unit2: boolean;
  HospAdmTime: number;
  ICULOS: number;
  SepsisLabel: number;
  vitals: VitalSigns;
  labs: LabValues;
}

export interface PredictionResult {
  probability: number;
  confidence: number;
  riskLevel: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  timeToSepsis?: number;
}

export interface Intervention {
  id: string;
  type: 'LAB_ORDER' | 'MEDICATION' | 'MONITORING' | 'CONSULTATION';
  name: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
  estimatedTime: number;
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED';
  timestamp: Date;
}