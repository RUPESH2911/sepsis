export interface DatasetInfo {
  id: string;
  name: string;
  uploadDate: Date;
  size: number;
  rowCount: number;
  columnCount: number;
  status: 'processing' | 'ready' | 'error';
  features: string[];
  targetColumn?: string;
  patientIds: string[];
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  confusionMatrix: number[][];
  featureImportance: { feature: string; importance: number }[];
  falsePositiveRate: number;
  falseNegativeRate: number;
}

export interface AIInsight {
  id: string;
  type: 'risk_assessment' | 'intervention_recommendation' | 'pattern_analysis' | 'clinical_alert' | 'uncertainty_warning';
  title: string;
  description: string;
  confidence: number;
  priority: 'low' | 'medium' | 'high' | 'critical' | 'uncertain';
  timestamp: Date;
  patientId?: string;
  recommendations?: string[];
}

export interface DatasetRow {
  [key: string]: string | number;
}

export interface ThresholdSettings {
  [key: string]: {
    min?: number;
    max?: number;
    critical?: number;
    enabled: boolean;
  };
}

export interface PatientAnalysisReport {
  patientId: string;
  overallRisk: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL' | 'UNCERTAIN';
  confidence: number;
  riskProbability: number;
  clinicalFindings: string[];
  recommendations: string[];
  thresholdViolations: Array<{
    parameter: string;
    value: number;
    threshold: number;
    severity: 'warning' | 'critical';
  }>;
  uncertaintyFactors: string[];
  treatmentPlan: string[];
  followUpActions: string[];
  timestamp: Date;
}